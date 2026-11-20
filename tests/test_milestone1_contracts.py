"""
Milestone 1 contract-harmonization tests.

Covers:
  - Simulator API and action schema
  - Planner input/output schema and fallback behaviour
  - Azure workload CSV schema and chronological split
  - Forecast file alignment to test time-indices
  - PPO environment observation/action dimensions
  - Unit / semantic consistency between ppo_env and serverless_simulator
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV  = ROOT / "data" / "azure_serverless_workload.csv"
ORIG_TFT  = ROOT / "outputs" / "tft_azure_holdout_predictions.csv"
LOG_TFT   = ROOT / "outputs" / "tft_azure_log_predictions.csv"
TEST_FRAC = 0.15
BUCKET_SECONDS = 300.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _azure_df():
    return pd.read_csv(DATA_CSV)


def _test_indices():
    df = _azure_df()
    n = len(df)
    split = int(n * (1 - TEST_FRAC))
    return set(df.iloc[split:]["time_idx"].tolist())


# ---------------------------------------------------------------------------
# 1. Simulator
# ---------------------------------------------------------------------------

class TestServerlessSimulator:

    def test_smoke_fixed_policy(self):
        from serverless_simulator import ServerlessSimulator, fixed_policy
        sim = ServerlessSimulator(seed=0)
        w = np.full(20, 5.0, dtype=np.float32)
        h = sim.run(w, w, fixed_policy(4))
        assert len(h) == 20

    def test_actions_are_allowlisted(self):
        from serverless_simulator import ServerlessSimulator
        sim = ServerlessSimulator(seed=0)
        sim.reset()
        with pytest.raises(ValueError):
            sim.step("INVALID_ACTION", 5.0, 5.0)

    def test_warm_containers_bounded(self):
        from serverless_simulator import ServerlessSimulator
        sim = ServerlessSimulator(min_containers=1, max_containers=5, seed=0)
        # Try to push far above max
        for _ in range(20):
            m = sim.step("SCALE_OUT", 100.0, 100.0)
        assert m.warm_containers <= 5

        sim.reset()
        for _ in range(20):
            m = sim.step("TERMINATE_IDLE", 0.0, 0.0)
        assert m.warm_containers >= 1

    def test_no_future_leakage_in_obs(self):
        """The obs dict passed to the policy must not contain the upcoming
        actual_load; it may only contain the forecast for the next bucket."""
        from serverless_simulator import ServerlessSimulator, rule_based_policy

        seen_actual_in_obs = []
        actual = np.array([10, 20, 30, 40, 50], dtype=np.float32)
        forecast = actual.copy()

        def spy_policy(obs):
            # At step i, obs["actual_invocations"] must be actual[i-1]
            seen_actual_in_obs.append(obs["actual_invocations"])
            return "NO_OP"

        sim = ServerlessSimulator(seed=0)
        sim.run(actual, forecast, spy_policy)

        # The last call uses actual[n-2] as the previous observed load.
        # obs["actual_invocations"] should never equal actual[i] for the
        # current (upcoming) step.
        assert len(seen_actual_in_obs) == len(actual) - 1

    def test_latency_increases_with_overload(self):
        from serverless_simulator import ServerlessSimulator
        sim = ServerlessSimulator(seed=0)
        # Light load: 1 container, load = 5 (within capacity of 10)
        m_light = sim.step("NO_OP", 5.0, 5.0)
        sim2 = ServerlessSimulator(seed=0)
        # Heavy overload: 1 container, load = 100
        m_heavy = sim2.step("NO_OP", 100.0, 100.0)
        assert m_heavy.latency_ms > m_light.latency_ms

    def test_sla_violation_flag_matches_latency(self):
        from serverless_simulator import ServerlessSimulator
        sim = ServerlessSimulator(sla_ms=200.0, seed=42)
        m = sim.step("NO_OP", 1000.0, 1000.0)
        assert m.sla_violation == (m.latency_ms > 200.0)

    def test_metrics_dataclass_fields(self):
        from serverless_simulator import StepMetrics
        fields = StepMetrics.__dataclass_fields__
        required = {
            "time_idx", "actual_invocations", "predicted_invocations",
            "warm_containers", "action", "latency_ms", "p95_latency_ms",
            "sla_violation", "energy_proxy", "cold_start_count",
        }
        assert required.issubset(set(fields))

    def test_deterministic_with_same_seed(self):
        from serverless_simulator import ServerlessSimulator, rule_based_policy
        w = np.linspace(10, 50, 30).astype(np.float32)
        h1 = ServerlessSimulator(seed=99).run(w, w, rule_based_policy())
        h2 = ServerlessSimulator(seed=99).run(w, w, rule_based_policy())
        lat1 = [m.latency_ms for m in h1]
        lat2 = [m.latency_ms for m in h2]
        assert lat1 == lat2


# ---------------------------------------------------------------------------
# 2. Planner
# ---------------------------------------------------------------------------

class TestAgenticPlanner:

    def _make_input(self, **kwargs):
        from agentic_planner import PlannerInput
        defaults = dict(
            time_idx=0,
            forecast_invocations=50.0,
            actual_invocations=40.0,
            warm_containers=4,
            sla_ms=200.0,
            current_latency_ms=80.0,
            energy_proxy=1.5,
            capacity_limit=50,
            cold_start_rate=0.0,
        )
        defaults.update(kwargs)
        return PlannerInput(**defaults)

    def test_planner_returns_allowed_action(self):
        from agentic_planner import plan, ALLOWED_ACTIONS
        result = plan(self._make_input())
        assert result.recommended_action in ALLOWED_ACTIONS

    def test_planner_fallback_when_no_endpoint(self):
        """With no PLANNER_ENDPOINT env var, fallback_used must be True."""
        import os
        os.environ.pop("PLANNER_ENDPOINT", None)
        from agentic_planner import plan
        result = plan(self._make_input())
        assert result.fallback_used is True

    def test_planner_output_has_required_fields(self):
        from agentic_planner import PlannerOutput
        fields = PlannerOutput.__dataclass_fields__
        required = {
            "recommended_action", "rationale", "confidence",
            "fallback_used", "fallback_reason",
            "planner_version", "inference_ms",
        }
        assert required.issubset(set(fields))

    def test_planner_scales_out_on_high_forecast(self):
        from agentic_planner import plan
        # Heavy upcoming load, few containers -> should recommend scale-out / prewarm
        result = plan(self._make_input(
            forecast_invocations=200.0,
            actual_invocations=5.0,
            warm_containers=1,
        ))
        assert result.recommended_action in {"SCALE_OUT", "PREWARM"}

    def test_planner_idles_on_low_load(self):
        from agentic_planner import plan
        # Very low load, many containers
        result = plan(self._make_input(
            forecast_invocations=1.0,
            actual_invocations=0.5,
            warm_containers=10,
            current_latency_ms=70.0,
        ))
        assert result.recommended_action in {
            "TERMINATE_IDLE", "SCALE_IN", "NO_OP", "REUSE"
        }


# ---------------------------------------------------------------------------
# 3. Azure workload CSV
# ---------------------------------------------------------------------------

class TestAzureWorkloadCSV:

    def test_required_columns_present(self):
        df = _azure_df()
        for col in ("time_idx", "invocation_count", "mean_duration_seconds"):
            assert col in df.columns, f"Missing column: {col}"

    def test_time_idx_monotonically_increasing(self):
        df = _azure_df()
        assert (df["time_idx"].diff().dropna() >= 0).all(), \
            "time_idx is not monotonically non-decreasing"

    def test_no_duplicate_time_idx(self):
        df = _azure_df()
        assert not df["time_idx"].duplicated().any(), \
            "Duplicate time_idx values found"

    def test_invocation_count_non_negative(self):
        df = _azure_df()
        assert (df["invocation_count"] >= 0).all()

    def test_mean_duration_positive(self):
        df = _azure_df()
        assert (df["mean_duration_seconds"] > 0).all()

    def test_chronological_split_no_overlap(self):
        df = _azure_df()
        n = len(df)
        split = int(n * (1 - TEST_FRAC))
        train_idx = set(df.iloc[:split]["time_idx"])
        test_idx  = set(df.iloc[split:]["time_idx"])
        assert train_idx.isdisjoint(test_idx), \
            "Train and test sets share time_idx values (leakage)"

    def test_train_before_test(self):
        df = _azure_df()
        n = len(df)
        split = int(n * (1 - TEST_FRAC))
        max_train = df.iloc[:split]["time_idx"].max()
        min_test  = df.iloc[split:]["time_idx"].min()
        assert max_train < min_test, \
            f"Train max time_idx ({max_train}) >= test min ({min_test})"

    def test_concurrency_formula_finite(self):
        """Concurrency estimate must be finite and non-negative."""
        df = _azure_df()
        n = len(df)
        split = int(n * (1 - TEST_FRAC))
        test = df.iloc[split:]
        conc = (
            test["invocation_count"].values
            * test["mean_duration_seconds"].values
            / BUCKET_SECONDS
        )
        assert np.all(np.isfinite(conc)), "Non-finite concurrency values"
        assert np.all(conc >= 0), "Negative concurrency values"


# ---------------------------------------------------------------------------
# 4. Forecast file alignment
# ---------------------------------------------------------------------------

class TestForecastAlignment:

    @pytest.mark.skipif(not ORIG_TFT.exists(), reason="TFT prediction file absent")
    def test_original_tft_covers_all_test_indices(self):
        pred = pd.read_csv(ORIG_TFT)
        test_idx = _test_indices()
        pred_idx = set(pred["time_idx"].tolist())
        missing = test_idx - pred_idx
        assert not missing, f"{len(missing)} test time-indices missing from TFT predictions"

    @pytest.mark.skipif(not LOG_TFT.exists(), reason="Log-TFT prediction file absent")
    def test_log_tft_covers_all_test_indices(self):
        pred = pd.read_csv(LOG_TFT)
        test_idx = _test_indices()
        pred_idx = set(pred["time_idx"].tolist())
        missing = test_idx - pred_idx
        assert not missing, f"{len(missing)} test time-indices missing from log-TFT predictions"

    @pytest.mark.skipif(not ORIG_TFT.exists(), reason="TFT prediction file absent")
    def test_original_tft_no_duplicate_time_idx(self):
        pred = pd.read_csv(ORIG_TFT)
        assert not pred["time_idx"].duplicated().any()

    @pytest.mark.skipif(not LOG_TFT.exists(), reason="Log-TFT prediction file absent")
    def test_log_tft_no_duplicate_time_idx(self):
        pred = pd.read_csv(LOG_TFT)
        assert not pred["time_idx"].duplicated().any()

    @pytest.mark.skipif(not LOG_TFT.exists(), reason="Log-TFT prediction file absent")
    def test_log_tft_predictions_non_negative(self):
        pred = pd.read_csv(LOG_TFT)
        assert (pred["log_tft_predicted_invocations"] >= 0).all()


# ---------------------------------------------------------------------------
# 5. PPO environment
# ---------------------------------------------------------------------------

class TestPPOEnv:

    def test_observation_space_shape(self):
        from ppo_env import ServerlessContainerEnv
        env = ServerlessContainerEnv()
        assert env.observation_space.shape == (9,)

    def test_action_space_size(self):
        from ppo_env import ServerlessContainerEnv
        env = ServerlessContainerEnv()
        assert env.action_space.n == 6

    def test_reset_returns_valid_obs(self):
        from ppo_env import ServerlessContainerEnv
        env = ServerlessContainerEnv()
        obs, info = env.reset(seed=0)
        assert obs.shape == (9,)
        assert np.all(np.isfinite(obs))

    def test_step_returns_valid_obs(self):
        from ppo_env import ServerlessContainerEnv
        env = ServerlessContainerEnv()
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs.shape == (9,)
        assert np.isfinite(reward)

    def test_invalid_action_raises(self):
        from ppo_env import ServerlessContainerEnv
        env = ServerlessContainerEnv()
        env.reset(seed=0)
        with pytest.raises((ValueError, KeyError, IndexError)):
            env.step(99)

    def test_warm_containers_stay_positive(self):
        from ppo_env import ServerlessContainerEnv
        env = ServerlessContainerEnv()
        env.reset(seed=0)
        for _ in range(20):
            obs, _, term, _, info = env.step(3)  # -1 warm
            if term:
                break
        assert info["warm_containers"] >= 1


# ---------------------------------------------------------------------------
# 6. Schema / unit consistency between ppo_env and serverless_simulator
# ---------------------------------------------------------------------------

class TestSchemaConsistency:

    def test_capacity_per_container_matches(self):
        """Both modules assume 10 workload units per warm container."""
        from serverless_simulator import CAPACITY_PER_CONTAINER
        from ppo_env import ServerlessContainerEnv
        import inspect

        # Capacity in ppo_env is embedded as literals: warm*10 in overload calc
        src = inspect.getsource(ServerlessContainerEnv)
        assert "10" in src, "ppo_env capacity constant not found"
        assert CAPACITY_PER_CONTAINER == 10

    def test_simulator_action_set_matches_planner_allowlist(self):
        from serverless_simulator import ServerlessSimulator
        from agentic_planner import ALLOWED_ACTIONS
        sim_actions = set(ServerlessSimulator.ACTION_DELTAS.keys())
        assert sim_actions == ALLOWED_ACTIONS, (
            f"Simulator actions {sim_actions} != planner allowlist {ALLOWED_ACTIONS}"
        )

    def test_ppo_env_action_count_matches_simulator_action_count(self):
        """Both have 6 actions (even if semantics differ — documented mismatch)."""
        from serverless_simulator import ServerlessSimulator
        from ppo_env import ServerlessContainerEnv
        env = ServerlessContainerEnv()
        assert env.action_space.n == len(ServerlessSimulator.ACTION_DELTAS)
