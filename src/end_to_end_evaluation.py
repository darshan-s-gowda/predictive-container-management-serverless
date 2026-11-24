
from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from typing import Callable, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

DATA_CSV = ROOT / "data" / "azure_serverless_workload.csv"

ORIGINAL_TFT_PATH = OUT / "tft_azure_holdout_predictions.csv"
LOG_TFT_PATH = OUT / "tft_azure_log_predictions.csv"

SLA_MS = 200.0
N_SEEDS = 10
TEST_FRAC = 0.15
BUCKET_SECONDS = 300.0


# ---------------------------------------------------------
# 1. Load Azure workload
# ---------------------------------------------------------

def load_azure_workload() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)

    required = {
        "time_idx",
        "invocation_count",
        "mean_duration_seconds",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Azure workload CSV is missing columns: {missing}"
        )

    df = df[
        [
            "time_idx",
            "invocation_count",
            "mean_duration_seconds",
        ]
    ].copy()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna()
        .sort_values("time_idx")
        .drop_duplicates("time_idx", keep="last")
        .reset_index(drop=True)
    )

    df = df[
        (df["invocation_count"] >= 0)
        & (df["mean_duration_seconds"] > 0)
    ].reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid Azure workload rows found.")

    return df


# ---------------------------------------------------------
# 2. Load forecasts and align by timestamp
# ---------------------------------------------------------

def load_aligned_prediction(
    test_df: pd.DataFrame,
    file_path: Path,
    prediction_column: str,
) -> np.ndarray:

    if not file_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {file_path}"
        )

    pred_df = pd.read_csv(file_path)

    required = {"time_idx", prediction_column}
    missing = required - set(pred_df.columns)

    if missing:
        raise ValueError(
            f"{file_path.name} is missing columns: {missing}"
        )

    pred_df = pred_df[
        ["time_idx", prediction_column]
    ].copy()

    for col in pred_df.columns:
        pred_df[col] = pd.to_numeric(
            pred_df[col], errors="coerce"
        )

    pred_df = pred_df.dropna()

    if pred_df["time_idx"].duplicated().any():
        raise ValueError(
            f"Duplicate timestamps found in {file_path.name}"
        )

    aligned = test_df[["time_idx"]].merge(
        pred_df,
        on="time_idx",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    if aligned[prediction_column].isna().any():
        missing_count = int(
            aligned[prediction_column].isna().sum()
        )

        raise ValueError(
            f"{file_path.name}: {missing_count} missing "
            "evaluation forecasts. No padding or shifting allowed."
        )

    values = aligned[prediction_column].to_numpy(
        dtype=np.float64
    )

    return np.maximum(values, 0.0)


# ---------------------------------------------------------
# 3. Convert workload and forecasts into concurrency
# ---------------------------------------------------------

def build_azure_test(df_full: pd.DataFrame):

    n = len(df_full)
    split = int(n * (1 - TEST_FRAC))

    if split <= 0 or split >= n:
        raise ValueError("Invalid train/test split.")

    train_df = df_full.iloc[:split].copy()
    test_df = df_full.iloc[split:].reset_index(drop=True)

    # Use training-period duration only for forecasts.
    training_mean_duration = float(
        train_df["mean_duration_seconds"].mean()
    )

    if (
        not np.isfinite(training_mean_duration)
        or training_mean_duration <= 0
    ):
        raise ValueError("Invalid training mean duration.")

    print(
        "\nTraining-period mean execution duration: "
        f"{training_mean_duration:.4f} seconds"
    )

    # Actual concurrency estimate:
    # invocations * mean execution duration / bucket duration.
    actual_concurrency = (
        test_df["invocation_count"].to_numpy(dtype=np.float64)
        * test_df["mean_duration_seconds"].to_numpy(
            dtype=np.float64
        )
        / BUCKET_SECONDS
    )

    # Previous observed concurrency is the persistence forecast.
    previous_train_concurrency = (
        float(train_df.iloc[-1]["invocation_count"])
        * float(train_df.iloc[-1]["mean_duration_seconds"])
        / BUCKET_SECONDS
    )

    persistence = np.concatenate(
        [
            [previous_train_concurrency],
            actual_concurrency[:-1],
        ]
    )

    # TFT forecasts are invocation-count forecasts.
    # Convert them using training-period mean duration.
    original_tft_counts = load_aligned_prediction(
        test_df,
        ORIGINAL_TFT_PATH,
        "tft_prediction",
    )

    log_tft_counts = load_aligned_prediction(
        test_df,
        LOG_TFT_PATH,
        "log_tft_predicted_invocations",
    )

    original_tft_concurrency = (
        original_tft_counts
        * training_mean_duration
        / BUCKET_SECONDS
    )

    log_tft_concurrency = (
        log_tft_counts
        * training_mean_duration
        / BUCKET_SECONDS
    )

    forecasts = {
        "persistence": persistence.astype(np.float32),
        "original_tft": original_tft_concurrency.astype(
            np.float32
        ),
        "log_tft": log_tft_concurrency.astype(np.float32),
    }

    actual = actual_concurrency.astype(np.float32)

    lengths = {len(actual)}
    lengths.update(len(v) for v in forecasts.values())

    if len(lengths) != 1:
        raise ValueError(
            "Actual workload and forecast lengths do not match."
        )

    test_df["estimated_actual_concurrency"] = actual

    return actual, forecasts, test_df


# ---------------------------------------------------------
# 4. Policy definitions
# ---------------------------------------------------------

def reactive_rule_action(obs: dict) -> str:
    """
    Reactive controller using the previous observed concurrency.
    """

    latency = obs["latency_ms"]
    warm = obs["warm_containers"]
    actual = obs["actual_invocations"]
    capacity = warm * 10.0

    if latency > SLA_MS:
        return "SCALE_OUT"

    if (
        latency < SLA_MS * 0.65
        and actual < capacity - 10
        and warm > 1
    ):
        return "SCALE_IN"

    return "NO_OP"


def planner_action_from_obs(obs: dict) -> str:
    """
    Structured planner using previous observed state
    and the next-bucket concurrency forecast.
    """

    try:
        from agentic_planner import PlannerInput, plan

        warm = obs["warm_containers"]
        actual = obs["actual_invocations"]
        capacity = warm * 10.0

        overload = max(0.0, actual - capacity)

        cold_rate = min(
            1.0,
            overload / max(actual, 1.0),
        )

        inp = PlannerInput(
            time_idx=obs["time_idx"],
            forecast_invocations=obs["predicted_invocations"],
            actual_invocations=actual,
            warm_containers=warm,
            sla_ms=obs["sla_ms"],
            current_latency_ms=obs["latency_ms"],
            energy_proxy=obs["energy_proxy"],
            capacity_limit=50,
            cold_start_rate=cold_rate,
        )

        result = plan(inp)
        return result.recommended_action

    except Exception as exc:
        print(
            f"Planner fallback at step {obs['time_idx']}: {exc}"
        )
        return reactive_rule_action(obs)


# ---------------------------------------------------------
# 5. Run one policy
# ---------------------------------------------------------

def run_policy(
    policy_name: str,
    actual: np.ndarray,
    forecast: np.ndarray,
    policy_fn: Callable,
    seed: int,
) -> dict:

    from serverless_simulator import ServerlessSimulator

    sim = ServerlessSimulator(
        sla_ms=SLA_MS,
        initial_warm=4,
        seed=seed,
    )

    history = sim.run(
        actual,
        forecast,
        policy_fn,
    )

    frame = pd.DataFrame(
        [asdict(item) for item in history]
    )

    if frame.empty:
        raise RuntimeError(
            f"No simulation records produced for {policy_name}"
        )

    return {
        "policy": policy_name,
        "seed": seed,
        "steps": len(frame),

        "mean_latency_ms": round(
            frame["latency_ms"].mean(), 2
        ),
        # P95 across all per-bucket mean-latency observations.
        # This can legitimately be below the overall mean when
        # a small number of extreme spikes pull the mean upward.
        "p95_latency_ms": round(
            frame["latency_ms"].quantile(0.95), 2
        ),

        # Mean of the simulator's per-bucket P95 estimates.
        # This is a different metric from the temporal P95 above.
        "mean_bucket_p95_latency_ms": round(
            frame["p95_latency_ms"].mean(), 2
        ),

        "cold_start_load_proxy": round(
            frame["cold_start_count"].sum(), 2
        ),
        "cold_start_rate": round(
            frame["cold_start_proxy"].mean(), 4
        ),

        "sla_violations": int(
            frame["sla_violation"].sum()
        ),
        "sla_violation_rate": round(
            frame["sla_violation"].mean(), 4
        ),

        "avg_warm_containers": round(
            frame["warm_containers"].mean(), 2
        ),
        "peak_warm_containers": int(
            frame["warm_containers"].max()
        ),

        "avg_energy_proxy": round(
            frame["energy_proxy"].mean(), 4
        ),

        "action_noop": int(
            (frame["action"] == "NO_OP").sum()
        ),
        "action_prewarm": int(
            (frame["action"] == "PREWARM").sum()
        ),
        "action_scale_out": int(
            (frame["action"] == "SCALE_OUT").sum()
        ),
        "action_scale_in": int(
            (frame["action"] == "SCALE_IN").sum()
        ),
        "action_terminate": int(
            (frame["action"] == "TERMINATE_IDLE").sum()
        ),
        "action_reuse": int(
            (frame["action"] == "REUSE").sum()
        ),
    }


# ---------------------------------------------------------
# 6. Summary and comparison plot
# ---------------------------------------------------------

def print_summary(frame: pd.DataFrame):

    metrics = [
        "mean_latency_ms",
        "p95_latency_ms",
        "mean_bucket_p95_latency_ms",
        "cold_start_load_proxy",
        "sla_violation_rate",
        "avg_warm_containers",
        "avg_energy_proxy",
    ]

    summary = (
        frame.groupby("policy")[metrics]
        .agg(["mean", "std"])
        .round(3)
    )

    summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in summary.columns
    ]

    print(summary.to_string())

    return summary


def plot_comparison(
    frame: pd.DataFrame,
    policy_order: List[str],
):

    metrics = [
        ("mean_latency_ms", "Mean latency (ms)"),
        ("p95_latency_ms", "P95 latency (ms)"),
        ("sla_violation_rate", "SLA violation rate"),
        ("avg_energy_proxy", "Energy proxy (a.u.)"),
        ("avg_warm_containers", "Average warm containers"),
        ("cold_start_load_proxy", "Cold-start load proxy"),
    ]

    fig, axes = plt.subplots(
        2, 3,
        figsize=(18, 10),
    )

    axes = axes.flatten()

    for ax, (metric, label) in zip(axes, metrics):

        grouped = frame.groupby("policy")[metric]

        means = grouped.mean().reindex(policy_order)
        stds = grouped.std().reindex(policy_order).fillna(0)

        ax.bar(
            range(len(policy_order)),
            means.values,
            yerr=stds.values,
            capsize=3,
        )

        ax.set_xticks(range(len(policy_order)))

        ax.set_xticklabels(
            [
                name.replace(" + ", "\n+ ")
                for name in policy_order
            ],
            fontsize=7,
            rotation=25,
            ha="right",
        )

        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle(
        "Calibrated concurrency policy comparison "
        "(Azure workload, simulated metrics)",
        fontsize=13,
    )

    fig.tight_layout()

    path = OUT / "end_to_end_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"Saved comparison plot: {path}")


# ---------------------------------------------------------
# 7. Main evaluation
# ---------------------------------------------------------

def main():

    print("=" * 65)
    print("CALIBRATED END-TO-END SERVERLESS EVALUATION")
    print("Workload unit: estimated average concurrency")
    print("Latency and energy metrics: simulated proxies")
    print("=" * 65)

    df_full = load_azure_workload()

    actual, forecasts, test_df = build_azure_test(
        df_full
    )

    print(f"\nTotal workload buckets: {len(df_full)}")
    print(f"Evaluation buckets: {len(actual)}")

    print(
        f"Evaluation time range: "
        f"{test_df['time_idx'].iloc[0]} - "
        f"{test_df['time_idx'].iloc[-1]}"
    )

    print("\nEstimated actual concurrency statistics:")

    print(
        pd.Series(actual).describe(
            percentiles=[0.50, 0.95, 0.99]
        ).round(3).to_string()
    )

    print("\nForecasts aligned and converted to concurrency:")

    for name, values in forecasts.items():
        print(
            f"  {name}: {len(values)} predictions | "
            f"mean={np.mean(values):.3f}"
        )

    # ----------------------------------------------------------------
    # Load calibrated PPO model (trained on Azure concurrency scale)
    # ----------------------------------------------------------------
    CALIBRATED_PPO_PATH = OUT / "ppo_container_controller_calibrated.zip"
    _calibrated_ppo = None

    try:
        from stable_baselines3 import PPO as _SB3PPO
        from ppo_calibrated_env import ACTION_MAP as _PPO_ACTION_MAP
        _calibrated_ppo = _SB3PPO.load(str(CALIBRATED_PPO_PATH))
        print(f"\nLoaded calibrated PPO model: {CALIBRATED_PPO_PATH}")
    except FileNotFoundError:
        print(
            f"\nCalibrated PPO model not found at {CALIBRATED_PPO_PATH}. "
            "Run src/train_ppo_calibrated.py first."
        )
    except Exception as _ppo_exc:
        print(f"\nCould not load calibrated PPO: {_ppo_exc}")

    def ppo_calibrated_action(obs: dict) -> str:
        """
        Convert the simulator obs dict to the 11-dim PPO observation vector
        and query the calibrated PPO policy. Returns a simulator action string.

        Observation vector layout (must match ppo_calibrated_env.py exactly):
          0  prev_load_norm         / LOAD_NORM (50)
          1  next_forecast_norm     / LOAD_NORM
          2  warm_containers_norm   / 50
          3  utilisation            load / max(capacity, 1)
          4  latency_norm           latency / SLA_MS
          5  sla_risk               clip(latency/SLA_MS - 1, -1, 1)
          6  energy_norm            energy / ENERGY_NORM (10)
          7  cold_start_proxy       0.0 (not available in this obs dict)
          8  headroom_norm          (capacity - load) / LOAD_NORM, signed
          9  time_in_episode_norm   0.5 (unknown mid-episode)
         10  cooldown_norm          0.5 (unknown)

        Units: same calibrated concurrency as all other policies.
        """
        if _calibrated_ppo is None:
            return reactive_rule_action(obs)

        LOAD_NORM = 50.0
        ENERGY_NORM = 10.0

        prev_load = float(obs["actual_invocations"])
        next_fc   = float(obs["predicted_invocations"])
        warm      = int(obs["warm_containers"])
        latency   = float(obs["latency_ms"])
        energy    = float(obs["energy_proxy"])

        capacity    = warm * 10.0
        utilisation = prev_load / max(capacity, 1.0)
        headroom    = (capacity - prev_load) / LOAD_NORM
        sla_risk    = float(np.clip(latency / SLA_MS - 1.0, -1.0, 1.0))

        obs_vec = np.array([
            prev_load / LOAD_NORM,
            next_fc   / LOAD_NORM,
            warm / 50.0,
            float(np.clip(utilisation, 0.0, 3.0)),
            latency / SLA_MS,
            sla_risk,
            energy / ENERGY_NORM,
            0.0,            # cold_start_proxy unavailable here
            float(np.clip(headroom, -3.0, 3.0)),
            0.5,            # time_in_episode_norm (mid-episode estimate)
            0.5,            # cooldown_norm (unknown)
        ], dtype=np.float32)

        action_int, _ = _calibrated_ppo.predict(obs_vec, deterministic=True)
        return _PPO_ACTION_MAP[int(action_int)]

    from serverless_simulator import (
        fixed_policy,
        rule_based_policy,
        persistence_policy,
    )

    policies = [
        (
            "Fixed-4",
            forecasts["persistence"],
            fixed_policy(4),
        ),
        (
            "Reactive Rule-based",
            forecasts["persistence"],
            reactive_rule_action,
        ),
        (
            "Persistence + Rule",
            forecasts["persistence"],
            persistence_policy(SLA_MS),
        ),
        (
            "Original TFT + Rule",
            forecasts["original_tft"],
            rule_based_policy(SLA_MS),
        ),
        (
            "Log TFT + Rule",
            forecasts["log_tft"],
            rule_based_policy(SLA_MS),
        ),
        (
            "Planner-assisted",
            forecasts["log_tft"],
            planner_action_from_obs,
        ),
    ]

    # Add calibrated PPO if model loaded successfully
    if _calibrated_ppo is not None:
        policies.append((
            "PPO (calibrated)",
            forecasts["persistence"],   # PPO uses persistence forecast as input
            ppo_calibrated_action,
        ))
        print(
            "\nPPO (calibrated): trained on Azure concurrency (training segment). "
            "Evaluated on the same test segment as all other policies.\n"
        )

    results = []
    policy_order = []

    print("\nStarting policy evaluations...\n")

    for policy_name, forecast, policy_fn in policies:

        policy_order.append(policy_name)

        for seed in range(N_SEEDS):

            row = run_policy(
                policy_name=policy_name,
                actual=actual,
                forecast=forecast,
                policy_fn=policy_fn,
                seed=seed,
            )

            results.append(row)

        print(
            f"Completed: {policy_name} "
            f"({N_SEEDS} seeds)"
        )

    results_df = pd.DataFrame(results)

    raw_path = OUT / "end_to_end_results.csv"
    results_df.to_csv(raw_path, index=False)

    print("\n" + "=" * 65)
    print("CALIBRATED EVALUATION SUMMARY")
    print("=" * 65)

    summary = print_summary(results_df)

    summary_path = OUT / "end_to_end_summary.csv"
    summary.to_csv(summary_path)

    plot_comparison(
        results_df,
        policy_order,
    )

    print("\nEvaluation completed.")
    print(f"Detailed results: {raw_path}")
    print(f"Summary results: {summary_path}")

    print(
        "\nInterpretation note: Azure invocation counts and "
        "execution durations are used to estimate average "
        "concurrency. This is not measured instantaneous "
        "concurrency. Latency, energy, cold starts and "
        "container metrics are simulator-generated proxies. ""cold_start_load_proxy is an overloaded-load estimate, not a count of real cold-start events. "
        "P95 latency is the 95th percentile across time-bucket "
        "mean-latency observations; mean_bucket_p95_latency_ms "
        "is the average of the simulator's per-bucket P95 estimates."
    )


if __name__ == "__main__":
    main()
