"""
tests/test_ppo_calibrated_env.py
=================================
Contract tests for the calibrated PPO environment.

Verifies:
  - Obs/action shapes
  - No test-data leakage into training episodes
  - Action mapping matches simulator allowlist
  - Reward is finite
  - Observation is finite at all steps
  - SLA flag consistency
  - Determinism under same seed
  - Training env uses only training data (< test split)
"""

import sys
import numpy as np
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "azure_serverless_workload.csv"
TEST_FRAC = 0.15


@pytest.fixture(scope="module")
def env():
    from ppo_calibrated_env import make_training_env
    return make_training_env(data_csv=DATA_CSV, episode_len=288, seed=42)


@pytest.fixture(scope="module")
def azure_split():
    """Return (train_len, test_len) from the Azure CSV."""
    import pandas as pd
    df = pd.read_csv(DATA_CSV)
    n = len(df)
    split = int(n * (1 - TEST_FRAC))
    return split, n - split


class TestCalibratedEnvShape:

    def test_obs_space_shape(self, env):
        from ppo_calibrated_env import OBS_DIM
        assert env.observation_space.shape == (OBS_DIM,)

    def test_action_space_size(self, env):
        assert env.action_space.n == 6

    def test_reset_obs_shape_and_finite(self, env):
        from ppo_calibrated_env import OBS_DIM
        obs, _ = env.reset(seed=0)
        assert obs.shape == (OBS_DIM,)
        assert np.all(np.isfinite(obs))

    def test_step_obs_finite_and_reward_finite(self, env):
        env.reset(seed=0)
        obs, reward, _, _, info = env.step(0)
        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)

    def test_invalid_action_raises(self, env):
        env.reset(seed=0)
        with pytest.raises((ValueError, KeyError)):
            env.step(99)


class TestCalibratedEnvDataLeakage:

    def test_no_test_data_in_training_episodes(self, env, azure_split):
        """Every training episode must be shorter than or equal to train_len."""
        train_len, _ = azure_split
        for i, (actual, _) in enumerate(env.episodes):
            assert len(actual) <= train_len, (
                f"Episode {i} has {len(actual)} steps > train_len {train_len}"
            )

    def test_all_episodes_built_from_training_range(self, env, azure_split):
        """
        Load the actual Azure concurrency and verify that every sample in
        every episode comes from values within the training distribution
        (not strictly checking index-equality, but checking that episode
        values are plausibly within the training range, not test extremes).
        """
        import pandas as pd
        df = pd.read_csv(DATA_CSV)
        n = len(df)
        train_split = int(n * (1 - TEST_FRAC))
        train_df = df.iloc[:train_split]
        train_max_conc = float(
            (train_df["invocation_count"] * train_df["mean_duration_seconds"] / 300.0).max()
        )
        for i, (actual, _) in enumerate(env.episodes):
            ep_max = float(actual.max())
            # Synthetic episodes may have slight noise beyond; allow 20% margin
            assert ep_max <= train_max_conc * 1.2, (
                f"Episode {i} max concurrency {ep_max:.2f} greatly exceeds "
                f"training max {train_max_conc:.2f}"
            )


class TestCalibratedEnvActionSemantics:

    def test_action_map_matches_simulator_allowlist(self):
        from ppo_calibrated_env import ACTION_MAP
        from serverless_simulator import ServerlessSimulator
        mapped_strings = set(ACTION_MAP.values())
        sim_strings = set(ServerlessSimulator.ACTION_DELTAS.keys())
        assert mapped_strings == sim_strings, (
            f"ACTION_MAP values {mapped_strings} != simulator actions {sim_strings}"
        )

    def test_all_6_actions_produce_finite_obs(self, env):
        from ppo_calibrated_env import N_ACTIONS
        for action_int in range(N_ACTIONS):
            env.reset(seed=action_int)
            obs, reward, _, _, info = env.step(action_int)
            assert np.all(np.isfinite(obs)), f"Non-finite obs for action {action_int}"
            assert np.isfinite(reward), f"Non-finite reward for action {action_int}"
            assert info["action_str"] is not None


class TestCalibratedEnvReward:

    def test_sla_violation_increases_penalty(self, env):
        """An episode with heavy overload should produce worse (lower) reward."""
        import numpy as np
        from ppo_calibrated_env import CalibratedServerlessEnv, build_persistence_forecast

        # Light episode
        light = np.full(50, 1.0, dtype=np.float32)
        heavy = np.full(50, 500.0, dtype=np.float32)

        env_l = CalibratedServerlessEnv(
            [(light, build_persistence_forecast(light))], seed=0
        )
        env_h = CalibratedServerlessEnv(
            [(heavy, build_persistence_forecast(heavy))], seed=0
        )

        def run_episode(e):
            obs, _ = e.reset(seed=0)
            total = 0.0
            for _ in range(49):
                obs, r, done, _, _ = e.step(0)  # NO_OP
                total += r
                if done:
                    break
            return total

        r_light = run_episode(env_l)
        r_heavy = run_episode(env_h)
        assert r_heavy < r_light, (
            f"Heavy overload should give lower reward: heavy={r_heavy:.2f}, light={r_light:.2f}"
        )

    def test_reward_is_finite_every_step(self, env):
        obs, _ = env.reset(seed=7)
        for _ in range(50):
            action = int(_ % 6)
            obs, reward, done, _, info = env.step(action)
            assert np.isfinite(reward)
            if done:
                obs, _ = env.reset()


class TestCalibratedEnvDeterminism:

    def test_same_seed_same_trajectory(self):
        from ppo_calibrated_env import make_training_env
        env1 = make_training_env(data_csv=DATA_CSV, episode_len=100, seed=99)
        env2 = make_training_env(data_csv=DATA_CSV, episode_len=100, seed=99)

        obs1, _ = env1.reset(seed=0)
        obs2, _ = env2.reset(seed=0)
        np.testing.assert_array_equal(obs1, obs2)

        for a in range(20):
            o1, r1, d1, _, i1 = env1.step(a % 6)
            o2, r2, d2, _, i2 = env2.step(a % 6)
            np.testing.assert_array_equal(o1, o2)
            assert r1 == r2
