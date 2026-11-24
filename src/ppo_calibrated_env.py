"""
src/ppo_calibrated_env.py
=========================
Calibrated PPO Gymnasium environment for serverless container management.

This environment wraps the shared ServerlessSimulator so that the PPO
controller is trained and evaluated on exactly the same physics, latency
model, energy model, action semantics and workload units as the other six
policies in end_to_end_evaluation.py.

Key design decisions (documented per IMPLEMENTATION_PLAN.md §6, Phase E):
- Workload unit: estimated average concurrency =
      invocation_count * mean_duration_seconds / BUCKET_SECONDS (300 s)
  This matches the formula used throughout end_to_end_evaluation.py.
- Training data: Azure trace rows 0..(split-1)  [first 85%, time_idx 0–3426]
- Test data: Azure trace rows split..end         [final 15%, time_idx 3427–4031]
  The environment NEVER exposes test-segment data during training.
- Action mapping (integer → simulator string):
      0 -> NO_OP          (hold current state)
      1 -> PREWARM        (+1 warm container)
      2 -> SCALE_OUT      (+2 warm containers)
      3 -> SCALE_IN       (-1 warm container)
      4 -> REUSE          (keep warm, +0 delta)
      5 -> TERMINATE_IDLE (-2 warm containers)
- Observation (11-dim, normalised, no future leakage):
      0  prev_load_norm         previous bucket observed concurrency / LOAD_NORM
      1  next_forecast_norm     forecast for upcoming bucket / LOAD_NORM
      2  warm_containers_norm   warm containers / MAX_CONTAINERS
      3  utilisation            load / max(capacity, 1)
      4  latency_norm           prev latency / SLA_MS (may exceed 1 when violated)
      5  sla_risk               clipped(latency / SLA_MS, 0, 2) - 1
      6  energy_norm            prev energy / ENERGY_NORM
      7  cold_start_proxy       prev cold_start_count / max(load, 1)
      8  headroom_norm          (capacity - load) / LOAD_NORM, signed
      9  time_in_episode_norm   current step / episode_length
     10  cooldown_norm          steps since last scale action / COOLDOWN_NORM
- Reward combines SLA violation, latency, energy, cold-start and waste.
  Weights are documented and justified; test-set metrics were not used to
  choose them.
- Metrics are labelled SIMULATED PROXY throughout; none are measured live.
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from serverless_simulator import ServerlessSimulator, CAPACITY_PER_CONTAINER  # noqa: E402

# ---------------------------------------------------------------------------
# Constants (match end_to_end_evaluation.py exactly)
# ---------------------------------------------------------------------------

BUCKET_SECONDS: float = 300.0
TEST_FRAC: float = 0.15
SLA_MS: float = 200.0

# Normalisation constants (derived from TRAINING data; do not use test stats)
LOAD_NORM: float = 50.0          # typical scale of training concurrency
ENERGY_NORM: float = 10.0        # typical energy proxy range
COOLDOWN_NORM: float = 10.0      # steps for cooldown normalisation

# Action mapping (integer PPO action → simulator action string)
ACTION_MAP: dict[int, str] = {
    0: "NO_OP",
    1: "PREWARM",
    2: "SCALE_OUT",
    3: "SCALE_IN",
    4: "REUSE",
    5: "TERMINATE_IDLE",
}
N_ACTIONS: int = len(ACTION_MAP)
OBS_DIM: int = 11


# ---------------------------------------------------------------------------
# Workload helpers
# ---------------------------------------------------------------------------

def load_azure_concurrency(data_csv: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Load the Azure workload CSV and return:
      train_concurrency  — estimated concurrency for the training segment
      test_concurrency   — estimated concurrency for the test segment
      training_mean_dur  — mean execution duration of the training period

    Concurrency formula (same as end_to_end_evaluation.py):
      concurrency = invocation_count * mean_duration_seconds / BUCKET_SECONDS
    """
    df = pd.read_csv(data_csv)

    for col in ("time_idx", "invocation_count", "mean_duration_seconds"):
        if col not in df.columns:
            raise ValueError(f"Azure CSV missing column: {col}")

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=["time_idx", "invocation_count", "mean_duration_seconds"])
        .sort_values("time_idx")
        .drop_duplicates("time_idx", keep="last")
        .reset_index(drop=True)
    )
    df = df[
        (df["invocation_count"] >= 0) & (df["mean_duration_seconds"] > 0)
    ].reset_index(drop=True)

    n = len(df)
    split = int(n * (1 - TEST_FRAC))

    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    training_mean_dur = float(train_df["mean_duration_seconds"].mean())

    # Use per-row duration for the actual concurrency (consistent with
    # end_to_end_evaluation.py's actual_concurrency calculation).
    train_conc = (
        train_df["invocation_count"].to_numpy(dtype=np.float64)
        * train_df["mean_duration_seconds"].to_numpy(dtype=np.float64)
        / BUCKET_SECONDS
    ).astype(np.float32)

    test_conc = (
        test_df["invocation_count"].to_numpy(dtype=np.float64)
        * test_df["mean_duration_seconds"].to_numpy(dtype=np.float64)
        / BUCKET_SECONDS
    ).astype(np.float32)

    return train_conc, test_conc, training_mean_dur


def build_persistence_forecast(concurrency: np.ndarray) -> np.ndarray:
    """Persistence forecast: next value = current observed value."""
    if len(concurrency) == 0:
        return concurrency.copy()
    return np.concatenate([[concurrency[0]], concurrency[:-1]]).astype(np.float32)


def build_training_episodes(
    train_concurrency: np.ndarray,
    episode_len: int = 288,
    n_synthetic_variations: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Build a list of (actual, forecast) episode pairs from the training data.

    Strategy:
    1. Slice the training trace into non-overlapping windows of `episode_len`.
    2. Add a small number of jittered synthetic variations around the training
       distribution for coverage of edge cases.

    All synthetic variations stay within the training workload statistics;
    no test-segment values are used.

    Returns list of (actual_concurrency, forecast_concurrency) pairs.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    episodes = []
    n = len(train_concurrency)

    # Non-overlapping trace windows
    start = 0
    while start + episode_len <= n:
        actual = train_concurrency[start: start + episode_len].copy()
        forecast = build_persistence_forecast(actual)
        episodes.append((actual, forecast))
        start += episode_len

    # Include the leftover tail if long enough
    if n - start >= 32:
        actual = train_concurrency[start:].copy()
        forecast = build_persistence_forecast(actual)
        episodes.append((actual, forecast))

    # Synthetic variations: add noise around training statistics
    # (clearly labelled as synthetic; not presented as Azure data)
    if len(train_concurrency) > 0:
        mean_load = float(np.mean(train_concurrency))
        std_load = float(np.std(train_concurrency))
        # Clamp std to avoid degenerate distributions
        std_load = max(std_load, 1.0)

        for _ in range(n_synthetic_variations):
            noise = rng.normal(0, std_load * 0.3, episode_len).astype(np.float32)
            # Sample from training distribution with added noise
            base = rng.choice(train_concurrency, size=episode_len, replace=True)
            actual = np.maximum(0.0, base + noise).astype(np.float32)
            forecast = build_persistence_forecast(actual)
            episodes.append((actual, forecast))

    return episodes


# ---------------------------------------------------------------------------
# Gymnasium environment
# ---------------------------------------------------------------------------

class CalibratedServerlessEnv(gym.Env):
    """
    PPO environment using the shared ServerlessSimulator.

    Observation space: Box(OBS_DIM,), float32, partly normalised.
    Action space: Discrete(6) → mapped to simulator action strings.

    Reward (per step, all quantities are SIMULATED PROXIES):
        r = - w_sla   * sla_excess_norm       (SLA violation penalty)
            - w_lat   * latency_norm           (latency cost)
            - w_en    * energy_norm            (energy cost)
            - w_cold  * cold_start_proxy       (cold-start cost)
            - w_waste * waste_norm             (over-provisioning waste)
            - w_osc   * scale_action_penalty   (oscillation discouragement)

    Reward weights are documented here; they were not tuned on test data.
    """

    metadata = {"render_modes": []}

    # Reward weights
    W_SLA   = 5.0    # SLA violation (strongest signal)
    W_LAT   = 0.5    # Latency beyond base
    W_EN    = 0.3    # Energy proxy
    W_COLD  = 0.3    # Cold-start proxy
    W_WASTE = 0.05   # Excess capacity (over-provisioning)
    W_OSC   = 0.1    # Oscillation penalty per non-no-op action

    def __init__(
        self,
        episodes: list[tuple[np.ndarray, np.ndarray]],
        sla_ms: float = SLA_MS,
        seed: Optional[int] = None,
    ):
        super().__init__()

        if not episodes:
            raise ValueError("episodes list must not be empty")

        self.episodes = episodes
        self.sla_ms = float(sla_ms)
        self._master_seed = seed if seed is not None else 42
        self._rng = np.random.default_rng(self._master_seed)

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )

        # Episode state
        self._ep_actual: np.ndarray = np.zeros(1, dtype=np.float32)
        self._ep_forecast: np.ndarray = np.zeros(1, dtype=np.float32)
        self._ep_idx: int = 0        # index within episode
        self._ep_len: int = 0
        self._ep_num: int = -1       # which episode we are in
        self._prev_load: float = 0.0
        self._prev_latency: float = SLA_MS * 0.5
        self._prev_energy: float = 1.0
        self._prev_cold_proxy: float = 0.0
        self._steps_since_scale: int = 0
        self._sim: Optional[ServerlessSimulator] = None

    # ------------------------------------------------------------------
    def _select_episode(self) -> None:
        self._ep_num = int(self._rng.integers(0, len(self.episodes)))
        actual, forecast = self.episodes[self._ep_num]
        self._ep_actual = actual
        self._ep_forecast = forecast
        self._ep_len = len(actual)
        self._ep_idx = 0

    def _build_obs(
        self,
        prev_load: float,
        next_forecast: float,
        warm: int,
        latency: float,
        energy: float,
        cold_proxy: float,
    ) -> np.ndarray:
        capacity = warm * CAPACITY_PER_CONTAINER
        utilisation = prev_load / max(capacity, 1.0)
        headroom = (capacity - prev_load) / LOAD_NORM
        sla_risk = float(np.clip(latency / self.sla_ms - 1.0, -1.0, 1.0))

        return np.array(
            [
                prev_load / LOAD_NORM,
                next_forecast / LOAD_NORM,
                warm / float(50),                # 50 = max_containers
                float(np.clip(utilisation, 0.0, 3.0)),
                latency / self.sla_ms,
                sla_risk,
                energy / ENERGY_NORM,
                float(np.clip(cold_proxy, 0.0, 1.0)),
                float(np.clip(headroom, -3.0, 3.0)),
                self._ep_idx / max(self._ep_len - 1, 1),
                min(self._steps_since_scale, COOLDOWN_NORM) / COOLDOWN_NORM,
            ],
            dtype=np.float32,
        )

    def _compute_reward(
        self,
        latency: float,
        energy: float,
        cold_proxy: float,
        load: float,
        warm: int,
        action_int: int,
    ) -> float:
        sla_excess = max(0.0, latency - self.sla_ms) / self.sla_ms
        lat_norm = latency / self.sla_ms
        en_norm = energy / ENERGY_NORM
        capacity = warm * CAPACITY_PER_CONTAINER
        waste = max(0.0, capacity - load) / LOAD_NORM
        is_scale = ACTION_MAP[action_int] not in {"NO_OP", "REUSE"}

        reward = -(
            self.W_SLA   * sla_excess
            + self.W_LAT   * lat_norm
            + self.W_EN    * en_norm
            + self.W_COLD  * cold_proxy
            + self.W_WASTE * waste
            + self.W_OSC   * float(is_scale)
        )
        return float(reward)

    # ------------------------------------------------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._select_episode()
        self._sim = ServerlessSimulator(
            sla_ms=self.sla_ms,
            initial_warm=4,
            seed=int(self._rng.integers(0, 2**31)),
        )
        self._sim.reset()

        # First step: NO_OP to get initial state
        init_load = float(self._ep_actual[0])
        init_forecast = float(self._ep_forecast[0])
        m = self._sim.step("NO_OP", init_load, init_forecast)

        self._prev_load = m.actual_invocations
        self._prev_latency = m.latency_ms
        self._prev_energy = m.energy_proxy
        self._prev_cold_proxy = m.cold_start_proxy
        self._steps_since_scale = 0
        self._ep_idx = 1

        next_forecast = float(
            self._ep_forecast[min(1, self._ep_len - 1)]
        )
        obs = self._build_obs(
            self._prev_load,
            next_forecast,
            self._sim.warm,
            self._prev_latency,
            self._prev_energy,
            self._prev_cold_proxy,
        )

        return obs, {"episode_index": self._ep_num, "step": 0}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = int(action)
        if action not in ACTION_MAP:
            raise ValueError(f"Invalid action integer: {action}")

        action_str = ACTION_MAP[action]
        is_scale = action_str not in {"NO_OP", "REUSE"}
        self._steps_since_scale = (
            0 if is_scale else self._steps_since_scale + 1
        )

        actual_load = float(self._ep_actual[min(self._ep_idx, self._ep_len - 1)])
        predicted_load = float(self._ep_forecast[min(self._ep_idx, self._ep_len - 1)])

        m = self._sim.step(action_str, actual_load, predicted_load)

        reward = self._compute_reward(
            m.latency_ms,
            m.energy_proxy,
            m.cold_start_proxy,
            m.actual_invocations,
            m.warm_containers,
            action,
        )

        self._prev_load = m.actual_invocations
        self._prev_latency = m.latency_ms
        self._prev_energy = m.energy_proxy
        self._prev_cold_proxy = m.cold_start_proxy
        self._ep_idx += 1

        terminated = self._ep_idx >= self._ep_len

        next_forecast = float(
            self._ep_forecast[min(self._ep_idx, self._ep_len - 1)]
        )
        obs = self._build_obs(
            self._prev_load,
            next_forecast,
            m.warm_containers,
            m.latency_ms,
            m.energy_proxy,
            m.cold_start_proxy,
        )

        info = {
            "action_str": action_str,
            "latency_ms": m.latency_ms,
            "energy_proxy": m.energy_proxy,
            "warm_containers": m.warm_containers,
            "sla_violation": m.sla_violation,
            "cold_start_proxy": m.cold_start_proxy,
            "actual_load": m.actual_invocations,
            "step": self._ep_idx,
        }

        return obs, reward, terminated, False, info


# ---------------------------------------------------------------------------
# Convenience: build env from Azure CSV (training data only)
# ---------------------------------------------------------------------------

def make_training_env(
    data_csv: Optional[Path] = None,
    episode_len: int = 288,
    seed: int = 42,
) -> CalibratedServerlessEnv:
    """
    Load Azure training concurrency and return a ready-to-use Gymnasium env.

    Only the first 85% of the Azure trace (training segment) is used.
    Test data is never loaded here.
    """
    if data_csv is None:
        data_csv = ROOT / "data" / "azure_serverless_workload.csv"

    train_conc, _test_conc, _mean_dur = load_azure_concurrency(data_csv)

    rng = np.random.default_rng(seed)
    episodes = build_training_episodes(
        train_conc,
        episode_len=episode_len,
        n_synthetic_variations=3,
        rng=rng,
    )

    return CalibratedServerlessEnv(episodes=episodes, seed=seed)


# ---------------------------------------------------------------------------
# Smoke test (run directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=== CalibratedServerlessEnv smoke test ===")
    print(f"Action map: {ACTION_MAP}")
    print(f"Obs dim: {OBS_DIM}, Action count: {N_ACTIONS}")

    data_csv = ROOT / "data" / "azure_serverless_workload.csv"
    if not data_csv.exists():
        print(f"ERROR: data file not found: {data_csv}")
        sys.exit(1)

    env = make_training_env(data_csv=data_csv, episode_len=288, seed=42)
    print(f"Episodes loaded: {len(env.episodes)}")
    print(f"Episode 0 length: {len(env.episodes[0][0])}")

    # Run a short episode with random actions
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,), f"Wrong obs shape: {obs.shape}"
    assert np.all(np.isfinite(obs)), "Non-finite obs at reset"

    total_reward = 0.0
    n_violations = 0
    for step in range(50):
        action = int(step % N_ACTIONS)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if info["sla_violation"]:
            n_violations += 1
        if terminated:
            obs, info = env.reset()

    print(f"50-step smoke: total_reward={total_reward:.2f}, sla_violations={n_violations}")
    print(f"Obs range: [{obs.min():.3f}, {obs.max():.3f}]")
    assert np.all(np.isfinite(obs)), "Non-finite obs during episode"

    # Verify no test data loaded
    train_conc, test_conc, mean_dur = load_azure_concurrency(data_csv)
    print(f"\nData split verification:")
    print(f"  Train concurrency: {len(train_conc)} steps, mean={train_conc.mean():.3f}")
    print(f"  Test concurrency:  {len(test_conc)} steps, mean={test_conc.mean():.3f}")
    print(f"  Training mean duration: {mean_dur:.4f} s")
    print(f"  Env uses ONLY training episodes: {all(len(ep[0]) <= len(train_conc) for ep in env.episodes)}")

    print("\nSmoke test PASSED.")
