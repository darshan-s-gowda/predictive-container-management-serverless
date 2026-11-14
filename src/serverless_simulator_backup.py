"""
Phase D: Python Simulator — Serverless Execution Layer.

Simulates warm/idle containers and lifecycle actions.
Simulates cold-start and service-time effects.
Enforces container bounds and applies allowed actions.
Collects latency, cold-start, capacity and energy-proxy metrics.

Run standalone:
    python src\\serverless_simulator.py
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import numpy as np


# ── Constants ──────────────────────────────────────────────────────────────────
CAPACITY_PER_CONTAINER = 10       # max invocations a single warm container handles
COLD_START_LATENCY_MS = 1500.0    # extra latency for a cold start
WARM_LATENCY_BASE_MS = 70.0       # base warm-container service time
LATENCY_SLOPE_MS = 9.0            # ms per unit of overload
ENERGY_PER_WARM = 0.8             # energy proxy per warm container per bucket
ENERGY_PER_CPU_PCT = 0.015        # energy proxy per CPU% per bucket
CPU_BASE = 15.0                   # baseline CPU% at zero load
CPU_SLOPE = 1.5                   # CPU% per unit of load
MEMORY_BASE = 20.0                # baseline memory%
MEMORY_PER_WARM = 7.0
MEMORY_PER_LOAD = 0.8


@dataclass
class ContainerState:
    """Snapshot of the container pool at one timestep."""
    time_idx: int
    warm_containers: int
    idle_containers: int
    total_capacity: float
    headroom: float


@dataclass
class StepMetrics:
    """Metrics collected for one simulation step."""
    time_idx: int
    actual_invocations: float
    predicted_invocations: float
    warm_containers: int
    action: str
    container_delta: int
    cold_start_count: float
    cold_start_proxy: float
    latency_ms: float
    p95_latency_ms: float
    sla_violation: bool
    sla_excess_ms: float
    cpu_pct: float
    memory_pct: float
    energy_proxy: float
    throughput: float               # invocations served per bucket


class ServerlessSimulator:
    """
    Python simulator for the serverless execution layer.

    Applies container-management actions (NO_OP, PREWARM, SCALE_OUT, SCALE_IN,
    REUSE, TERMINATE_IDLE) and returns per-step metrics.
    """

    ACTION_DELTAS = {
        "NO_OP": 0,
        "REUSE": 0,
        "PREWARM": 1,
        "SCALE_OUT": 2,
        "SCALE_IN": -1,
        "TERMINATE_IDLE": -2,
    }

    def __init__(
        self,
        sla_ms: float = 200.0,
        initial_warm: int = 4,
        min_containers: int = 1,
        max_containers: int = 50,
        noise_std: float = 1.5,
        seed: int = 42,
    ):
        self.sla_ms = float(sla_ms)
        self.initial_warm = int(initial_warm)
        self.min_containers = int(min_containers)
        self.max_containers = int(max_containers)
        self.noise_std = float(noise_std)
        self._rng = np.random.default_rng(seed)

        self.warm: int = self.initial_warm
        self.t: int = 0
        self.history: List[StepMetrics] = []

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.warm = self.initial_warm
        self.t = 0
        self.history = []

    def _noisy_load(self, base_load: float) -> float:
        noise = self._rng.normal(0.0, self.noise_std)
        return max(0.0, base_load + noise)

    def _compute_latency(self, load: float, warm: int) -> tuple[float, float]:
        """Return (mean_latency_ms, p95_latency_ms)."""
        capacity = warm * CAPACITY_PER_CONTAINER
        overload = max(0.0, load - capacity)

        # Cold-start fraction — linear with overload relative to capacity
        cold_fraction = min(1.0, overload / max(load, 1.0))
        cold_latency_contribution = cold_fraction * COLD_START_LATENCY_MS

        warm_latency = WARM_LATENCY_BASE_MS + overload * LATENCY_SLOPE_MS
        mean_latency = warm_latency + cold_latency_contribution

        # P95 adds noise tail (±25%)
        p95_latency = mean_latency * 1.25

        return mean_latency, p95_latency

    def _compute_energy(self, load: float, warm: int) -> tuple[float, float, float]:
        """Return (cpu_pct, memory_pct, energy_proxy)."""
        cpu = min(100.0, CPU_BASE + load * CPU_SLOPE)
        memory = min(100.0, MEMORY_BASE + warm * MEMORY_PER_WARM + load * MEMORY_PER_LOAD)
        energy = warm * ENERGY_PER_WARM + cpu * ENERGY_PER_CPU_PCT
        return cpu, memory, energy

    def step(
        self,
        action: str,
        actual_load: float,
        predicted_load: float,
    ) -> StepMetrics:
        """Apply action, advance timestep, return metrics."""
        if action not in self.ACTION_DELTAS:
            raise ValueError(f"Unknown action: {action!r}")

        delta = self.ACTION_DELTAS[action]
        prev_warm = self.warm
        self.warm = int(
            max(self.min_containers,
                min(self.max_containers, self.warm + delta))
        )
        actual_delta = self.warm - prev_warm

        # Noisy actual load
        load = self._noisy_load(actual_load)

        capacity = self.warm * CAPACITY_PER_CONTAINER
        overload = max(0.0, load - capacity)

        # Cold-start metrics
        cold_fraction = min(1.0, overload / max(load, 1.0))
        cold_count = cold_fraction * load   # estimated cold-start invocations

        latency, p95 = self._compute_latency(load, self.warm)
        cpu, memory, energy = self._compute_energy(load, self.warm)

        sla_violation = latency > self.sla_ms
        sla_excess = max(0.0, latency - self.sla_ms)

        throughput = min(load, capacity)

        m = StepMetrics(
            time_idx=self.t,
            actual_invocations=load,
            predicted_invocations=predicted_load,
            warm_containers=self.warm,
            action=action,
            container_delta=actual_delta,
            cold_start_count=round(cold_count, 2),
            cold_start_proxy=round(cold_fraction, 4),
            latency_ms=round(latency, 2),
            p95_latency_ms=round(p95, 2),
            sla_violation=sla_violation,
            sla_excess_ms=round(sla_excess, 2),
            cpu_pct=round(cpu, 2),
            memory_pct=round(memory, 2),
            energy_proxy=round(energy, 4),
            throughput=round(throughput, 2),
        )

        self.history.append(m)
        self.t += 1
        return m

    def run(
        self,
        actual_workload: np.ndarray,
        predicted_workload: np.ndarray,
        policy_fn,          # callable(obs_dict) -> str action
    ) -> List[StepMetrics]:
        """
        Run the simulator over a full workload trace using a policy function.
        policy_fn receives a dict observation and must return an action string.
        """
        self.reset()
        n = min(len(actual_workload), len(predicted_workload))

        for i in range(n):
            obs = {
                "time_idx": self.t,
                "actual_invocations": float(actual_workload[i]),
                "predicted_invocations": float(predicted_workload[i]),
                "warm_containers": self.warm,
                "sla_ms": self.sla_ms,
                "latency_ms": self._compute_latency(
                    float(actual_workload[i]), self.warm
                )[0],
                "energy_proxy": self._compute_energy(
                    float(actual_workload[i]), self.warm
                )[2],
            }
            action = policy_fn(obs)
            self.step(
                action=action,
                actual_load=float(actual_workload[i]),
                predicted_load=float(predicted_workload[i]),
            )

        return self.history


# ── Built-in policies ──────────────────────────────────────────────────────────
def fixed_policy(n_containers: int = 4):
    """Always keeps a fixed number of containers."""
    def _policy(obs: dict) -> str:
        current = obs["warm_containers"]
        if current < n_containers:
            return "SCALE_OUT"
        elif current > n_containers:
            return "SCALE_IN"
        return "NO_OP"
    return _policy


def rule_based_policy(sla_ms: float = 200.0):
    """Threshold-based rule policy using the predicted workload."""
    def _policy(obs: dict) -> str:
        predicted = obs["predicted_invocations"]
        warm = obs["warm_containers"]
        capacity = warm * CAPACITY_PER_CONTAINER
        latency = obs["latency_ms"]

        if latency > sla_ms * 0.9 or predicted > capacity + 15:
            return "SCALE_OUT"
        elif predicted > capacity + 5:
            return "PREWARM"
        elif predicted < capacity - 20 and warm > 1:
            return "TERMINATE_IDLE"
        elif predicted < capacity - 10 and warm > 1:
            return "SCALE_IN"
        elif abs(predicted - capacity) < 5:
            return "REUSE"
        return "NO_OP"
    return _policy


def persistence_policy(sla_ms: float = 200.0):
    """Uses the current actual load as the prediction (persistence)."""
    def _policy(obs: dict) -> str:
        actual = obs["actual_invocations"]
        warm = obs["warm_containers"]
        capacity = warm * CAPACITY_PER_CONTAINER
        latency = obs["latency_ms"]

        if latency > sla_ms * 0.9 or actual > capacity + 15:
            return "SCALE_OUT"
        elif actual > capacity + 5:
            return "PREWARM"
        elif actual < capacity - 20 and warm > 1:
            return "TERMINATE_IDLE"
        elif actual < capacity - 10 and warm > 1:
            return "SCALE_IN"
        return "NO_OP"
    return _policy


# ── Smoke test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 100
    t = np.arange(n)
    workload = 30 + 20 * np.sin(t * np.pi / 24) + rng.normal(0, 3, n)
    workload = np.maximum(0, workload).astype(np.float32)
    persistence = np.concatenate([[workload[0]], workload[:-1]])

    policies = {
        "Fixed-4": fixed_policy(4),
        "Rule-based": rule_based_policy(),
        "Persistence-driven": persistence_policy(),
    }

    for name, pol in policies.items():
        sim = ServerlessSimulator(seed=42)
        hist = sim.run(workload, persistence, pol)
        df = pd.DataFrame([asdict(m) for m in hist])
        print(f"\n{name}:")
        print(f"  avg latency:    {df['latency_ms'].mean():.1f} ms")
        print(f"  SLA violations: {df['sla_violation'].sum()}")
        print(f"  avg energy:     {df['energy_proxy'].mean():.3f}")
        print(f"  cold starts:    {df['cold_start_count'].sum():.1f}")
        print(f"  avg warm conts: {df['warm_containers'].mean():.2f}")

    print("\nSimulator smoke test complete.")
