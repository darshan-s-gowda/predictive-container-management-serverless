
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


# Simulator configuration
CAPACITY_PER_CONTAINER = 10
COLD_START_LATENCY_MS = 1500.0
WARM_LATENCY_BASE_MS = 70.0
LATENCY_SLOPE_MS = 9.0
ENERGY_PER_WARM = 0.8
ENERGY_PER_CPU_PCT = 0.015
CPU_BASE = 15.0
CPU_SLOPE = 1.5
MEMORY_BASE = 20.0
MEMORY_PER_WARM = 7.0
MEMORY_PER_LOAD = 0.8


@dataclass
class ContainerState:
    time_idx: int
    warm_containers: int
    idle_containers: int
    total_capacity: float
    headroom: float


@dataclass
class StepMetrics:
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
    throughput: float


class ServerlessSimulator:
    """Serverless container simulator with next-bucket decisions."""

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

        self.warm = self.initial_warm
        self.t = 0
        self.history: List[StepMetrics] = []

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.warm = self.initial_warm
        self.t = 0
        self.history = []

    def _noisy_load(self, base_load: float) -> float:
        return max(
            0.0,
            float(base_load) + self._rng.normal(0.0, self.noise_std)
        )

    def _compute_latency(self, load: float, warm: int):
        capacity = warm * CAPACITY_PER_CONTAINER
        overload = max(0.0, load - capacity)

        cold_fraction = min(1.0, overload / max(load, 1.0))
        cold_contribution = cold_fraction * COLD_START_LATENCY_MS

        mean_latency = (
            WARM_LATENCY_BASE_MS
            + overload * LATENCY_SLOPE_MS
            + cold_contribution
        )
        p95_latency = mean_latency * 1.25

        return mean_latency, p95_latency

    def _compute_energy(self, load: float, warm: int):
        cpu = min(100.0, CPU_BASE + load * CPU_SLOPE)
        memory = min(
            100.0,
            MEMORY_BASE + warm * MEMORY_PER_WARM
            + load * MEMORY_PER_LOAD,
        )
        energy = warm * ENERGY_PER_WARM + cpu * ENERGY_PER_CPU_PCT

        return cpu, memory, energy

    def step(
        self,
        action: str,
        actual_load: float,
        predicted_load: float,
    ) -> StepMetrics:
        """Apply the selected action and simulate one workload bucket."""

        if action not in self.ACTION_DELTAS:
            raise ValueError(f"Unknown action: {action!r}")

        previous_warm = self.warm
        requested_delta = self.ACTION_DELTAS[action]

        self.warm = max(
            self.min_containers,
            min(self.max_containers, self.warm + requested_delta),
        )
        actual_delta = self.warm - previous_warm

        load = self._noisy_load(actual_load)

        capacity = self.warm * CAPACITY_PER_CONTAINER
        overload = max(0.0, load - capacity)

        cold_fraction = min(1.0, overload / max(load, 1.0))
        cold_count = cold_fraction * load

        latency, p95 = self._compute_latency(load, self.warm)
        cpu, memory, energy = self._compute_energy(load, self.warm)

        violation = latency > self.sla_ms
        sla_excess = max(0.0, latency - self.sla_ms)
        throughput = min(load, capacity)

        metrics = StepMetrics(
            time_idx=self.t,
            actual_invocations=round(float(load), 2),
            predicted_invocations=float(predicted_load),
            warm_containers=self.warm,
            action=action,
            container_delta=actual_delta,
            cold_start_count=round(float(cold_count), 2),
            cold_start_proxy=round(float(cold_fraction), 4),
            latency_ms=round(float(latency), 2),
            p95_latency_ms=round(float(p95), 2),
            sla_violation=bool(violation),
            sla_excess_ms=round(float(sla_excess), 2),
            cpu_pct=round(float(cpu), 2),
            memory_pct=round(float(memory), 2),
            energy_proxy=round(float(energy), 4),
            throughput=round(float(throughput), 2),
        )

        self.history.append(metrics)
        self.t += 1

        return metrics

    def run(
        self,
        actual_workload: np.ndarray,
        predicted_workload: np.ndarray,
        policy_fn,
    ) -> List[StepMetrics]:
        """
        Evaluate a policy without exposing the next bucket's actual load.

        The first bucket uses NO_OP to establish the initial observation.
        For each later bucket, the policy receives the previous observed
        load and the forecast for the upcoming bucket.
        """

        self.reset()

        actual = np.asarray(actual_workload, dtype=np.float32)
        predicted = np.asarray(predicted_workload, dtype=np.float32)

        n = min(len(actual), len(predicted))

        if n == 0:
            return self.history

        # First bucket: initial pool, no advance knowledge.
        first = self.step(
            action="NO_OP",
            actual_load=float(actual[0]),
            predicted_load=float(predicted[0]),
        )

        previous_observed_load = first.actual_invocations

        # Subsequent buckets: observe previous state, forecast next load,
        # choose action, then apply it to the upcoming bucket.
        for i in range(1, n):
            warm = self.warm

            previous_latency = self._compute_latency(
                previous_observed_load, warm
            )[0]

            previous_energy = self._compute_energy(
                previous_observed_load, warm
            )[2]

            obs = {
                "time_idx": i - 1,
                "actual_invocations": previous_observed_load,
                "predicted_invocations": float(predicted[i]),
                "warm_containers": warm,
                "sla_ms": self.sla_ms,
                "latency_ms": previous_latency,
                "energy_proxy": previous_energy,
            }

            action = policy_fn(obs)

            metrics = self.step(
                action=action,
                actual_load=float(actual[i]),
                predicted_load=float(predicted[i]),
            )

            previous_observed_load = metrics.actual_invocations

        return self.history


def fixed_policy(n_containers: int = 4):
    """Maintain a fixed target container count."""

    def _policy(obs: dict) -> str:
        current = obs["warm_containers"]

        if current < n_containers:
            return "SCALE_OUT"
        if current > n_containers:
            return "SCALE_IN"

        return "NO_OP"

    return _policy


def rule_based_policy(sla_ms: float = 200.0):
    """Threshold-based policy using the upcoming workload forecast."""

    def _policy(obs: dict) -> str:
        predicted = obs["predicted_invocations"]
        warm = obs["warm_containers"]
        capacity = warm * CAPACITY_PER_CONTAINER
        latency = obs["latency_ms"]

        if latency > sla_ms * 0.9 or predicted > capacity + 15:
            return "SCALE_OUT"
        if predicted > capacity + 5:
            return "PREWARM"
        if predicted < capacity - 20 and warm > 1:
            return "TERMINATE_IDLE"
        if predicted < capacity - 10 and warm > 1:
            return "SCALE_IN"
        if abs(predicted - capacity) < 5:
            return "REUSE"

        return "NO_OP"

    return _policy


def persistence_policy(sla_ms: float = 200.0):
    """Use the previous observed bucket as the next-bucket forecast."""

    def _policy(obs: dict) -> str:
        predicted = obs["actual_invocations"]
        warm = obs["warm_containers"]
        capacity = warm * CAPACITY_PER_CONTAINER
        latency = obs["latency_ms"]

        if latency > sla_ms * 0.9 or predicted > capacity + 15:
            return "SCALE_OUT"
        if predicted > capacity + 5:
            return "PREWARM"
        if predicted < capacity - 20 and warm > 1:
            return "TERMINATE_IDLE"
        if predicted < capacity - 10 and warm > 1:
            return "SCALE_IN"

        return "NO_OP"

    return _policy


if __name__ == "__main__":
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 100
    t = np.arange(n)

    workload = 30 + 20 * np.sin(t * np.pi / 24)
    workload += rng.normal(0, 3, n)
    workload = np.maximum(0, workload).astype(np.float32)

    persistence = np.concatenate(
        [[workload[0]], workload[:-1]]
    ).astype(np.float32)

    policies = {
        "Fixed-4": fixed_policy(4),
        "Rule-based": rule_based_policy(),
        "Persistence-driven": persistence_policy(),
    }

    for name, policy in policies.items():
        sim = ServerlessSimulator(seed=42)
        history = sim.run(workload, persistence, policy)
        frame = pd.DataFrame([m.__dict__ for m in history])

        print(f"\n{name}:")
        print(f"  avg latency: {frame['latency_ms'].mean():.1f} ms")
        print(f"  SLA violations: {frame['sla_violation'].sum()}")
        print(f"  avg energy: {frame['energy_proxy'].mean():.3f}")
        print(f"  cold starts: {frame['cold_start_count'].sum():.1f}")
        print(f"  avg warm containers: {frame['warm_containers'].mean():.2f}")

    print("\nSimulator smoke test complete.")
