"""
Phase E: Representative serverless application functions.

Implements three example function types:
  1. user_info   — User-profile lookup (low latency, low CPU)
  2. search      — Full-text search (medium latency, medium CPU)
  3. recommend   — ML recommendation inference (high latency, high CPU)

A workload driver replays requests from the workload CSV, tags each
invocation with a function_id (round-robin or weighted), routes to
the appropriate function handler, and collects per-function metrics.

Run from D:\\major_1:
    python src\\function_services.py
"""

from __future__ import annotations
import json
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


# ── Function definitions ───────────────────────────────────────────────────────
@dataclass
class FunctionSpec:
    function_id: str
    description: str
    base_duration_ms: float     # mean execution time
    duration_std_ms: float      # std dev for variability
    cpu_intensity: float        # 0–1, fraction of a CPU core
    memory_mb: float            # RAM per invocation
    weight: float               # probability weight in workload mix


FUNCTION_REGISTRY: Dict[str, FunctionSpec] = {
    "user_info": FunctionSpec(
        function_id="user_info",
        description="User-profile lookup — low latency, low CPU",
        base_duration_ms=25.0,
        duration_std_ms=5.0,
        cpu_intensity=0.05,
        memory_mb=32.0,
        weight=0.50,
    ),
    "search": FunctionSpec(
        function_id="search",
        description="Full-text search — medium latency, medium CPU",
        base_duration_ms=120.0,
        duration_std_ms=30.0,
        cpu_intensity=0.25,
        memory_mb=128.0,
        weight=0.35,
    ),
    "recommend": FunctionSpec(
        function_id="recommend",
        description="ML recommendation inference — high latency, high CPU",
        base_duration_ms=350.0,
        duration_std_ms=80.0,
        cpu_intensity=0.70,
        memory_mb=512.0,
        weight=0.15,
    ),
}


# ── Invocation result ──────────────────────────────────────────────────────────
@dataclass
class InvocationResult:
    time_idx: int
    function_id: str
    cold_start: bool
    duration_ms: float
    total_latency_ms: float     # duration + cold-start overhead if any
    cpu_intensity: float
    memory_mb: float
    sla_ms: float
    sla_violated: bool
    success: bool
    error: Optional[str] = None


# ── Function handler ───────────────────────────────────────────────────────────
def invoke_function(
    spec: FunctionSpec,
    time_idx: int,
    cold_start: bool,
    sla_ms: float = 500.0,
    rng: Optional[np.random.Generator] = None,
) -> InvocationResult:
    """
    Simulate a single function invocation.
    Cold starts add 1500 ms overhead.
    """
    if rng is None:
        rng = np.random.default_rng()

    duration = max(
        1.0,
        rng.normal(spec.base_duration_ms, spec.duration_std_ms)
    )

    cold_overhead = 1500.0 if cold_start else 0.0
    total_latency = duration + cold_overhead

    return InvocationResult(
        time_idx=time_idx,
        function_id=spec.function_id,
        cold_start=cold_start,
        duration_ms=round(duration, 2),
        total_latency_ms=round(total_latency, 2),
        cpu_intensity=spec.cpu_intensity,
        memory_mb=spec.memory_mb,
        sla_ms=sla_ms,
        sla_violated=total_latency > sla_ms,
        success=True,
    )


# ── Workload driver ────────────────────────────────────────────────────────────
class WorkloadDriver:
    """
    Replays invocations from the workload CSV, assigns function IDs
    using the configured weight distribution, and collects per-function
    metrics.
    """

    def __init__(
        self,
        workload_csv: Path,
        function_registry: Dict[str, FunctionSpec] = None,
        sla_ms: float = 500.0,
        seed: int = 42,
        max_steps: Optional[int] = None,
    ):
        self.registry = function_registry or FUNCTION_REGISTRY
        self.sla_ms = sla_ms
        self.rng = np.random.default_rng(seed)
        self.max_steps = max_steps

        df = pd.read_csv(workload_csv)
        time_col = next(
            c for c in df.columns
            if c.lower() in ("time_idx", "time_index", "timestamp_idx")
        )
        load_col = next(
            c for c in df.columns
            if c.lower() in (
                "invocation_rate", "invocations", "invocation_count",
                "request_count", "load"
            )
        )
        df = df[[time_col, load_col]].copy()
        df.columns = ["time_idx", "invocations"]
        df["time_idx"] = pd.to_numeric(df["time_idx"], errors="coerce")
        df["invocations"] = pd.to_numeric(df["invocations"], errors="coerce")
        df = df.dropna().sort_values("time_idx").reset_index(drop=True)
        if max_steps:
            df = df.iloc[:max_steps]
        self.workload = df

        fn_ids = list(self.registry.keys())
        weights = np.array([self.registry[f].weight for f in fn_ids])
        weights /= weights.sum()
        self._fn_ids = fn_ids
        self._fn_weights = weights

    def _assign_function(self) -> str:
        return self.rng.choice(self._fn_ids, p=self._fn_weights)

    def run(self, cold_start_rate: float = 0.05) -> pd.DataFrame:
        """
        Replay the workload.  cold_start_rate is the per-invocation probability
        of a cold start (simplified; in a full simulation this comes from the
        container state).
        """
        results: List[InvocationResult] = []

        for _, row in self.workload.iterrows():
            time_idx = int(row["time_idx"])
            n_invocations = max(0, int(round(row["invocations"])))

            # Limit to a sample for performance (max 30 per bucket)
            sample = min(n_invocations, 30)

            for _ in range(sample):
                fn_id = self._assign_function()
                spec = self.registry[fn_id]
                cold = bool(self.rng.random() < cold_start_rate)
                result = invoke_function(
                    spec=spec,
                    time_idx=time_idx,
                    cold_start=cold,
                    sla_ms=self.sla_ms,
                    rng=self.rng,
                )
                results.append(result)

        return pd.DataFrame([asdict(r) for r in results])


# ── Per-function summary ────────────────────────────────────────────────────────
def summarize(df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for fn_id, grp in df.groupby("function_id"):
        summary_rows.append({
            "function_id": fn_id,
            "invocations": len(grp),
            "cold_starts": grp["cold_start"].sum(),
            "cold_start_rate": round(grp["cold_start"].mean(), 4),
            "mean_duration_ms": round(grp["duration_ms"].mean(), 2),
            "p95_duration_ms": round(grp["duration_ms"].quantile(0.95), 2),
            "mean_total_latency_ms": round(grp["total_latency_ms"].mean(), 2),
            "p95_total_latency_ms": round(grp["total_latency_ms"].quantile(0.95), 2),
            "sla_violations": grp["sla_violated"].sum(),
            "sla_violation_rate": round(grp["sla_violated"].mean(), 4),
        })
    return pd.DataFrame(summary_rows)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    workload_csv = ROOT / "data" / "azure_serverless_workload.csv"
    if not workload_csv.exists():
        # Fall back to synthetic data
        workload_csv = ROOT / "data" / "synthetic_serverless_workload.csv"

    print(f"Loading workload from: {workload_csv.name}")

    driver = WorkloadDriver(
        workload_csv=workload_csv,
        sla_ms=500.0,
        seed=42,
        max_steps=200,          # use first 200 buckets for demo
    )

    print("Replaying workload with cold_start_rate=0.05...")
    df = driver.run(cold_start_rate=0.05)

    raw_path = OUT / "function_invocations.csv"
    df.to_csv(raw_path, index=False)

    summary = summarize(df)
    summary_path = OUT / "function_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"\nTotal invocations simulated: {len(df)}")
    print("\n=== Per-function summary ===")
    print(summary.to_string(index=False))

    print(f"\nSaved: {raw_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
