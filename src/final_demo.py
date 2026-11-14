"""
Phase H: Final integrated demonstration.

Pipeline:
  Azure workload -> persistence forecast -> planner decision
  -> serverless simulator -> monitoring -> results and plots.

Run from the project root:
    python src\\final_demo.py

All latency, SLA, energy, and cold-start-related values are simulator
estimates/proxies, not measurements from a deployed cloud platform.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
OUT = ROOT / "outputs"
DATA = ROOT / "data" / "azure_serverless_workload.csv"

sys.path.insert(0, str(SRC))
OUT.mkdir(parents=True, exist_ok=True)

SLA_MS = 200.0
DEMO_STEPS = 288
SEED = 42
INVOC_SCALE = 12.0


def load_workload(n: int):
    """
    Load the first n Azure workload buckets and convert invocation counts
    to estimated average concurrency.

    Returns:
        time_idx, actual_raw, actual_scaled, persistence_forecast
    """
    if not DATA.exists():
        raise FileNotFoundError(f"Azure workload file not found: {DATA}")

    df = pd.read_csv(DATA)

    required = {"time_idx", "invocation_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Workload CSV missing columns: {sorted(missing)}")

    df = df[["time_idx", "invocation_count"]].copy()
    df["time_idx"] = pd.to_numeric(df["time_idx"], errors="coerce")
    df["invocation_count"] = pd.to_numeric(
        df["invocation_count"], errors="coerce"
    )
    df = (
        df.dropna()
        .sort_values("time_idx")
        .drop_duplicates("time_idx", keep="last")
        .reset_index(drop=True)
    )

    if len(df) < 2:
        raise ValueError("At least two valid workload buckets are required.")

    df = df.iloc[:n].copy()
    if df.empty:
        raise ValueError("No workload rows available for the demo.")

    time_idx = df["time_idx"].to_numpy()
    actual_raw = df["invocation_count"].to_numpy(dtype=np.float32)

    # One simulator load unit represents INVOC_SCALE trace invocations.
    actual_scaled = (actual_raw / INVOC_SCALE).astype(np.float32)

    # Persistence forecast for bucket i uses the actual load from bucket i-1.
    # The first value is only a placeholder: the first bucket uses NO_OP and
    # does not make a forecast-driven decision.
    persistence = np.concatenate(
        ([actual_scaled[0]], actual_scaled[:-1])
    ).astype(np.float32)

    return time_idx, actual_raw, actual_scaled, persistence


def build_planner_action(previous_metrics, next_forecast: float, step_index: int):
    """Make a decision using only the previous bucket's observation."""
    from agentic_planner import PlannerInput, plan

    previous_load = float(previous_metrics.actual_invocations)
    warm = int(previous_metrics.warm_containers)
    capacity = warm * 10.0
    overload = max(0.0, previous_load - capacity)
    cold_rate = min(1.0, overload / max(previous_load, 1.0))

    planner_input = PlannerInput(
        time_idx=step_index - 1,
        forecast_invocations=float(next_forecast),
        actual_invocations=previous_load,
        warm_containers=warm,
        sla_ms=SLA_MS,
        current_latency_ms=float(previous_metrics.latency_ms),
        energy_proxy=float(previous_metrics.energy_proxy),
        capacity_limit=50,
        cold_start_rate=cold_rate,
    )

    result = plan(planner_input)
    return result.recommended_action


def main():
    print("=" * 68)
    print("PREDICTIVE CONTAINER MANAGEMENT - FINAL INTEGRATED DEMO")
    print("Azure workload -> forecast -> planner -> simulator -> monitoring")
    print("All operational metrics are simulated estimates/proxies.")
    print("=" * 68)

    print(f"\n[1/5] Loading up to {DEMO_STEPS} Azure workload buckets...")
    time_idx, actual_raw, actual_scaled, forecast = load_workload(DEMO_STEPS)

    print(f"      Buckets loaded       : {len(actual_scaled)}")
    print(
        f"      Raw invocations     : mean={actual_raw.mean():.2f}, "
        f"max={actual_raw.max():.2f} per bucket"
    )
    print(
        f"      Estimated concurrency: mean={actual_scaled.mean():.2f}, "
        f"max={actual_scaled.max():.2f} simulator units"
    )
    print(f"      Scale factor        : {INVOC_SCALE:.1f} invocations/unit")

    from serverless_simulator import ServerlessSimulator
    from monitoring import MonitoringLogger, plot_monitoring_dashboard

    sim = ServerlessSimulator(
        sla_ms=SLA_MS,
        initial_warm=4,
        seed=SEED,
    )
    sim.reset()

    logger = MonitoringLogger(
        policy_name="Planner-assisted",
        policy_version="2.0",
        seed=SEED,
        forecast_model="persistence",
        sla_ms=SLA_MS,
    )

    print("\n[2/5] Running leak-free planner-assisted simulation...")
    records = []
    previous_metrics = None

    for i, load in enumerate(actual_scaled):
        next_forecast = float(forecast[i])

        # The first bucket initializes the simulator. No same-bucket actual
        # workload is passed to the planner before the action is selected.
        if i == 0:
            action = "NO_OP"
        else:
            action = build_planner_action(
                previous_metrics=previous_metrics,
                next_forecast=next_forecast,
                step_index=i,
            )

        metrics = sim.step(
            action=action,
            actual_load=float(load),
            predicted_load=next_forecast,
        )
        records.append(metrics)
        previous_metrics = metrics

        logger.log(
            time_idx=i,
            actual_invocations=metrics.actual_invocations,
            predicted_invocations=metrics.predicted_invocations,
            warm_containers=metrics.warm_containers,
            action=metrics.action,
            container_delta=metrics.container_delta,
            latency_ms=metrics.latency_ms,
            p95_latency_ms=metrics.p95_latency_ms,
            cold_start_count=metrics.cold_start_count,
            cold_start_proxy=metrics.cold_start_proxy,
            cpu_pct=metrics.cpu_pct,
            memory_pct=metrics.memory_pct,
            energy_proxy=metrics.energy_proxy,
            sla_violated=metrics.sla_violation,
            sla_excess_ms=metrics.sla_excess_ms,
        )

        if (i + 1) % 72 == 0:
            print(f"      Processed {i + 1}/{len(actual_scaled)} buckets")

    monitored = logger.to_dataframe()
    logger.close()

    if monitored.empty:
        raise RuntimeError("Demo completed without producing monitoring rows.")

    # Preserve time and raw workload alongside the simulator/monitoring data.
    result_df = monitored.copy()
    result_df["time_idx"] = time_idx[:len(result_df)]
    result_df["actual_raw_invocations"] = actual_raw[:len(result_df)]
    result_df["forecast_raw_invocations"] = (
        forecast[:len(result_df)] * INVOC_SCALE
    )

    print("\n[3/5] Simulation summary")
    print(f"      Steps                 : {len(result_df)}")
    print(
        f"      Mean latency         : "
        f"{result_df['latency_ms'].mean():.2f} ms"
    )
    print(
        f"      P95 over bucket means: "
        f"{result_df['latency_ms'].quantile(0.95):.2f} ms"
    )
    print(
        f"      Mean bucket P95 est. : "
        f"{result_df['p95_latency_ms'].mean():.2f} ms"
    )
    print(
        f"      SLA violation rate   : "
        f"{result_df['sla_violated'].mean():.2%}"
    )
    print(
        f"      Cold-start load proxy: "
        f"{result_df['cold_start_count'].sum():.2f} "
        "(not actual event count)"
    )
    print(
        f"      Avg warm containers  : "
        f"{result_df['warm_containers'].mean():.2f}"
    )
    print(
        f"      Avg energy proxy     : "
        f"{result_df['energy_proxy'].mean():.4f}"
    )
    print("      Actions:")
    for action, count in result_df["action"].value_counts().items():
        print(f"        {action:<20} {count}")

    print("\n[4/5] Generating monitoring and demo plots...")
    plot_monitoring_dashboard(
        monitored,
        policy_name="Planner-assisted",
        sla_ms=SLA_MS,
    )

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 8), sharex=True
    )

    ax1.plot(
        time_idx[:len(result_df)],
        actual_raw[:len(result_df)],
        label="Actual Azure invocations",
        linewidth=1.1,
    )
    ax1.plot(
        time_idx[:len(result_df)],
        forecast[:len(result_df)] * INVOC_SCALE,
        label="Persistence forecast (previous bucket)",
        linewidth=1.0,
        linestyle="--",
        alpha=0.8,
    )
    ax1.set_ylabel("Invocations / 5-minute bucket")
    ax1.set_title("Azure workload: actual vs persistence forecast")
    ax1.legend()
    ax1.grid(alpha=0.2)

    x = time_idx[:len(result_df)]
    ax2.step(
        x,
        result_df["warm_containers"],
        where="mid",
        label="Warm containers",
        linewidth=1.2,
    )
    ax2b = ax2.twinx()
    ax2b.plot(
        x,
        result_df["latency_ms"],
        label="Simulated mean latency",
        linewidth=0.8,
        alpha=0.7,
    )
    ax2b.axhline(
        SLA_MS,
        linestyle=":",
        linewidth=1.0,
        label=f"SLA threshold ({SLA_MS:.0f} ms)",
    )
    ax2.set_xlabel("Azure workload time index (5-minute buckets)")
    ax2.set_ylabel("Warm containers")
    ax2b.set_ylabel("Latency (ms)")
    ax2.set_title("Planner-assisted container state and simulated latency")

    lines_a, labels_a = ax2.get_legend_handles_labels()
    lines_b, labels_b = ax2b.get_legend_handles_labels()
    ax2.legend(lines_a + lines_b, labels_a + labels_b, fontsize=8)
    ax2.grid(alpha=0.2)

    fig.tight_layout()
    demo_plot = OUT / "final_demo.png"
    fig.savefig(demo_plot, dpi=150)
    plt.close(fig)

    demo_csv = OUT / "final_demo_results.csv"
    result_df.to_csv(demo_csv, index=False)

    print(f"      Saved plot : {demo_plot}")
    print(f"      Saved CSV  : {demo_csv}")
    print(f"      Log file   : {logger.log_path}")

    print("\n[5/5] Integrated pipeline completed.")
    print("      Azure workload -> persistence forecast -> planner")
    print("      -> simulator -> monitoring -> CSV/plots")
    print(
        "\nNOTE: The planner uses a local deterministic fallback unless "
        "a compatible LLM endpoint is configured. This demo does not "
        "claim a live Qwen3 inference or production cloud deployment."
    )


if __name__ == "__main__":
    main()
