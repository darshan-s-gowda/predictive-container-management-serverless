"""
Phase F: Monitoring and Feedback Module.

Logs the following per step:
  - time_idx, actual and predicted workload
  - warm-container count and chosen action
  - simulated response latency (mean and P95)
  - cold-start count / proxy
  - CPU / memory
  - energy proxy and its formula
  - SLA threshold and violation flag
  - policy / model version and random seed

Writes structured JSONL event log and produces monitoring plots.
Optional: expose Prometheus-style metrics as a text endpoint.

Run from D:\\major_1:
    python src\\monitoring.py
"""

from __future__ import annotations
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
LOG_DIR = OUT / "monitoring_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ── Log entry ─────────────────────────────────────────────────────────────────
@dataclass
class MonitoringEvent:
    # Identity
    wall_clock_utc: str
    policy_name: str
    policy_version: str
    seed: int
    # Workload
    time_idx: int
    actual_invocations: float
    predicted_invocations: float
    forecast_model: str
    # Container state
    warm_containers: int
    action: str
    container_delta: int
    # Latency
    latency_ms: float
    p95_latency_ms: float
    # Cold start
    cold_start_count: float
    cold_start_proxy: float
    # Resources
    cpu_pct: float
    memory_pct: float
    # Energy proxy
    # Formula: energy_proxy = warm_containers * 0.8 + cpu_pct * 0.015
    energy_proxy: float
    energy_formula: str
    # SLA
    sla_ms: float
    sla_violated: bool
    sla_excess_ms: float
    # Reward (if RL policy)
    reward: Optional[float] = None


class MonitoringLogger:
    """
    Append-only JSONL logger for monitoring events.
    Thread-unsafe; designed for single-process use.
    """

    ENERGY_FORMULA = "warm_containers * 0.8 + cpu_pct * 0.015"

    def __init__(
        self,
        policy_name: str,
        policy_version: str = "1.0",
        seed: int = 0,
        forecast_model: str = "persistence",
        sla_ms: float = 200.0,
        log_path: Optional[Path] = None,
    ):
        self.policy_name = policy_name
        self.policy_version = policy_version
        self.seed = seed
        self.forecast_model = forecast_model
        self.sla_ms = sla_ms

        if log_path is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            log_path = LOG_DIR / f"{policy_name.replace(' ', '_')}_{ts}.jsonl"

        self.log_path = log_path
        self._file = open(log_path, "a", encoding="utf-8")
        self.events: List[MonitoringEvent] = []

    def log(
        self,
        time_idx: int,
        actual_invocations: float,
        predicted_invocations: float,
        warm_containers: int,
        action: str,
        container_delta: int,
        latency_ms: float,
        p95_latency_ms: float,
        cold_start_count: float,
        cold_start_proxy: float,
        cpu_pct: float,
        memory_pct: float,
        energy_proxy: float,
        sla_violated: bool,
        sla_excess_ms: float,
        reward: Optional[float] = None,
    ) -> MonitoringEvent:
        event = MonitoringEvent(
            wall_clock_utc=datetime.now(timezone.utc).isoformat(),
            policy_name=self.policy_name,
            policy_version=self.policy_version,
            seed=self.seed,
            time_idx=time_idx,
            actual_invocations=round(actual_invocations, 2),
            predicted_invocations=round(predicted_invocations, 2),
            forecast_model=self.forecast_model,
            warm_containers=warm_containers,
            action=action,
            container_delta=container_delta,
            latency_ms=round(latency_ms, 2),
            p95_latency_ms=round(p95_latency_ms, 2),
            cold_start_count=round(cold_start_count, 2),
            cold_start_proxy=round(cold_start_proxy, 4),
            cpu_pct=round(cpu_pct, 2),
            memory_pct=round(memory_pct, 2),
            energy_proxy=round(energy_proxy, 4),
            energy_formula=self.ENERGY_FORMULA,
            sla_ms=self.sla_ms,
            sla_violated=sla_violated,
            sla_excess_ms=round(sla_excess_ms, 2),
            reward=round(reward, 4) if reward is not None else None,
        )
        self._file.write(json.dumps(asdict(event)) + "\n")
        self._file.flush()
        self.events.append(event)
        return event

    def close(self):
        self._file.close()

    # ── Feedback integration (Milestone 7) ────────────────────────────────────
    def record_feedback(self, decision_record, step_metrics: dict) -> dict:
        """
        Link a DecisionRecord to its observed outcome metrics and write to
        outputs/feedback_log.jsonl.

        Parameters
        ----------
        decision_record : DecisionRecord (dataclass from decision_engine.py)
        step_metrics    : dict with keys like latency_ms, sla_violation,
                          energy_proxy, warm_containers, cold_start_proxy

        Returns
        -------
        feedback dict written to log
        """
        from dataclasses import asdict as _asdict
        import os as _os

        feedback = {
            "decision_correlation_id": getattr(decision_record, "correlation_id", ""),
            "timestamp_step":          getattr(decision_record, "timestamp_step", -1),
            "final_action":            getattr(decision_record, "final_action", "?"),
            "planner_provider":        getattr(decision_record, "planner_provider", "?"),
            "shield_overridden":       getattr(decision_record, "shield_overridden", False),
            # Observed outcomes
            "observed_latency_ms":     float(step_metrics.get("latency_ms", 0.0)),
            "observed_sla_violation":  bool(step_metrics.get("sla_violation", False)),
            "observed_energy_proxy":   float(step_metrics.get("energy_proxy", 0.0)),
            "observed_warm_containers":int(step_metrics.get("warm_containers", 0)),
            "observed_cold_start_proxy": float(step_metrics.get("cold_start_proxy", 0.0)),
            "feedback_note":           str(step_metrics.get("feedback_note", "")),
            "wall_clock_utc":          datetime.now(timezone.utc).isoformat(),
        }

        fb_path = str(self.log_path).replace(str(LOG_DIR), str(OUT)).replace(
            ".jsonl", "_feedback.jsonl"
        )
        # Always write to canonical feedback log
        fb_path = str(OUT / "feedback_log.jsonl")
        _os.makedirs(_os.path.dirname(fb_path), exist_ok=True)
        with open(fb_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(feedback) + "\n")

        return feedback

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(e) for e in self.events])

    # ── Prometheus-style text exposition ───────────────────────────────────────
    def prometheus_snapshot(self) -> str:
        """
        Returns a Prometheus-format text string of the latest metrics.
        (Write to a file or expose via HTTP if needed.)
        """
        if not self.events:
            return "# No events logged yet\n"

        last = self.events[-1]
        lines = [
            "# HELP serverless_warm_containers Current warm container count",
            "# TYPE serverless_warm_containers gauge",
            f'serverless_warm_containers{{policy="{last.policy_name}"}} {last.warm_containers}',
            "# HELP serverless_latency_ms Simulated mean response latency in ms",
            "# TYPE serverless_latency_ms gauge",
            f'serverless_latency_ms{{policy="{last.policy_name}"}} {last.latency_ms}',
            "# HELP serverless_sla_violated SLA violation flag (1=violated)",
            "# TYPE serverless_sla_violated gauge",
            f'serverless_sla_violated{{policy="{last.policy_name}"}} {int(last.sla_violated)}',
            "# HELP serverless_energy_proxy Energy proxy value",
            "# TYPE serverless_energy_proxy gauge",
            f'serverless_energy_proxy{{policy="{last.policy_name}"}} {last.energy_proxy}',
            "# HELP serverless_cold_start_count Estimated cold starts this bucket",
            "# TYPE serverless_cold_start_count gauge",
            f'serverless_cold_start_count{{policy="{last.policy_name}"}} {last.cold_start_count}',
        ]
        return "\n".join(lines) + "\n"


# ── Dashboard plot ─────────────────────────────────────────────────────────────
def plot_monitoring_dashboard(
    df: pd.DataFrame,
    policy_name: str,
    sla_ms: float = 200.0,
    output_path: Optional[Path] = None,
):
    """
    Four-panel monitoring dashboard:
      1. Workload (actual vs predicted) + warm containers
      2. Latency (mean and P95) with SLA line
      3. Energy proxy over time
      4. Cold-start proxy and SLA violations
    """
    if output_path is None:
        output_path = OUT / f"monitoring_dashboard_{policy_name.replace(' ', '_')}.png"

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.3)

    t = df["time_idx"].to_numpy()

    # ── Panel 1: Workload ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, df["actual_invocations"], label="Actual", lw=1.2)
    ax1.plot(t, df["predicted_invocations"], label="Predicted", lw=1.0, alpha=0.7)
    ax1b = ax1.twinx()
    ax1b.step(t, df["warm_containers"], where="mid", color="tab:green",
              lw=1.0, alpha=0.6, label="Warm containers")
    ax1b.set_ylabel("Warm containers", color="tab:green")
    ax1.set_title(f"{policy_name} — Workload & containers")
    ax1.set_xlabel("Bucket index")
    ax1.set_ylabel("Invocations")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines1b, labels1b = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines1b, labels1 + labels1b, fontsize=7)

    # ── Panel 2: Latency ───────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, df["latency_ms"], label="Mean latency", lw=1.2)
    if "p95_latency_ms" in df.columns:
        ax2.plot(t, df["p95_latency_ms"], label="P95 latency", lw=1.0, alpha=0.7)
    ax2.axhline(sla_ms, color="red", linestyle="--", lw=1.0,
                label=f"SLA ({sla_ms:.0f} ms)")
    ax2.set_title(f"{policy_name} — Latency")
    ax2.set_xlabel("Bucket index")
    ax2.set_ylabel("Latency (ms)")
    ax2.legend(fontsize=7)

    # ── Panel 3: Energy proxy ──────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(t, df["energy_proxy"], color="tab:orange", lw=1.2)
    ax3.fill_between(t, df["energy_proxy"], alpha=0.2, color="tab:orange")
    ax3.set_title(f"{policy_name} — Energy proxy")
    ax3.set_xlabel("Bucket index")
    ax3.set_ylabel("Energy proxy (a.u.)")
    formula = "warm × 0.8 + cpu% × 0.015"
    ax3.text(0.02, 0.97, f"Formula: {formula}", transform=ax3.transAxes,
             fontsize=7, va="top")

    # ── Panel 4: Cold starts & SLA violations ──────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    if "cold_start_proxy" in df.columns:
        ax4.bar(t, df["cold_start_proxy"], label="Cold-start proxy", alpha=0.6, width=1)
    if "sla_violated" in df.columns:
        viol = df["sla_violated"].astype(float)
        ax4b = ax4.twinx()
        ax4b.step(t, viol, where="mid", color="red", lw=0.8, alpha=0.7,
                  label="SLA violated")
        ax4b.set_ylabel("SLA violated (1=yes)", color="red")
        ax4b.set_ylim(-0.05, 1.5)
    ax4.set_title(f"{policy_name} — Cold starts & SLA violations")
    ax4.set_xlabel("Bucket index")
    ax4.set_ylabel("Cold-start proxy")
    ax4.legend(fontsize=7)

    fig.suptitle(
        f"Monitoring Dashboard — {policy_name} (simulated, not measured in cloud)",
        fontsize=11,
    )
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved monitoring dashboard: {output_path}")
    return output_path


# ── Standalone demo ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from serverless_simulator import ServerlessSimulator, rule_based_policy

    # Generate synthetic workload
    rng = np.random.default_rng(42)
    n = 150
    t_arr = np.arange(n)
    workload = (30 + 20 * np.sin(t_arr * np.pi / 24)
                + rng.normal(0, 5, n)).clip(min=0).astype(np.float32)
    persistence = np.concatenate([[workload[0]], workload[:-1]])

    policy = rule_based_policy(sla_ms=200.0)
    sim = ServerlessSimulator(seed=42)
    hist = sim.run(workload, persistence, policy)

    logger = MonitoringLogger(
        policy_name="Rule-based",
        policy_version="1.0",
        seed=42,
        forecast_model="persistence",
        sla_ms=200.0,
    )

    for m in hist:
        logger.log(
            time_idx=m.time_idx,
            actual_invocations=m.actual_invocations,
            predicted_invocations=m.predicted_invocations,
            warm_containers=m.warm_containers,
            action=m.action,
            container_delta=m.container_delta,
            latency_ms=m.latency_ms,
            p95_latency_ms=m.p95_latency_ms,
            cold_start_count=m.cold_start_count,
            cold_start_proxy=m.cold_start_proxy,
            cpu_pct=m.cpu_pct,
            memory_pct=m.memory_pct,
            energy_proxy=m.energy_proxy,
            sla_violated=m.sla_violation,
            sla_excess_ms=m.sla_excess_ms,
        )

    df = logger.to_dataframe()
    logger.close()

    # Print Prometheus snapshot
    print("\n=== Prometheus snapshot (last step) ===")
    print(logger.prometheus_snapshot())

    # Generate dashboard
    plot_monitoring_dashboard(df, policy_name="Rule-based", sla_ms=200.0)

    # Summary
    print("\n=== Monitoring summary ===")
    print(f"Steps logged    : {len(df)}")
    print(f"Avg latency     : {df['latency_ms'].mean():.1f} ms")
    print(f"SLA violations  : {df['sla_violated'].sum()}")
    print(f"Avg energy proxy: {df['energy_proxy'].mean():.3f}")
    print(f"Log written to  : {logger.log_path}")
