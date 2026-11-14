"""
Phase G: End-to-end evaluation pipeline.

Runs the same held-out Azure workload through six policies:
  1. Fixed-container (fixed-4)
  2. Rule-based controller
  3. Persistence forecast + rule controller
  4. TFT forecast + rule controller
  5. PPO controller
  6. Planner-assisted (Agentic planner → rule override)

Uses the ServerlessSimulator for a consistent, fair comparison.
Reports mean/P95 latency, cold-start count/rate, SLA violations/rate,
average/peak warm containers, resource/energy proxy, and action distribution.

Run from D:\\major_1:
    python src\\end_to_end_evaluation.py
"""

from __future__ import annotations
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

DATA_CSV = ROOT / "data" / "azure_serverless_workload.csv"
MODEL_PATH = OUT / "ppo_container_controller.zip"

SLA_MS = 200.0
N_SEEDS = 10
TEST_FRAC = 0.15            # use the last 15% as the evaluation set


# ── Data loading ───────────────────────────────────────────────────────────────
def load_azure_workload() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)
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
    df.columns = ["time_idx", "actual"]
    df["time_idx"] = pd.to_numeric(df["time_idx"], errors="coerce")
    df["actual"] = pd.to_numeric(df["actual"], errors="coerce")
    df = (
        df.dropna()
        .sort_values("time_idx")
        .drop_duplicates("time_idx", keep="last")
        .reset_index(drop=True)
    )
    return df


def build_forecasts(df: pd.DataFrame):
    """
    Build persistence and (optionally) saved TFT forecasts for the test segment.
    Returns dict of forecast_name -> np.ndarray (same length as test segment).
    """
    n = len(df)
    split = int(n * (1 - TEST_FRAC))
    test_df = df.iloc[split:].reset_index(drop=True)
    seed_actual = df.iloc[split - 1]["actual"]

    actual = test_df["actual"].to_numpy(dtype=np.float32)

    # Persistence: t = actual[t-1], seeded with last train value
    persistence = np.concatenate(
        [[seed_actual], actual[:-1]]
    ).astype(np.float32)

    # TFT: load from saved predictions if available
    orig_path = OUT / "tft_azure_holdout_predictions.csv"
    log_path = OUT / "tft_azure_log_predictions.csv"

    tft_orig = None
    tft_log = None

    if orig_path.exists():
        pdf = pd.read_csv(orig_path)
        if "time_idx" in pdf.columns and "tft_prediction" in pdf.columns:
            merged = test_df.merge(
                pdf[["time_idx", "tft_prediction"]], on="time_idx", how="inner"
            )
            if len(merged) >= len(test_df) // 2:
                tft_orig = merged["tft_prediction"].to_numpy(dtype=np.float32)

    if log_path.exists():
        ldf = pd.read_csv(log_path)
        if "time_idx" in ldf.columns and "log_tft_predicted_invocations" in ldf.columns:
            merged = test_df.merge(
                ldf[["time_idx", "log_tft_predicted_invocations"]], on="time_idx", how="inner"
            )
            if len(merged) >= len(test_df) // 2:
                tft_log = merged["log_tft_predicted_invocations"].to_numpy(dtype=np.float32)

    forecasts = {"persistence": persistence}
    if tft_orig is not None:
        forecasts["original_tft"] = tft_orig
    if tft_log is not None:
        forecasts["log_tft"] = tft_log

    return actual, forecasts, test_df


# ── Policy wrappers ────────────────────────────────────────────────────────────
def rule_action_from_obs(obs_dict: dict) -> str:
    """Rule-based action as string for the simulator."""
    predicted = obs_dict["predicted_invocations"]
    warm = obs_dict["warm_containers"]
    capacity = warm * 10.0
    latency = obs_dict["latency_ms"]

    if latency > SLA_MS * 0.9 or predicted > capacity + 15:
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


def planner_action_from_obs(obs_dict: dict) -> str:
    """Agentic planner wrapping the rule-based policy with structured output."""
    try:
        from agentic_planner import PlannerInput, plan
        inp = PlannerInput(
            time_idx=obs_dict["time_idx"],
            forecast_invocations=obs_dict["predicted_invocations"],
            actual_invocations=obs_dict["actual_invocations"],
            warm_containers=obs_dict["warm_containers"],
            sla_ms=obs_dict["sla_ms"],
            current_latency_ms=obs_dict["latency_ms"],
            energy_proxy=obs_dict["energy_proxy"],
            capacity_limit=50,
            cold_start_rate=0.05,
        )
        out = plan(inp)
        return out.recommended_action
    except Exception:
        return rule_action_from_obs(obs_dict)


def ppo_action_from_obs(obs_dict: dict, ppo_model) -> str:
    """Convert simulator obs dict to PPO numpy obs vector, get action."""
    from serverless_simulator import CAPACITY_PER_CONTAINER, CPU_BASE, CPU_SLOPE

    warm = obs_dict["warm_containers"]
    load = obs_dict["actual_invocations"]
    pred = obs_dict["predicted_invocations"]
    cpu = min(100.0, CPU_BASE + load * CPU_SLOPE)
    memory = min(100.0, 20.0 + warm * 7.0 + load * 0.8)
    capacity = warm * CAPACITY_PER_CONTAINER
    overload = max(0.0, load - capacity)
    cold = min(1.0, overload / max(load, 1.0))
    latency = obs_dict["latency_ms"]
    energy = obs_dict["energy_proxy"]
    risk = max(0.0, latency / SLA_MS - 0.8)

    obs_vec = np.array(
        [load, pred, cpu, memory, warm, cold, latency, energy, risk],
        dtype=np.float32,
    )
    action_int, _ = ppo_model.predict(obs_vec, deterministic=True)
    action_map = {
        0: "NO_OP", 1: "PREWARM", 2: "SCALE_OUT",
        3: "SCALE_IN", 4: "NO_OP", 5: "TERMINATE_IDLE",
    }
    return action_map.get(int(action_int), "NO_OP")


# ── Run one evaluation ─────────────────────────────────────────────────────────
def run_policy(
    policy_name: str,
    actual: np.ndarray,
    forecast: np.ndarray,
    policy_fn: Callable,
    seed: int,
) -> dict:
    from serverless_simulator import ServerlessSimulator

    sim = ServerlessSimulator(sla_ms=SLA_MS, initial_warm=4, seed=seed)
    hist = sim.run(actual, forecast, policy_fn)
    df = pd.DataFrame([asdict(m) for m in hist])

    return {
        "policy": policy_name,
        "seed": seed,
        "steps": len(df),
        "mean_latency_ms": round(df["latency_ms"].mean(), 2),
        "p95_latency_ms": round(df["latency_ms"].quantile(0.95), 2),
        "cold_start_events": round(df["cold_start_count"].sum(), 1),
        "cold_start_rate": round(df["cold_start_proxy"].mean(), 4),
        "sla_violations": int(df["sla_violation"].sum()),
        "sla_violation_rate": round(df["sla_violation"].mean(), 4),
        "avg_warm_containers": round(df["warm_containers"].mean(), 2),
        "peak_warm_containers": int(df["warm_containers"].max()),
        "avg_energy_proxy": round(df["energy_proxy"].mean(), 4),
        "action_noop": int((df["action"] == "NO_OP").sum()),
        "action_prewarm": int((df["action"] == "PREWARM").sum()),
        "action_scale_out": int((df["action"] == "SCALE_OUT").sum()),
        "action_scale_in": int((df["action"] == "SCALE_IN").sum()),
        "action_terminate": int((df["action"] == "TERMINATE_IDLE").sum()),
        "action_reuse": int((df["action"] == "REUSE").sum()),
    }


# ── Summary + plots ────────────────────────────────────────────────────────────
def print_summary(df: pd.DataFrame):
    metrics = [
        "mean_latency_ms", "p95_latency_ms", "cold_start_events",
        "sla_violation_rate", "avg_warm_containers", "avg_energy_proxy",
    ]
    summary = df.groupby("policy")[metrics].agg(["mean", "std"]).round(3)
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    print(summary.to_string())
    return summary


def plot_comparison(df: pd.DataFrame, policy_order: List[str]):
    """Six-panel comparison plot."""
    metrics_cfg = [
        ("mean_latency_ms", "Mean latency (ms)", "Mean latency"),
        ("p95_latency_ms", "P95 latency (ms)", "P95 latency"),
        ("sla_violation_rate", "SLA violation rate", "SLA violation rate"),
        ("avg_energy_proxy", "Energy proxy (a.u.)", "Avg energy proxy"),
        ("avg_warm_containers", "Warm containers", "Avg warm containers"),
        ("cold_start_events", "Cold-start events", "Cold-start events"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for ax, (metric, ylabel, title) in zip(axes, metrics_cfg):
        grouped = df.groupby("policy")[metric]
        means = grouped.mean().reindex(policy_order)
        stds = grouped.std().reindex(policy_order)
        bars = ax.bar(range(len(policy_order)), means.values,
                      yerr=stds.values, capsize=4, width=0.6)
        ax.set_xticks(range(len(policy_order)))
        ax.set_xticklabels(
            [p.replace(" + ", "\n+ ") for p in policy_order],
            fontsize=7, rotation=20, ha="right"
        )
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9)

    fig.suptitle(
        "End-to-end policy comparison — Azure test segment (simulated metrics)",
        fontsize=11,
    )
    fig.tight_layout()
    path = OUT / "end_to_end_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=== Phase G: End-to-End Evaluation ===\n")

    df_full = load_azure_workload()
    actual, forecasts, test_df = build_azure_test(df_full)

    print(f"Test segment: {len(actual)} buckets")
    print(f"Forecasts available: {list(forecasts.keys())}\n")

    # Load PPO model if available
    ppo_model = None
    if MODEL_PATH.exists():
        try:
            from stable_baselines3 import PPO
            ppo_model = PPO.load(str(MODEL_PATH))
            print("Loaded PPO model.")
        except Exception as e:
            print(f"Could not load PPO: {e}")
    else:
        print("PPO model not found — will skip PPO policy. Run retrain_ppo.py first.")

    forecast_for_rule = forecasts.get("log_tft",
                        forecasts.get("original_tft",
                        forecasts["persistence"]))
    forecast_name = (
        "log_tft" if "log_tft" in forecasts
        else "original_tft" if "original_tft" in forecasts
        else "persistence"
    )

    # Policies
    from serverless_simulator import fixed_policy, rule_based_policy, persistence_policy

    policies = [
        ("Fixed-4", actual, fixed_policy(4)),
        ("Rule-based", forecast_for_rule, rule_based_policy(SLA_MS)),
        ("Persistence + Rule", forecasts["persistence"], persistence_policy(SLA_MS)),
        ("TFT + Rule", forecast_for_rule, rule_based_policy(SLA_MS)),
        ("Planner-assisted", forecast_for_rule,
         lambda obs: planner_action_from_obs(obs)),
    ]

    if ppo_model is not None:
        _ppo = ppo_model  # capture in closure
        policies.append((
            "PPO",
            forecast_for_rule,
            lambda obs: ppo_action_from_obs(obs, _ppo),
        ))

    results = []
    policy_order = []

    for policy_name, forecast, pol_fn in policies:
        policy_order.append(policy_name)
        min_len = min(len(actual), len(forecast))
        act = actual[:min_len]
        fcast = forecast[:min_len]

        for seed in range(N_SEEDS):
            row = run_policy(policy_name, act, fcast, pol_fn, seed)
            results.append(row)

        print(f"  Completed: {policy_name}")

    results_df = pd.DataFrame(results)
    raw_path = OUT / "end_to_end_results.csv"
    results_df.to_csv(raw_path, index=False)

    print("\n=== End-to-end summary (mean ± std across seeds) ===")
    summary = print_summary(results_df)
    summary_path = OUT / "end_to_end_summary.csv"
    summary.to_csv(summary_path)

    plot_comparison(results_df, policy_order)

    print(f"\nSaved: {raw_path}")
    print(f"Saved: {summary_path}")
    print("\nNote: all latency and energy figures are simulated, not measured in production.")


def build_azure_test(df_full: pd.DataFrame):
    """Extract test segment and return actual workload + forecasts."""
    n = len(df_full)
    split = int(n * (1 - TEST_FRAC))
    test_df = df_full.iloc[split:].reset_index(drop=True)
    seed_actual = float(df_full.iloc[split - 1]["actual"])

    actual = test_df["actual"].to_numpy(dtype=np.float32)
    persistence = np.concatenate(
        [[seed_actual], actual[:-1]]
    ).astype(np.float32)

    orig_path = OUT / "tft_azure_holdout_predictions.csv"
    log_path = OUT / "tft_azure_log_predictions.csv"

    forecasts = {"persistence": persistence}

    if orig_path.exists():
        pdf = pd.read_csv(orig_path)
        if "time_idx" in pdf.columns and "tft_prediction" in pdf.columns:
            merged = test_df.merge(pdf[["time_idx", "tft_prediction"]], on="time_idx", how="inner")
            if len(merged) >= len(test_df) // 2:
                arr = merged["tft_prediction"].to_numpy(dtype=np.float32)
                # pad/clip to test length
                arr = np.pad(arr, (0, max(0, len(actual) - len(arr))))[:len(actual)]
                forecasts["original_tft"] = arr

    if log_path.exists():
        ldf = pd.read_csv(log_path)
        if "time_idx" in ldf.columns and "log_tft_predicted_invocations" in ldf.columns:
            merged = test_df.merge(
                ldf[["time_idx", "log_tft_predicted_invocations"]], on="time_idx", how="inner"
            )
            if len(merged) >= len(test_df) // 2:
                arr = merged["log_tft_predicted_invocations"].to_numpy(dtype=np.float32)
                arr = np.pad(arr, (0, max(0, len(actual) - len(arr))))[:len(actual)]
                forecasts["log_tft"] = arr

    return actual, forecasts, test_df


if __name__ == "__main__":
    main()
