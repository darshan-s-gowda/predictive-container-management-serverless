
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
MODEL_PATH = OUT / "ppo_container_controller.zip"

ORIGINAL_TFT_PATH = OUT / "tft_azure_holdout_predictions.csv"
LOG_TFT_PATH = OUT / "tft_azure_log_predictions.csv"

SLA_MS = 200.0
N_SEEDS = 10
TEST_FRAC = 0.15


# ---------------------------------------------------------
# 1. Load workload and align forecasts by timestamp
# ---------------------------------------------------------

def load_azure_workload() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)

    time_col = next(
        c for c in df.columns
        if c.lower() in ("time_idx", "time_index", "timestamp_idx")
    )

    load_col = next(
        c for c in df.columns
        if c.lower() in (
            "invocation_rate",
            "invocations",
            "invocation_count",
            "request_count",
            "load",
        )
    )

    df = df[[time_col, load_col]].copy()
    df.columns = ["time_idx", "actual"]

    df["time_idx"] = pd.to_numeric(
        df["time_idx"], errors="coerce"
    )
    df["actual"] = pd.to_numeric(
        df["actual"], errors="coerce"
    )

    df = (
        df.dropna()
        .sort_values("time_idx")
        .drop_duplicates("time_idx", keep="last")
        .reset_index(drop=True)
    )

    return df


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

    pred_df = pred_df[["time_idx", prediction_column]].copy()

    pred_df["time_idx"] = pd.to_numeric(
        pred_df["time_idx"], errors="coerce"
    )
    pred_df[prediction_column] = pd.to_numeric(
        pred_df[prediction_column], errors="coerce"
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
            "evaluation forecasts. Evaluation stopped; "
            "no padding or timestamp shifting is allowed."
        )

    values = aligned[prediction_column].to_numpy(
        dtype=np.float32
    )

    return np.maximum(values, 0.0)


def build_azure_test(df_full: pd.DataFrame):
    n = len(df_full)
    split = int(n * (1 - TEST_FRAC))

    test_df = df_full.iloc[split:].reset_index(drop=True)

    if split <= 0 or len(test_df) == 0:
        raise ValueError("Invalid train/test split.")

    actual = test_df["actual"].to_numpy(
        dtype=np.float32
    )

    previous_actual = float(
        df_full.iloc[split - 1]["actual"]
    )

    # Persistence forecast: previous observed bucket.
    persistence = np.concatenate(
        [[previous_actual], actual[:-1]]
    ).astype(np.float32)

    original_tft = load_aligned_prediction(
        test_df,
        ORIGINAL_TFT_PATH,
        "tft_prediction",
    )

    log_tft = load_aligned_prediction(
        test_df,
        LOG_TFT_PATH,
        "log_tft_predicted_invocations",
    )

    forecasts = {
        "persistence": persistence,
        "original_tft": original_tft,
        "log_tft": log_tft,
    }

    lengths = {len(actual)}
    lengths.update(len(v) for v in forecasts.values())

    if len(lengths) != 1:
        raise ValueError(
            "Actual workload and forecast lengths do not match."
        )

    return actual, forecasts, test_df


# ---------------------------------------------------------
# 2. Policy definitions
# ---------------------------------------------------------

def reactive_rule_action(obs: dict) -> str:
    """
    Reactive controller using only the previous observed bucket.
    It responds to measured latency rather than a future forecast.
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
    """Structured planner using observed state and next-bucket forecast."""

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


def ppo_action_from_obs(obs: dict, ppo_model) -> str:
    """
    Construct the same nine-feature observation used in PPO training.
    """

    from serverless_simulator import (
        CAPACITY_PER_CONTAINER,
        CPU_BASE,
        CPU_SLOPE,
    )

    warm = obs["warm_containers"]
    load = obs["actual_invocations"]
    predicted = obs["predicted_invocations"]

    cpu = min(100.0, CPU_BASE + load * CPU_SLOPE)
    memory = min(100.0, 20.0 + warm * 7.0 + load * 0.8)

    capacity = warm * CAPACITY_PER_CONTAINER
    overload = max(0.0, load - capacity)

    cold = min(
        1.0,
        overload / max(load, 1.0),
    )

    latency = obs["latency_ms"]
    energy = obs["energy_proxy"]
    risk = max(0.0, latency / SLA_MS - 0.8)

    observation = np.array(
        [
            load,
            predicted,
            cpu,
            memory,
            warm,
            cold,
            latency,
            energy,
            risk,
        ],
        dtype=np.float32,
    )

    action, _ = ppo_model.predict(
        observation,
        deterministic=True,
    )

    action_map = {
        0: "NO_OP",
        1: "PREWARM",
        2: "SCALE_OUT",
        3: "SCALE_IN",
        4: "NO_OP",
        5: "TERMINATE_IDLE",
    }

    return action_map.get(int(action), "NO_OP")


# ---------------------------------------------------------
# 3. Run one policy
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
        "p95_latency_ms": round(
            frame["latency_ms"].quantile(0.95), 2
        ),

        "cold_start_events": round(
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
# 4. Summary and comparison plot
# ---------------------------------------------------------

def print_summary(frame: pd.DataFrame):
    metrics = [
        "mean_latency_ms",
        "p95_latency_ms",
        "cold_start_events",
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
        ("cold_start_events", "Cold-start events"),
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
            [name.replace(" + ", "\n+ ") for name in policy_order],
            fontsize=7,
            rotation=25,
            ha="right",
        )

        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle(
        "End-to-end policy comparison "
        "(Azure workload, simulated metrics)",
        fontsize=13,
    )

    fig.tight_layout()

    path = OUT / "end_to_end_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"Saved comparison plot: {path}")


# ---------------------------------------------------------
# 5. Main evaluation
# ---------------------------------------------------------

def main():
    print("=" * 65)
    print("END-TO-END SERVERLESS CONTAINER EVALUATION")
    print("All latency and energy metrics are simulated proxies.")
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

    print("\nForecasts aligned successfully:")

    for name, values in forecasts.items():
        print(f"  {name}: {len(values)} predictions")

    # Load trained PPO model.
    ppo_model = None

    if MODEL_PATH.exists():
        try:
            from stable_baselines3 import PPO

            ppo_model = PPO.load(str(MODEL_PATH))
            print("\nPPO model loaded successfully.")

        except Exception as exc:
            print(f"\nPPO model could not be loaded: {exc}")

    else:
        print("\nPPO model not found. PPO evaluation will be skipped.")

    from serverless_simulator import (
        fixed_policy,
        rule_based_policy,
        persistence_policy,
    )

    # Each policy uses the same actual workload.
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

    if ppo_model is not None:
        policies.append(
            (
                "PPO",
                forecasts["log_tft"],
                lambda obs: ppo_action_from_obs(
                    obs, ppo_model
                ),
            )
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

    # Save per-seed results.
    raw_path = OUT / "end_to_end_results.csv"
    results_df.to_csv(raw_path, index=False)

    # Save summary statistics.
    print("\n" + "=" * 65)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 65)

    summary = print_summary(results_df)

    summary_path = OUT / "end_to_end_summary.csv"
    summary.to_csv(summary_path)

    # Save plots.
    plot_comparison(
        results_df,
        policy_order,
    )

    print("\nEvaluation completed.")
    print(f"Detailed results: {raw_path}")
    print(f"Summary results: {summary_path}")

    print(
        "\nImportant: workload comes from the Azure trace, "
        "but latency, energy, cold starts and container "
        "metrics are generated by the simulator."
    )


if __name__ == "__main__":
    main()
