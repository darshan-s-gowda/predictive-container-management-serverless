
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from ppo_env import ServerlessContainerEnv


MODEL_PATH = "outputs/ppo_container_controller.zip"
PREDICTION_PATH = "outputs/tft_timestamp_level_predictions.csv"
OUTPUT_DIR = "outputs"

N_SEEDS = 30
SAFETY_MARGIN = 1.0


def choose_rule_action(obs):
    predicted_load = float(obs[1])
    warm = float(obs[4])
    capacity = warm * 10.0

    if predicted_load > capacity + 10:
        return 2
    elif predicted_load > capacity:
        return 1
    elif predicted_load < capacity - 25 and warm > 1:
        return 3

    return 0


def load_workloads():
    if not os.path.exists(PREDICTION_PATH):
        raise FileNotFoundError(
            f"Missing {PREDICTION_PATH}. "
            "Run python src/evaluate_tft.py first."
        )

    df = pd.read_csv(PREDICTION_PATH)
    df = df.sort_values("time_idx").reset_index(drop=True)

    required = {
        "actual",
        "tft_prediction",
        "time_idx",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Prediction CSV is missing columns: {sorted(missing)}"
        )

    actual = df["actual"].to_numpy(dtype=np.float32)
    tft = df["tft_prediction"].to_numpy(dtype=np.float32)

    # This baseline forecast uses the previous actual workload.
    # It is a simple comparator, not a learned prediction model.
    persistence = np.concatenate(
        [actual[:1], actual[:-1]]
    ).astype(np.float32)

    # Use the training-derived threshold recorded by the evaluation
    # configuration; keep the safety margin explicit and configurable.
    adjusted_tft = np.maximum(0.0, tft + SAFETY_MARGIN)

    return actual, tft, adjusted_tft, persistence


def evaluate(policy_name, seed, actual, forecast, model=None):
    env = ServerlessContainerEnv(
        actual_workload=actual,
        predicted_workload=forecast,
        sla_ms=200,
        noise_std=2.0,
    )

    obs, _ = env.reset(seed=seed)

    total_reward = 0.0
    records = []
    terminated = False
    truncated = False

    while not (terminated or truncated):
        if policy_name == "PPO":
            action, _ = model.predict(obs, deterministic=True)
        elif policy_name == "Rule-based":
            action = choose_rule_action(obs)
        else:
            action = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        records.append(info)

    env.close()

    frame = pd.DataFrame(records)

    return {
        "policy": policy_name,
        "seed": seed,
        "steps": len(frame),
        "avg_latency_ms": frame["latency_ms"].mean(),
        "max_latency_ms": frame["latency_ms"].max(),
        "sla_violation_steps": (
            frame["sla_violation"] > 0
        ).sum(),
        "total_sla_violation_ms": frame["sla_violation"].sum(),
        "avg_energy_proxy": frame["energy"].mean(),
        "avg_warm_containers": frame["warm_containers"].mean(),
        "cold_start_events": frame["cold_start_event"].sum(),
        "total_reward": total_reward,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Missing PPO model: {MODEL_PATH}. "
            "Run python src/train_ppo.py first."
        )

    model = PPO.load(MODEL_PATH)
    actual, tft, adjusted_tft, persistence = load_workloads()

    policies = [
        ("Fixed baseline", actual, None),
        ("Rule-based", tft, None),
        ("PPO original forecast", tft, model),
        ("PPO TFT + safety margin", adjusted_tft, model),
    ]

    results = []

    for seed in range(N_SEEDS):
        for policy_name, forecast, policy_model in policies:
            result = evaluate(
                policy_name=policy_name,
                seed=seed,
                actual=actual,
                forecast=forecast,
                model=policy_model,
            )
            results.append(result)

        print(f"Completed seed {seed + 1}/{N_SEEDS}")

    results_df = pd.DataFrame(results)

    raw_path = os.path.join(
        OUTPUT_DIR, "tft_ppo_comparison_results.csv"
    )
    results_df.to_csv(raw_path, index=False)

    metrics = [
        "avg_latency_ms",
        "max_latency_ms",
        "sla_violation_steps",
        "total_sla_violation_ms",
        "avg_energy_proxy",
        "avg_warm_containers",
        "cold_start_events",
        "total_reward",
    ]

    summary = results_df.groupby("policy")[metrics].agg(["mean", "std"])
    summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in summary.columns
    ]
    summary = summary.reset_index()

    summary_path = os.path.join(
        OUTPUT_DIR, "tft_ppo_comparison_summary.csv"
    )
    summary.to_csv(summary_path, index=False)

    print("\n=== Mean results across seeds ===")
    print(
        results_df.groupby("policy")[metrics]
        .mean()
        .round(3)
        .to_string()
    )

    policy_order = [
        "Fixed baseline",
        "Rule-based",
        "PPO original forecast",
        "PPO TFT + safety margin",
    ]

    for metric, title, ylabel, filename in [
        (
            "avg_latency_ms",
            "Average latency by policy",
            "Latency (ms)",
            "tft_ppo_latency_comparison.png",
        ),
        (
            "avg_energy_proxy",
            "Average energy proxy by policy",
            "Energy proxy (arbitrary units)",
            "tft_ppo_energy_comparison.png",
        ),
        (
            "avg_warm_containers",
            "Average warm containers by policy",
            "Warm containers",
            "tft_ppo_warm_comparison.png",
        ),
    ]:
        grouped = results_df.groupby("policy")[metric]
        means = grouped.mean().reindex(policy_order)
        stds = grouped.std().reindex(policy_order)

        plt.figure(figsize=(11, 5))
        plt.bar(
            policy_order,
            means.values,
            yerr=stds.values,
            capsize=4,
        )
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xticks(rotation=18, ha="right")
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_DIR, filename),
            dpi=150,
        )
        plt.close()

    print(f"\nSaved raw results: {raw_path}")
    print(f"Saved summary: {summary_path}")
    print("Saved latency, energy, and warm-container comparison plots.")


if __name__ == "__main__":
    main()
