"""
Phase B: PPO retraining with fixed reward shaping and validation.

Trains PPO on diverse synthetic workloads, evaluates on held-out traces,
checks action distribution (no-op should not dominate), compares against
fixed-container, rule-based, and persistence-driven baselines.

Run from D:\\major_1:
    python src\\retrain_ppo.py
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = OUTPUT_DIR / "ppo_container_controller.zip"


# ── Workload generators ────────────────────────────────────────────────────────
def make_workload(seed: int, length: int = 288, kind: str = "mixed"):
    """Generate synthetic workload and a noisy persistence forecast."""
    rng = np.random.default_rng(seed)
    t = np.arange(length, dtype=np.float32)

    if kind == "daily":
        base = 20 + 25 * np.maximum(0, np.sin((t / 12 - 7) * np.pi / 12))
    elif kind == "bursty":
        base = np.full(length, 15.0)
        for center in rng.integers(30, length - 30, size=8):
            base += rng.uniform(15, 40) * np.exp(
                -0.5 * ((t - center) / rng.uniform(3, 10)) ** 2
            )
    elif kind == "steady":
        base = np.full(length, rng.uniform(20, 50))
    else:  # mixed
        daily = 20 + 17 * np.maximum(0, np.sin((t / 12 - 7) * np.pi / 12))
        bursts = np.zeros(length)
        for center in rng.integers(30, length - 30, size=5):
            bursts += rng.uniform(8, 22) * np.exp(
                -0.5 * ((t - center) / rng.uniform(3, 12)) ** 2
            )
        base = daily + bursts

    actual = np.maximum(2, base + rng.normal(0, 3, length)).astype(np.float32)
    predicted = np.maximum(0, actual + rng.normal(0, 4, length)).astype(np.float32)
    return actual, predicted


# ── Baseline policies ──────────────────────────────────────────────────────────
def choose_rule_action(obs) -> int:
    predicted_load = float(obs[1])
    warm = float(obs[4])
    capacity = warm * 10.0
    latency = float(obs[6])
    if latency > 190 or predicted_load > capacity + 15:
        return 2
    elif predicted_load > capacity + 5:
        return 1
    elif predicted_load < capacity - 20 and warm > 1:
        return 5
    elif predicted_load < capacity - 10 and warm > 1:
        return 3
    return 0


def fixed_action(target_warm: int, obs) -> int:
    warm = int(obs[4])
    if warm < target_warm:
        return 1
    elif warm > target_warm:
        return 3
    return 0


# ── Evaluation function ────────────────────────────────────────────────────────
def evaluate_policy(
    policy_name: str,
    seed: int,
    actual: np.ndarray,
    forecast: np.ndarray,
    model=None,
    fixed_warm: int = 4,
) -> dict:
    from ppo_env import ServerlessContainerEnv

    env = ServerlessContainerEnv(
        actual_workload=actual,
        predicted_workload=forecast,
        sla_ms=200,
        noise_std=1.5,
    )
    obs, _ = env.reset(seed=seed)

    records = []
    actions_taken = []
    total_reward = 0.0
    terminated = truncated = False

    while not (terminated or truncated):
        if policy_name == "PPO":
            action, _ = model.predict(obs, deterministic=True)
        elif policy_name == "Rule-based":
            action = choose_rule_action(obs)
        elif policy_name == "Fixed-4":
            action = fixed_action(fixed_warm, obs)
        else:  # No-op / fixed
            action = 0

        action = int(action)
        actions_taken.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        records.append(info)

    env.close()
    frame = pd.DataFrame(records)

    # Action distribution
    action_dist = np.bincount(actions_taken, minlength=6)
    noop_frac = action_dist[0] / max(len(actions_taken), 1)

    return {
        "policy": policy_name,
        "seed": seed,
        "steps": len(frame),
        "avg_latency_ms": frame["latency_ms"].mean(),
        "max_latency_ms": frame["latency_ms"].max(),
        "sla_violation_steps": (frame["sla_violation"] > 0).sum(),
        "sla_violation_rate": (frame["sla_violation"] > 0).mean(),
        "avg_energy_proxy": frame["energy"].mean(),
        "avg_warm_containers": frame["warm_containers"].mean(),
        "cold_start_events": frame["cold_start_event"].sum(),
        "total_reward": total_reward,
        "noop_fraction": round(noop_frac, 4),
        "action_dist": action_dist.tolist(),
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=== Phase B: PPO Retraining & Validation ===\n")

    # ── Train ──────────────────────────────────────────────────────────────────
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor
    from ppo_env import ServerlessContainerEnv

    TRAIN_KINDS = ["mixed", "daily", "bursty", "steady"]

    def make_env():
        kind = np.random.choice(TRAIN_KINDS)
        actual, predicted = make_workload(
            seed=np.random.randint(0, 1_000_000), kind=kind
        )
        return Monitor(
            ServerlessContainerEnv(
                actual_workload=actual,
                predicted_workload=predicted,
                sla_ms=200,
                noise_std=1.0,
            )
        )

    env = make_vec_env(make_env, n_envs=4, seed=42)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.03,          # slightly higher entropy to discourage no-op collapse
        clip_range=0.2,
        verbose=1,
        seed=42,
        policy_kwargs={"net_arch": [128, 128]},
    )

    print("Training PPO on diverse synthetic workloads (200,000 steps)...")
    model.learn(total_timesteps=200_000)
    model.save(str(MODEL_PATH))
    env.close()
    print(f"Saved: {MODEL_PATH}\n")

    # ── Evaluate across multiple seeds and workload kinds ──────────────────────
    N_SEEDS = 20
    EVAL_KINDS = ["mixed", "daily", "bursty"]
    results = []

    policies_cfg = [
        ("Fixed-4", None),
        ("Rule-based", None),
        ("PPO", model),
    ]

    for kind in EVAL_KINDS:
        for seed in range(N_SEEDS):
            actual, predicted = make_workload(seed=seed + 1000, kind=kind)
            persistence = np.concatenate([[actual[0]], actual[:-1]])

            for policy_name, pol_model in policies_cfg:
                forecast = predicted if policy_name in ("Rule-based", "PPO") else actual
                row = evaluate_policy(
                    policy_name=policy_name,
                    seed=seed,
                    actual=actual,
                    forecast=forecast,
                    model=pol_model,
                )
                row["workload_kind"] = kind
                results.append(row)

        print(f"Completed {kind} workload ({N_SEEDS} seeds × {len(policies_cfg)} policies)")

    results_df = pd.DataFrame(results)
    results_df.drop(columns=["action_dist"], inplace=True)

    raw_path = OUTPUT_DIR / "ppo_retrain_results.csv"
    results_df.to_csv(raw_path, index=False)

    # ── Summary ────────────────────────────────────────────────────────────────
    metrics = [
        "avg_latency_ms", "sla_violation_rate", "avg_energy_proxy",
        "avg_warm_containers", "cold_start_events", "noop_fraction",
    ]

    summary = (
        results_df.groupby(["policy", "workload_kind"])[metrics]
        .agg(["mean", "std"])
    )
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    summary_path = OUTPUT_DIR / "ppo_retrain_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n=== PPO Evaluation Summary (mean across seeds) ===")
    pivot = results_df.groupby(["policy", "workload_kind"])[metrics].mean().round(3)
    print(pivot.to_string())

    # ── Action distribution check ──────────────────────────────────────────────
    ppo_rows = results_df[results_df["policy"] == "PPO"]
    avg_noop = ppo_rows["noop_fraction"].mean()
    print(f"\nPPO average no-op fraction: {avg_noop:.2%}")
    if avg_noop > 0.8:
        print("  WARNING: No-op fraction > 80%. Policy may not be scaling appropriately.")
    else:
        print("  OK: No-op fraction is acceptable.")

    # ── Plot ───────────────────────────────────────────────────────────────────
    policy_order = ["Fixed-4", "Rule-based", "PPO"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, metric, ylabel, title in zip(
        axes,
        ["avg_latency_ms", "avg_energy_proxy", "sla_violation_rate"],
        ["Latency (ms)", "Energy proxy (a.u.)", "SLA violation rate"],
        ["Average Latency", "Energy Proxy", "SLA Violation Rate"],
    ):
        grouped = results_df.groupby("policy")[metric]
        means = grouped.mean().reindex(policy_order)
        stds = grouped.std().reindex(policy_order)
        ax.bar(policy_order, means.values, yerr=stds.values, capsize=5)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(policy_order, rotation=15, ha="right")

    fig.suptitle("PPO vs Baselines — All workload kinds (simulated)")
    fig.tight_layout()
    plot_path = OUTPUT_DIR / "ppo_retrain_comparison.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"\nSaved: {raw_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
