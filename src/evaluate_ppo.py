
import numpy as np
from stable_baselines3 import PPO
from ppo_env import ServerlessContainerEnv


MODEL_PATH = "outputs/ppo_container_controller.zip"


def evaluate(mode, seed=42):
    env = ServerlessContainerEnv()
    obs, _ = env.reset(seed=seed)

    model = PPO.load(MODEL_PATH) if mode == "PPO" else None
    records = []
    total_reward = 0.0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            # Baseline: keep the warm-container count unchanged
            action = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        records.append(info)

    env.close()

    return {
        "policy": mode,
        "steps": len(records),
        "avg_latency_ms": np.mean(
            [x["latency_ms"] for x in records]
        ),
        "max_latency_ms": np.max(
            [x["latency_ms"] for x in records]
        ),
        "sla_violation_steps": sum(
            x["sla_violation"] > 0 for x in records
        ),
        "avg_energy_proxy": np.mean(
            [x["energy"] for x in records]
        ),
        "avg_warm_containers": np.mean(
            [x["warm_containers"] for x in records]
        ),
        "total_reward": total_reward,
    }


if __name__ == "__main__":
    results = [
        evaluate("Fixed baseline", seed=42),
        evaluate("PPO", seed=42),
    ]

    for result in results:
        print("\n", "=" * 45)
        for key, value in result.items():
            print(f"{key}: {value}")
