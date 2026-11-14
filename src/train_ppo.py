
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from ppo_env import ServerlessContainerEnv


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = OUTPUT_DIR / "ppo_container_controller.zip"


def make_training_workload(seed, length=288):
    """
    Generate a varied synthetic training workload.
    The environment remains a simulation, not a real cloud deployment.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(length)

    daily = 20 + 17 * np.maximum(
        0, np.sin((t / 12 - 7) * np.pi / 12)
    )

    bursts = np.zeros(length)
    for center in rng.choice(
        np.arange(30, length - 30),
        size=5,
        replace=False,
    ):
        bursts += rng.uniform(8, 22) * np.exp(
            -0.5 * ((t - center) / rng.uniform(3, 12)) ** 2
        )

    actual = np.maximum(
        2,
        daily + bursts + rng.normal(0, 3, length),
    ).astype(np.float32)

    # Training forecast: noisy estimate, not the actual future.
    predicted = np.maximum(
        0,
        actual + rng.normal(0, 3, length),
    ).astype(np.float32)

    return actual, predicted


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Validate the environment interface first.
    check_env(
        ServerlessContainerEnv(),
        warn=True,
    )

    def make_env():
        actual, predicted = make_training_workload(
            np.random.randint(0, 1_000_000)
        )
        return Monitor(
            ServerlessContainerEnv(
                actual_workload=actual,
                predicted_workload=predicted,
                sla_ms=200,
                noise_std=1.0,
            )
        )

    # Vectorized environments provide diverse training episodes.
    from stable_baselines3.common.env_util import make_vec_env

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
        ent_coef=0.02,
        clip_range=0.2,
        verbose=1,
        seed=42,
        policy_kwargs={"net_arch": [64, 64]},
    )

    print("Training PPO with varied synthetic workloads...")
    model.learn(total_timesteps=100_000)

    model.save(str(MODEL_PATH))
    env.close()

    print(f"\nSaved retrained PPO model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
