
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ServerlessContainerEnv(gym.Env):
    """
    Simulated serverless container environment.

    Actual workload and predicted workload are separate inputs.
    Actions: 0=no-op, 1=+1 warm, 2=+2 warm,
             3=-1 warm, 4=no-op, 5=-2 warm.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        actual_workload=None,
        predicted_workload=None,
        sla_ms=200,
        initial_warm=4,
        noise_std=2.0,
    ):
        super().__init__()

        default_load = np.array(
            [25, 28, 32, 38, 45, 50, 48, 42, 36, 30],
            dtype=np.float32,
        )

        self.actual = np.asarray(
            actual_workload if actual_workload is not None
            else default_load,
            dtype=np.float32,
        )

        self.pred = np.asarray(
            predicted_workload if predicted_workload is not None
            else self.actual,
            dtype=np.float32,
        )

        if len(self.actual) == 0 or len(self.pred) == 0:
            raise ValueError("Workload arrays cannot be empty.")

        if len(self.actual) != len(self.pred):
            raise ValueError(
                "Actual and predicted workload arrays must have "
                "the same length."
            )

        self.sla_ms = float(sla_ms)
        self.initial_warm = int(initial_warm)
        self.noise_std = float(noise_std)

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(9,),
            dtype=np.float32,
        )

    def state(self):
        cpu = min(100.0, 15.0 + self.load * 1.5)
        memory = min(100.0, 20.0 + self.warm * 7.0 + self.load * 0.8)

        overload = max(0.0, self.load - self.warm * 10.0)
        latency = 70.0 + overload * 9.0
        cold = min(1.0, overload / max(self.load, 1.0))
        energy = self.warm * 0.8 + cpu * 0.015
        risk = max(0.0, latency / self.sla_ms - 0.8)

        return np.array(
            [
                self.load,
                self.pred[min(self.t, len(self.pred) - 1)],
                cpu,
                memory,
                self.warm,
                cold,
                latency,
                energy,
                risk,
            ],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.t = 0
        self.load = max(0.0, float(self.actual[0]))
        self.warm = self.initial_warm

        return self.state(), {}

    def step(self, action):
        changes = {
            0: 0,
            1: 1,
            2: 2,
            3: -1,
            4: 0,
            5: -2,
        }

        action = int(action)
        if action not in changes:
            raise ValueError(f"Invalid action: {action}")

        previous_warm = self.warm
        self.warm = max(1, self.warm + changes[action])

        # Advance one workload timestep.
        self.t += 1

        index = min(self.t, len(self.actual) - 1)
        noise = self.np_random.normal(0.0, self.noise_std)
        self.load = max(0.0, float(self.actual[index]) + noise)

        obs = self.state()

        latency = float(obs[6])
        energy = float(obs[7])
        cold = float(obs[5] * 25.0)
        violation = max(0.0, latency - self.sla_ms)

        # Penalize excessive provisioning as well as underprovisioning.
        capacity = self.warm * 10.0
        underprovision = max(0.0, self.load - capacity)
        excess_capacity = max(0.0, capacity - self.load)

        reward = -(
            0.4 * latency
            + 1.2 * energy
            + 0.3 * cold
            + 0.25 * underprovision
            + 3.5 * violation
            + 0.03 * excess_capacity
        )

        terminated = self.t >= len(self.actual) - 1

        info = {
            "time_index": index,
            "actual_load": float(self.load),
            "predicted_load": float(
                self.pred[min(index, len(self.pred) - 1)]
            ),
            "latency_ms": latency,
            "energy": energy,
            "warm_containers": self.warm,
            "cold_start_proxy": cold,
            "sla_violation": violation,
            "cold_start_event": int(
                self.load > previous_warm * 10.0
            ),
        }

        return obs, float(reward), terminated, False, info
