import os
from stable_baselines3 import PPO
from src.env.hvac_env import SimpleHVACEnv

TIMESTEPS = 200_000


def train():
    """Train PPO agent on the HVAC environment."""

    os.makedirs("results/models", exist_ok=True)

    env = SimpleHVACEnv()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=42,
        tensorboard_log="results/tensorboard/"
    )

    model.learn(total_timesteps=TIMESTEPS)

    model.save("results/models/ppo_hvac")


if __name__ == "__main__":
    train()