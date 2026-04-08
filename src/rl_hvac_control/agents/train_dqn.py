import os
from stable_baselines3 import DQN
from src.env.hvac_env import SimpleHVACEnv

TIMESTEPS = 200_000


def train():
    """
    Train a DQN agent on the SimpleHVAC environment.
    """

    # Create results directory if it does not exist
    os.makedirs("results/models", exist_ok=True)

    env = SimpleHVACEnv()

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        seed=42,
        tensorboard_log="results/tensorboard/"
    )

    model.learn(total_timesteps=TIMESTEPS)

    model.save("results/models/dqn_hvac")


if __name__ == "__main__":
    train()