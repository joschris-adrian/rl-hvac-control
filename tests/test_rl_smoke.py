import pytest

def test_ppo_smoke_runs():
    sb3 = pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO  
    from rl_hvac_control.env.hvac_env import SimpleHVACEnv

    env = SimpleHVACEnv()
    model = PPO("MlpPolicy", env, verbose=0)
    # Very small learn call to ensure training loop runs
    model.learn(total_timesteps=100)
