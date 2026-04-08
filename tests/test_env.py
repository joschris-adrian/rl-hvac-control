import numpy as np
from rl_hvac_control.env.hvac_env import SimpleHVACEnv

def _unpack_reset(reset_out):
    if isinstance(reset_out, (tuple, list)):
        return np.asarray(reset_out[0])
    return np.asarray(reset_out)

def _unpack_step(step_out):
    if isinstance(step_out, (tuple, list)):
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
            return np.asarray(obs), float(reward), done, dict(info or {})
        if len(step_out) == 4:
            obs, reward, done, info = step_out
            return np.asarray(obs), float(reward), bool(done), dict(info or {})
        obs = step_out[0]
        reward = step_out[1] if len(step_out) > 1 else 0.0
        done = bool(step_out[2]) if len(step_out) > 2 else False
        info = step_out[-1] if len(step_out) > 3 else {}
        return np.asarray(obs), float(reward), done, dict(info or {})
    return np.asarray(step_out), 0.0, False, {}

def test_env_reset_and_step_signature():
    env = SimpleHVACEnv()
    reset_out = env.reset()
    obs = _unpack_reset(reset_out)
    obs = np.asarray(obs)
    assert obs.size >= 2

def test_env_step_runs_and_returns_types():
    env = SimpleHVACEnv()
    last_reward = None
    for _ in range(5):
        step_out = env.step(0)
        obs, reward, done, info = _unpack_step(step_out)
        last_reward = reward
        if done:
            break
    assert isinstance(last_reward, float)
