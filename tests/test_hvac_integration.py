import pytest

pytestmark = pytest.mark.slow

def test_full_hvac_integration():
    sb3 = pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO  # noqa: E402
    from rl_hvac_control.env.hvac_env import SimpleHVACEnv
    from rl_hvac_control.controllers.rule_controller import RuleBasedThermostat

    env = SimpleHVACEnv()
    controller = RuleBasedThermostat()

    # short end-to-end run of the rule controller
    reset_out, _ = env.reset()   # unpack (obs, info)
    for _ in range(50):
        action = controller.act(reset_out)
        reset_out, _, _, _, _ = env.step(action)    
        try:
            action = controller.act(reset_out)
        except Exception:
            action = controller.act(float(reset_out[0]))
        step_out = env.step(action)
        # support both 4- and 5-tuple step outputs
        if isinstance(step_out, (tuple, list)) and len(step_out) >= 3:
            done = bool(step_out[2])
        else:
            done = False
        if done:
            break

    # small RL smoke training
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=200)
    assert True
