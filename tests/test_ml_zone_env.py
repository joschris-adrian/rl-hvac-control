from rl_hvac_control.env.multi_zone_env import MultiZoneHVACEnv

def test_multi_zone_reset_and_step():
    env = MultiZoneHVACEnv(n_zones=3)
    obs, info = env.reset()
    assert obs.shape == (3 * 3 + 4,)  # 3*n_zones + 4

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (13,)
    assert isinstance(reward, float)
    assert "energy_cost" in info
    assert "comfort_penalty" in info

def test_scalability():
    for n in [2, 5, 10]:
        env = MultiZoneHVACEnv(n_zones=n)
        obs, _ = env.reset()
        assert obs.shape == (3 * n + 4,)