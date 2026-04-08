import numpy as np
from rl_hvac_control.controllers.rule_controller import RuleBasedThermostat
from rl_hvac_control.evaluation import evaluate_with_logs, evaluate_baseline

def test_evaluate_with_logs_shapes_and_types():
    from rl_hvac_control.env.hvac_env import SimpleHVACEnv
    env = SimpleHVACEnv()
    controller = RuleBasedThermostat()
    temps, energies, actions = evaluate_with_logs(env, controller_or_model=controller, is_rl=False, steps=3)
    assert len(temps) == len(energies) == len(actions)
    assert all(isinstance(t, float) for t in temps)

def test_energy_unit_conversion():
    from rl_hvac_control.env.hvac_env import SimpleHVACEnv
    env = SimpleHVACEnv()
    env.dt = 3600.0
    env.max_power = 1.0

    controller = RuleBasedThermostat()
    temps, energies, actions = evaluate_with_logs(env, controller_or_model=controller, is_rl=False, steps=1)
    assert len(energies) == 1
    assert energies[0] >= 0.0

def test_evaluate_baseline_returns_metrics():
    from rl_hvac_control.env.hvac_env import SimpleHVACEnv
    env = SimpleHVACEnv()
    controller = RuleBasedThermostat()
    metrics = evaluate_baseline(env, controller, steps=5)
    assert "total_energy_kWh" in metrics
    assert "avg_reward" in metrics
    assert metrics["total_energy_kWh"] >= 0.0
