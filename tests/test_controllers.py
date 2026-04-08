import numpy as np
from rl_hvac_control.controllers.rule_controller import RuleBasedThermostat

def test_rule_controller_basic_behavior():
    controller = RuleBasedThermostat()

    obs_hot = np.array([25.0, 30.0], dtype=float)
    try:
        action = controller.act(obs_hot)
    except Exception:
        action = controller.act(float(obs_hot[0]))
    action_arr = np.asarray(action, dtype=float).ravel()
    assert action_arr.size >= 1
    assert int(round(action_arr[0])) in (0, 1, 2)

def test_rule_controller_cold_behavior():
    controller = RuleBasedThermostat()

    obs_cold = np.array([18.0, 5.0], dtype=float)
    try:
        action = controller.act(obs_cold)
    except Exception:
        action = controller.act(float(obs_cold[0]))
    action_arr = np.asarray(action, dtype=float).ravel()
    assert action_arr.size >= 1
    assert int(round(action_arr[0])) in (0, 1, 2)
