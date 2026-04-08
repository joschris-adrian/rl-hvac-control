import numpy as np


class RuleBasedThermostat:
    """
    Simple rule-based thermostat controller.

    Maintains indoor temperature near a target setpoint using a
    deadband to prevent frequent heating/cooling switching.
    """

    COOL = 1
    HEAT = 2
    IDLE = 0

    def __init__(self, target: float = 22.0, deadband: float = 1.0):
        self.target = target
        self.deadband = deadband

    def act(self, observation) -> int:
        if isinstance(observation, (tuple, list)):
            observation = observation[0]
        indoor_temp = float(np.asarray(observation).flat[0])
    
        if indoor_temp > self.target + self.deadband:
            return self.COOL

        elif indoor_temp < self.target - self.deadband:
            return self.HEAT

        else:
            return self.IDLE