import numpy as np


class PIDController:
    """
    Simple PID controller for HVAC temperature regulation.

    The controller attempts to maintain a target temperature
    using proportional, integral, and derivative control.
    """

    def __init__(self, target: float = 22.0, kp: float = 1.0, ki: float = 0.01, kd: float = 0.1):
        self.target = target
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.integral = 0.0
        self.prev_error = 0.0

        self.threshold = 0.5

    def reset(self) -> None:
        """Reset controller state."""
        self.integral = 0.0
        self.prev_error = 0.0

    def act(self, observation: np.ndarray) -> int:
        """Compute HVAC action based on PID control."""

        temp = observation[0]

        error = self.target - temp

        self.integral += error
        derivative = error - self.prev_error

        self.prev_error = error

        control = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * derivative
        )

        if control > self.threshold:
            return 2  # heat
        elif control < -self.threshold:
            return 1  # cool
        else:
            return 0  # idle