import numpy as np

class DummyHVACEnv:
    """
    Minimal deterministic environment used by unit tests.
    - dt default is seconds (3600 -> 1 hour)
    - step returns (obs, reward, done, info) in older gym style
    """
    def __init__(self, dt=3600.0, max_power=5.0, horizon=24):
        self.dt = dt
        self.max_power = max_power
        self.horizon = horizon
        self._step = 0
        self._tin = 22.0
        self.action_space = None
        self.observation_space = None

    def reset(self):
        self._step = 0
        self._tin = 22.0
        obs = np.array([self._tin, 10.0], dtype=float)
        return obs

    def step(self, action):
        a = float(np.asarray(action).ravel()[0])
        tout = 10.0
        # simple discrete-time thermal update
        self._tin = self._tin + (-0.1 * (self._tin - tout)) + 0.2 * a
        self._step += 1
        reward = -abs(self._tin - 22.0) - 0.01 * abs(a) * self.max_power
        done = self._step >= self.horizon
        info = {}
        return np.array([self._tin, tout], dtype=float), float(reward), bool(done), info

    def seed(self, s):
        return [s]
