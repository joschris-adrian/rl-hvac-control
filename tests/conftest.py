import pytest
from rl_hvac_control.env.hvac_env import SimpleHVACEnv

@pytest.fixture
def simple_env():
    """
    Deterministic SimpleHVACEnv instance for unit tests.
    If the environment supports seeding, this fixture will attempt to seed it.
    """
    env = SimpleHVACEnv()
    if hasattr(env, "seed"):
        try:
            env.seed(0)
        except Exception:
            pass
    return env
