import numpy as np
from typing import Any, Dict, List, Tuple, Union

ArrayLike = Union[np.ndarray, List[float], Tuple[float, ...]]

# Small helpers for robustness

def _unpack_reset(reset_out: Any) -> np.ndarray:
    """Return observation array from env.reset() output (supports obs or (obs, info))."""
    if isinstance(reset_out, (tuple, list)):
        return np.asarray(reset_out[0])
    return np.asarray(reset_out)


def _unpack_step(step_out: Any) -> Tuple[np.ndarray, float, bool, dict]:
    """
    Normalize env.step() output to (obs, reward, done, info).
    Handles older Gym (obs, reward, done, info) and newer (obs, reward, terminated, truncated, info).
    """
    if isinstance(step_out, (tuple, list)):
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
            return np.asarray(obs), float(np.asarray(reward).flat[0]), done, dict(info or {})
        if len(step_out) == 4:
            obs, reward, done, info = step_out
            return np.asarray(obs), float(np.asarray(reward).flat[0]), bool(done), dict(info or {})
        # Fallback: best-effort unpack
        obs = step_out[0]
        reward = step_out[1] if len(step_out) > 1 else 0.0
        done = bool(step_out[2]) if len(step_out) > 2 else False
        info = step_out[-1] if len(step_out) > 3 else {}
        return np.asarray(obs), float(np.asarray(reward).flat[0]), done, dict(info or {})
    # Unexpected single return
    return np.asarray(step_out), 0.0, False, {}


def _ensure_obs_array(obs: Any) -> np.ndarray:
    """Return a 1-D numpy array for the observation."""
    arr = np.asarray(obs)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr


def _dt_hours_from_env(env: Any) -> float:
    """
    Return time step length in hours.
    If env.dt_hours exists use it; otherwise infer from env.dt (seconds -> hours).
    If neither exists, default to 1.0 hour per step.
    """
    if hasattr(env, "dt_hours"):
        return float(env.dt_hours)
    if hasattr(env, "dt"):
        dt = float(env.dt)
        return dt / 3600.0 if dt > 1.0 else dt
    return 1.0


# Evaluation functions

def evaluate_baseline(env: Any, controller: Any, steps: int = 24) -> Dict[str, float]:
    """
    Run one episode using a rule-based controller and return summary metrics.

    Expected controller interface:
      - controller.act(obs_array) -> action_array
      - If controller.act expects a scalar temperature, a fallback will pass obs[0].

    Returned metrics:
      - total_energy_kWh: float
      - temp_variance: float
      - comfort_violations: float (sum of degrees outside [21, 23])
      - avg_reward: float
    """
    reset_out = env.reset()
    obs = _unpack_reset(reset_out)
    obs = _ensure_obs_array(obs)

    temp_log: List[float] = []
    energy_log: List[float] = []
    reward_log: List[float] = []

    dt_hours = _dt_hours_from_env(env)
    max_power = float(getattr(env, "max_power", 1.0))

    for _ in range(steps):
        # Try to call controller with full observation; fallback to scalar
        try:
            action = controller.act(obs)
        except Exception:
            action = controller.act(float(obs[0]))

        action = np.atleast_1d(np.asarray(action, dtype=float))
        step_out = env.step(action)
        obs, reward, done, info = _unpack_step(step_out)
        obs = _ensure_obs_array(obs)

        # Log the state after the action so the temperature reflects the action's effect
        temp_log.append(float(obs[0]))
        reward_log.append(float(reward))

        power = abs(float(action[0]) * max_power)
        energy_log.append(power * dt_hours)

        if done:
            break

    total_energy = float(np.sum(energy_log))
    temp_variance = float(np.var(temp_log)) if temp_log else float("nan")
    comfort_violations = float(
        np.sum([max(0.0, 21.0 - t) + max(0.0, t - 23.0) for t in temp_log])
    )
    avg_reward = float(np.mean(reward_log)) if reward_log else float("nan")

    return {
        "total_energy_kWh": total_energy,
        "temp_variance": temp_variance,
        "comfort_violations": comfort_violations,
        "avg_reward": avg_reward,
    }


def evaluate_rl(env: Any, model: Any, steps: int = 24) -> Dict[str, float]:
    """
    Run one episode using an RL model and return summary metrics.

    Expected model interface:
      - model.predict(obs_array, deterministic=True) -> (action_array, state)

    Observations are logged after env.step so they reflect the action taken.
    """
    reset_out = env.reset()
    obs = _unpack_reset(reset_out)
    obs = _ensure_obs_array(obs)

    temp_log: List[float] = []
    energy_log: List[float] = []
    reward_log: List[float] = []

    dt_hours = _dt_hours_from_env(env)
    max_power = float(getattr(env, "max_power", 1.0))

    for _ in range(steps):
        try:
            action, _state = model.predict(obs, deterministic=True)
        except Exception as e:
            raise RuntimeError(f"model.predict failed: {e}")

        action = np.atleast_1d(np.asarray(action, dtype=float))
        step_out = env.step(action)
        obs, reward, done, info = _unpack_step(step_out)
        obs = _ensure_obs_array(obs)

        temp_log.append(float(obs[0]))
        reward_log.append(float(reward))

        power = abs(float(action[0]) * max_power)
        energy_log.append(power * dt_hours)

        if done:
            break

    total_energy = float(np.sum(energy_log))
    temp_variance = float(np.var(temp_log)) if temp_log else float("nan")
    comfort_violations = float(
        np.sum([max(0.0, 21.0 - t) + max(0.0, t - 23.0) for t in temp_log])
    )
    avg_reward = float(np.mean(reward_log)) if reward_log else float("nan")

    return {
        "total_energy_kWh": total_energy,
        "temp_variance": temp_variance,
        "comfort_violations": comfort_violations,
        "avg_reward": avg_reward,
    }


def evaluate_with_logs(env: Any, controller_or_model: Any, is_rl: bool = False, steps: int = 24) -> Tuple[List[float], List[float], List[float]]:
    """
    Run one episode and return raw logs: (temp_log, energy_log, action_log).

    - If is_rl is True, controller_or_model must implement .predict(obs, deterministic=True).
    - If is_rl is False, controller_or_model must implement .act(obs) or .act(scalar_temp).
    """
    reset_out = env.reset()
    obs = _unpack_reset(reset_out)
    obs = _ensure_obs_array(obs)

    temp_log: List[float] = []
    energy_log: List[float] = []
    action_log: List[float] = []

    dt_hours = _dt_hours_from_env(env)
    max_power = float(getattr(env, "max_power", 1.0))

    for _ in range(steps):
        if is_rl:
            try:
                action, _state = controller_or_model.predict(obs, deterministic=True)
            except Exception as e:
                raise RuntimeError(f"RL model.predict failed: {e}")
        else:
            try:
                action = controller_or_model.act(obs)
            except Exception:
                action = controller_or_model.act(float(obs[0]))

        action = np.atleast_1d(np.asarray(action, dtype=float))
        step_out = env.step(action)
        obs, reward, done, info = _unpack_step(step_out)
        obs = _ensure_obs_array(obs)

        temp_log.append(float(obs[0]))
        action_log.append(float(action[0]))

        power = abs(float(action[0]) * max_power)
        energy_log.append(power * dt_hours)

        if done:
            break

    return temp_log, energy_log, action_log
