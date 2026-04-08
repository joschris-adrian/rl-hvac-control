# HVAC Reinforcement Learning Control

A reinforcement learning project that trains a PPO agent to control an HVAC (heating, ventilation, and air conditioning) system, benchmarked against rule-based and PID controllers.

---

## Problem Statement

HVAC systems are responsible for roughly 40% of building energy consumption. Traditional controllers (thermostats, PID loops) maintain comfort but are not optimized for energy efficiency. This project frames HVAC control as a sequential decision-making problem and trains a deep RL agent to learn a control policy directly from environment interactions.

**Goal**: Keep indoor temperature in the comfort zone [21°C–23°C] while minimizing energy use.

---

## Environment

`SimpleHVACEnv` is a custom [Gymnasium](https://gymnasium.farama.org/) environment:

| Property | Value |
|---|---|
| Observation space | `Box(0, 50, shape=(2,))` — [indoor temp, outdoor temp] |
| Action space | `Discrete(3)` — cool, off, heat |
| Reward | Negative comfort penalty + negative energy cost |
| Episode length | Configurable (default 100 steps) |

The reward function penalizes deviation from the 21–23°C comfort band and the energy cost of each action, forcing the agent to learn efficient control.

---

## Controllers

Three controllers are implemented and compared:

| Controller | Description |
|---|---|
| **Rule-based** | Simple thermostat: heat below setpoint, cool above setpoint |
| **PID** | Proportional-integral-derivative controller tracking the setpoint |
| **PPO** | Proximal Policy Optimization agent (Stable-Baselines3), trained for 20,000 timesteps |

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `env_debug.ipynb` | Explore the environment — observation space, random policy behavior |
| `controller_comparison.ipynb` | Compare rule-based vs. PID controllers |
| `train_ppo.ipynb` | Train the PPO agent and visualize the learning curve |
| `results_analysis.ipynb` | Full three-way comparison: Rule vs. PID vs. PPO with metrics |

Run notebooks in the order listed above. `results_analysis.ipynb` requires a trained model saved by `train_ppo.ipynb`.

---

## Results

The PPO agent learns to hug the comfort band tightly while avoiding unnecessary heating/cooling cycles. Key metrics over 100-step evaluation episodes:

| Controller | Avg Reward | Comfort Violations (°C) | Energy (kWh) |
|---|---|---|---|
| Rule-based | ~−194 | High | Moderate |
| PID | ~−131 | Moderate | Lower |
| PPO | Lower loss | Lowest | Lowest |

*(See `results_analysis.ipynb` for the full chart and per-run numbers.)*

---

## Setup

```bash
# Create environment
conda create -n hvac python=3.11
conda activate hvac

# Install dependencies
pip install gymnasium stable-baselines3 matplotlib numpy
pip install -e .
```

---

## Project Structure

```
rl-hvac-control/
├── src/
│   └── rl_hvac_control/
│       ├── env/
│       │   └── hvac_env.py          # Custom Gymnasium environment
│       ├── controllers/
│       │   ├── rule_controller.py   # Rule-based thermostat
│       │   └── pid_controller.py    # PID controller
│       └── evaluation/
│           └── evaluate.py          # Shared evaluation utilities
├── notebooks/
│   ├── env_debug.ipynb
│   ├── controller_comparison.ipynb
│   ├── train_ppo.ipynb
│   └── results_analysis.ipynb
├── results/
│   └── ppo_hvac.zip                 # Saved trained model
└── tests/
    ├── test_controllers.py
    ├── test_evaluation.py
    └── test_hvac_integration.py
```

---

## Key Design Choices

**Why PPO?** PPO is sample-efficient, stable with discrete action spaces, and well-supported in Stable-Baselines3. For a low-dimensional HVAC environment, it converges quickly without needing complex architectures.

**Why a custom env?** Real HVAC simulators (EnergyPlus, etc.) are slow and hard to install. `SimpleHVACEnv` is lightweight and self-contained, making the RL loop fast enough to train and experiment interactively in a notebook.

**Why Discrete(3) actions?** Real HVAC systems often operate in on/off or staged modes. Discrete actions are interpretable and map directly to physical actuator states.
