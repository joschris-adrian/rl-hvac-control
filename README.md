# RL HVAC Control

A reinforcement learning framework for **HVAC energy optimization** in single‑zone and multi‑zone buildings.  
The project simulates indoor temperature dynamics and trains agents to balance:

- **Thermal comfort** (target ~22°C)
- **Energy consumption**
- **Outdoor temperature fluctuations**

The environments are built with **Gymnasium** and support both classical controllers and RL algorithms.

---

# Features

- Custom **Gymnasium HVAC environments** (simple, feature‑engineered, and multi‑zone)
- **Feature‑engineered state representations**
- **Multi‑zone thermal coupling** with physically meaningful inter-zone heat transfer
- Multiple control strategies:
  - Reinforcement Learning (PPO, DQN)
  - Classical Control (PID)
  - Rule‑based thermostat baseline
- Continuous and discrete action spaces
- Scalable from 2 to N zones
- Lightweight simulation for rapid experimentation

---

# Environments

## 1. SimpleHVACEnv

A minimal environment designed for quick testing and algorithm prototyping.

### State

```
[indoor_temperature, outdoor_temperature]
```

### Actions (Discrete)

| Action | Meaning |
|--------|---------|
| 0      | Idle    |
| 1      | Cool    |
| 2      | Heat    |

### Reward

Penalizes:

- Energy usage
- Deviation from the comfort temperature (22°C)

---

## 2. HVACEnvFeatureEngineered

A more detailed single‑zone environment with engineered features for improved learning stability.

### State Features

```
[
  indoor_temperature,
  outdoor_temperature,
  temperature_derivative,
  sin(hour),
  cos(hour),
  occupancy_weight,
  previous_action
]
```

### Action Space (Continuous)

```
[-1, 1]
```

Mapped to HVAC heating/cooling power.

### Thermal Model

Indoor temperature dynamics follow:

```
dT = (Tout - Tin) / (R * C) + Power / C
```

Where:

- `Tin` = indoor temperature
- `Tout` = outdoor temperature
- `R` = thermal resistance
- `C` = thermal capacitance

---

## 3. MultiZoneHVACEnv

Extended from single‑zone to **multi‑zone HVAC with thermal coupling between rooms**. Each zone has its own temperature state and control action, while exchanging heat with adjacent zones and the outdoor environment.

### State Features

```
[
  temps (n_zones),
  outdoor_temperature,
  dT (n_zones),
  sin(hour),
  cos(hour),
  occupancy_weight,
  previous_actions (n_zones)
]
```

Total observation dimension: `3 * n_zones + 4`

### Action Space (Continuous)

```
[-1, 1]^n_zones
```

One continuous action per zone, mapped to heating/cooling power.

### Thermal Model

Each zone `i` evolves as:

```
dT_i = (Tout - T_i) / (R * C)          # outdoor exchange
     + Power_i / C                       # HVAC input
     + Σ (T_j - T_i) / (R_adj * C)     # coupling with adjacent zones j ≠ i
```

Where:

- `R` = wall thermal resistance (zone ↔ outdoor)
- `R_adj` = coupling resistance (zone ↔ zone)
- `C` = thermal capacitance

### Parameters

| Parameter       | Default | Description                        |
|-----------------|---------|------------------------------------|
| `n_zones`       | 3       | Number of zones                    |
| `R`             | 2.0     | Outdoor thermal resistance         |
| `R_adj`         | 1.0     | Inter-zone coupling resistance     |
| `C`             | 3.0     | Thermal capacitance                |
| `max_power`     | 5.0     | Max HVAC power per zone (kW)       |
| `target_temp`   | 22.0    | Comfort setpoint (°C)              |
| `lambda_comfort`| 0.5     | Comfort penalty weighting factor   |

### Usage

```python
from rl_hvac_control.env.multi_zone_env import MultiZoneHVACEnv

env = MultiZoneHVACEnv(n_zones=3)
obs, info = env.reset()

action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

---

# Algorithms

## PPO (Proximal Policy Optimization)

- Used for continuous control in the feature‑engineered and multi‑zone environments
- Suitable for smooth HVAC power modulation across multiple zones

## DQN (Deep Q‑Network)

- Used with the discrete SimpleHVACEnv
- Good baseline for simple control tasks

## Rule‑Based Thermostat

```
if temperature < setpoint:
    heat
elif temperature > setpoint:
    cool
else:
    idle
```

## PID Controller

Classical control using proportional, integral, and derivative terms.

---

# Installation

```bash
git clone https://github.com/joschris-adrian/rl-hvac-control.git
cd rl-hvac-control
pip install -r requirements.txt
```

---

# Training

## PPO Agent

```bash
python src/agents/train_ppo.py
```

## DQN Agent

```bash
python src/agents/train_dqn.py
```

---

# Project Structure

```
rl-hvac-control
│
├── src
│   └── rl_hvac_control
│       ├── env
│       │   ├── hvac_env.py                   # SimpleHVACEnv + HVACEnvFeatureEngineered
│       │   └── multi_zone_env.py             # MultiZoneHVACEnv
│       │
│       ├── agents
│       │   ├── train_ppo.py
│       │   └── train_dqn.py
│       │
│       ├── controllers
│       │   ├── pid_controller.py
│       │   └── rule_based_controller.py
│       │
│       └── utils
│
├── tests
│   ├── test_hvac_integration.py
│   └── test_ml_zone_env.py
│
├── notebooks
├── experiments
├── requirements.txt
├── setup.py
└── README.md
```

---

# Reward Function

The reward balances **energy cost** and **thermal comfort**:

```
Reward = - energy_cost - λ * comfort_penalty
```

Where:

- `energy_cost` = HVAC power × electricity price
- `comfort_penalty` = mean |T_i − target_temperature| across zones
- `λ` = comfort weighting factor

---

# Future Improvements

- Training PPO on the multi‑zone environment with observation normalization
- Integration with real weather datasets (e.g. EnergyPlus, NOAA)
- Time‑of‑use electricity pricing
- Model Predictive Control (MPC) comparison
- Offline RL using real building datasets
- Occupancy prediction models
- Stochastic occupancy and disturbance modeling

---

# Research Applications

Useful for exploring:

- RL for building energy systems
- Multi‑agent and multi‑zone control
- Smart building energy optimization
- Energy‑efficient HVAC strategies
- Comparisons between RL and classical control

---

# License

MIT License
