# RL HVAC Control

A reinforcement learning framework for **HVAC energy optimization** in a single‑zone building.  
The project simulates indoor temperature dynamics and trains agents to balance:

- **Thermal comfort** (target ~22°C)
- **Energy consumption**
- **Outdoor temperature fluctuations**

The environments are built with **Gymnasium** and support both classical controllers and RL algorithms.

---

# Features

- Custom **Gymnasium HVAC environments**
- **Feature‑engineered state representations**
- Multiple control strategies:
  - Reinforcement Learning (PPO, DQN)
  - Classical Control (PID)
  - Rule‑based thermostat baseline
- Continuous and discrete action spaces
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

| Action | Meaning      |
|--------|--------------|
| 0      | Idle         |
| 1      | Cool         |
| 2      | Heat         |

### Reward

Penalizes:

- Energy usage  
- Deviation from the comfort temperature (22°C)

---

## 2. HVACEnvFeatureEngineered

A more detailed environment with engineered features for improved learning stability.

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

- Tin = indoor temperature  
- Tout = outdoor temperature  
- R = thermal resistance  
- C = thermal capacitance  

---

# Algorithms

## PPO (Proximal Policy Optimization)

- Used for continuous control in the feature‑engineered environment  
- Suitable for smooth HVAC power modulation

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
git clone https://github.com/yourusername/rl-hvac-control.git
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
│   ├── agents
│   │   ├── train_ppo.py
│   │   └── train_dqn.py
│   │
│   ├── environments
│   │   ├── simple_env.py
│   │   └── hvac_env_feature_engineered.py
│   │
│   ├── controllers
│   │   ├── pid_controller.py
│   │   └── rule_based_controller.py
│   │
│   └── utils
│
├── requirements.txt
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
- `comfort_penalty` = |Tin − target_temperature|  
- `λ` = comfort weighting factor  

---

# Future Improvements

- Multi‑zone HVAC control  
- Integration with real weather datasets  
- Time‑of‑use electricity pricing  
- Model Predictive Control (MPC) comparison  
- Offline RL using building datasets  
- Occupancy prediction models  

---

# Research Applications

Useful for exploring:

- RL for building energy systems  
- Smart building control  
- Energy‑efficient HVAC strategies  
- Comparisons between RL and classical control  

---

# License

MIT License
