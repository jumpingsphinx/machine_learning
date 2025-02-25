# CarRacing & LunarLander DQN Training

This repository contains Deep Q-Network (DQN) implementations to train AI agents for playing **CarRacing-v2** and **LunarLander-v2** from OpenAI Gym. The models are trained using reinforcement learning with experience replay to improve stability and efficiency.

## Table of Contents
- [Project Overview](#project-overview)
- [Dependencies & Installation](#dependencies--installation)
- [Training Process](#training-process)
- [Model Architecture](#model-architecture)
- [Hyperparameters](#hyperparameters)
- [Saving & Loading Model](#saving--loading-model)

## Project Overview
This project implements reinforcement learning agents for playing **CarRacing-v2** and **LunarLander-v2** using a DQN approach. The models:
- Use deep neural networks for Q-learning.
- Implement an **epsilon-greedy** policy to balance exploration and exploitation.
- Use a **target network** to stabilize learning.
- Implement **experience replay** to break temporal correlations and improve training efficiency.

## Training Process
The models are trained using a reinforcement learning approach:
1. The agent interacts with the **CarRacing-v2** or **LunarLander-v2** environment, choosing actions based on an **epsilon-greedy** policy.
2. The agent stores state transitions in a **ReplayMemory** buffer.
3. The models are optimized using Q-learning, with the target Q-values updated at a slower rate.
4. Training happens for **600 episodes** on GPU (if available) or **50 episodes** on CPU for CarRacing.
5. Training happens for **800 episodes** on GPU (if available) or **50 episodes** on CPU for LunarLander.
6. The agent's performance is visualized in real-time.

## Model Architecture
The DQN models consist of:
- **CarRacing Model:**
  - `Input Layer`: Flattens the 3D state input into a 1D vector.
  - `Hidden Layer 1`: 128 neurons (ReLU activation)
  - `Hidden Layer 2`: 128 neurons (ReLU activation)
  - `Output Layer`: `n_actions` neurons predicting Q-values for each possible action.

- **LunarLander Model:**
  - `Input Layer`: Takes in the raw environment observations.
  - `Hidden Layer 1`: 128 neurons (ReLU activation)
  - `Hidden Layer 2`: 128 neurons (ReLU activation)
  - `Output Layer`: `n_actions` neurons predicting Q-values for each possible action.

## Hyperparameters
| Parameter | Value |
|-----------|-------|
| `BATCH_SIZE` | 128 |
| `GAMMA` | 0.99 |
| `EPS_START` | 0.9 (CarRacing), 0.93 (LunarLander) |
| `EPS_END` | 0.05 (CarRacing), 0.03 (LunarLander) |
| `EPS_DECAY` | 1000 |
| `TAU` | 0.005 |
| `LR` | 1e-4 |
| `Memory Capacity` | 10,000 |



## Saving & Loading Model
- The trained models are saved as:
  ```
  models/carracing.pth
  models/lunarlander.pth
  ```
- To load a pre-trained CarRacing model:
  ```python
  policy_net.load_state_dict(torch.load('models/carracing.pth'))
  ```
- To load a pre-trained LunarLander model:
  ```python
  policy_net.load_state_dict(torch.load('models/lunarlander.pth'))
  ```


