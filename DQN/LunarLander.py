import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Initialize the environment
env = gym.make("LunarLander-v2", enable_wind = False, render_mode="human")

# Enable interactive mode for plotting
plt.ion()

# Select the device to run the model
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# Named tuple to store transitions in the replay memory
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Determine the number of episodes based on device availability
num_episodes = 800 if torch.cuda.is_available() or torch.backends.mps.is_available() else 50


# Replay Memory class to store transitions
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Deep Q-Network (DQN) model definition
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_observations, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# Hyperparameters
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.93
EPS_END = 0.03
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get the number of actions and observations from the environment
n_actions = env.action_space.n
n_observations = env.observation_space.shape[0]

# Instantiate the policy and target networks
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# Optimizer and replay memory
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

# List to track episode durations
episode_durations = []


# Function to plot the duration of episodes
def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)


# Function to optimize the model
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state).float()
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward).float()

    # Compute Q-values for the current state-action pairs
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute Q-values for the next state
    next_state_values = torch.zeros(BATCH_SIZE, device=device).float()
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute loss
    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


# Function to preprocess the state
def preprocess_state(state):
    return torch.from_numpy(state).float().unsqueeze(0).to(device)


# Function to select an action based on epsilon-greedy policy
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


# Main training loop
for i_episode in range(num_episodes):
    state, info = env.reset()
    state = preprocess_state(state)

    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        next_state = None if done else preprocess_state(observation)

        memory.push(state, action, next_state, reward)

        state = next_state

        optimize_model()

        # Update target network with a small amount of the policy network
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            print(f"Episode {len(episode_durations)} complete")
            break

# Plot the results
print('Complete')
torch.save(policy_net.state_dict(), 'models/lunarlander.pth')
plot_durations(show_result=True)
plt.ioff()
plt.show()
