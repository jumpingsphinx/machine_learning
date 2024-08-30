import torch
import gymnasium as gym
from discrete_DQN import DQN

# Load the environment and get model parameters
env = gym.make("CarRacing-v2", continuous=False, render_mode="human")
n_actions = env.action_space.n
input_shape = env.observation_space.shape
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model architecture
model = DQN(input_shape, n_actions).to(device)
print(model.layer1.weight)
# Load the model weights
model.load_state_dict(torch.load('carracing_dqn.pth'))

# Set the model to evaluation mode
model.eval()

# Print model's layer weights to verify loading

