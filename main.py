import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend
import matplotlib.pyplot as plt
import os
import mujoco
import torch
import torch.nn as nn
import torch.optim as optim
import copy

# Constants
STRING_LENGTH = 0.1
BALL_WEIGHT = 0.2
GRAVITY = 9.81
SIMULATION_TIMESTEP = 1 / 50

# Environment Definition
class ManipulatorEnv(gym.Env):
    def __init__(self, trajectory, model_path):
        super().__init__()
        self.trajectory = trajectory
        self.current_step = 0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.initial_position = self.calculate_initial_position()
        self.ball_pos = self.initial_position + np.array([0, 0, -STRING_LENGTH])

    def calculate_initial_position(self):
        base_height = 0.2
        link1_length = 0.2
        link2_length = 0.2
        return np.array([0.0, 0.0, base_height + link1_length + link2_length])

    def calculate_tension(self, desired, end_effector):
        distance = np.linalg.norm(desired - end_effector)
        return STRING_LENGTH / distance if distance > STRING_LENGTH else 1.0

    def step(self, action):
        self.data.ctrl[:2] = action
        mujoco.mj_step(self.model, self.data)

        end_effector = self.data.site_xpos[0]
        desired_ball = end_effector + np.array([0, 0, -STRING_LENGTH])
        if self.current_step > 0:
            tension = self.calculate_tension(desired_ball, end_effector)
            self.ball_pos += (desired_ball - self.ball_pos) * tension / (BALL_WEIGHT * GRAVITY)

        target = self.trajectory[self.current_step]
        error = np.linalg.norm(end_effector - target)
        reward = -error

        obs = np.concatenate([end_effector, target])
        self.current_step += 1
        done = self.current_step >= len(self.trajectory)
        return obs, reward, done, {}, {}

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.model.opt.gravity = np.zeros(3)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:2] = 0
        self.initial_position = self.calculate_initial_position()
        self.ball_pos = self.initial_position + np.array([0, 0, -STRING_LENGTH])
        return np.concatenate([[0.0, 0.0, 0.6], [0.0, 0.0, 0.6]]), {}


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.fc(x)

def reset_policy_weights(policy):
    def weight_reset(m):
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
            m.reset_parameters()
    policy.apply(weight_reset)

# REINFORCE Algorithm
def train(env, policy, episodes=1000, gamma=0.99):
    # Decrease exploration noise gradually
    exploration_decay = 0.995
    exploration_std = 0.5
    policy_attempt = PolicyNetwork(input_dim=6, output_dim=2)
    optimizer_attempt = optim.Adam(policy_attempt.parameters(), lr=1e-3)

    best_Total_Reward=-1000000
    final_log_probs, final_returns = [], []
    sampling_attempts=5
    for episode in range(episodes):
        #exploration_std *= exploration_decay  # Reduce std dev over time
        obs, _ = env.reset()
        if best_Total_Reward<-15: # avoid bad model initialization
           reset_policy_weights(policy_attempt)
        log_probs, rewards = [], []
        done = False
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            action_probs = policy_attempt(obs_tensor)
            dist = torch.distributions.Normal(action_probs, torch.ones_like(action_probs) * 0.5)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()
            obs, reward, done, _, _ = env.step(action.detach().numpy())
            log_probs.append(log_prob)
            rewards.append(reward)

        Current_total_reward=sum(rewards)
        if Current_total_reward>best_Total_Reward: # avoid to use the bad trajectory to  update the model
            print(f"Episode {episode}, total reward: {sum(rewards):.2f}")
            best_Total_Reward=Current_total_reward
            final_returns = [sum(gamma**t * r for t, r in enumerate(rewards[i:])) for i in range(len(rewards))]
            final_returns = torch.tensor(final_returns, dtype=torch.float32)
            final_log_probs=log_probs

            optimizer_attempt.zero_grad()
            loss = -torch.stack(final_log_probs) @ final_returns
            loss.backward()
            optimizer_attempt.step()

            policy = copy.deepcopy(policy_attempt)  # Save best model



    return policy



# Trajectory definition
num_steps = 250
time_steps = np.linspace(0, 1, num_steps)
start = np.array([0.0, 0.0, 0.6])
end = np.array([0.1, 0.0, 0.3])
trajectory = np.outer(time_steps, (end - start)) + start

# Setup
model_path = r"D:\research projects\Robotic-Reinforcement-Learning\manipulator.xml"
env = ManipulatorEnv(trajectory, model_path)
policy_net = PolicyNetwork(input_dim=6, output_dim=2)
# for name, param in policy_net.named_parameters():
#     if param.requires_grad:
#         print(name)
#optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)

# Train
policy_net=train(env, policy_net, episodes=50)

# Evaluate
obs, _ = env.reset()
positions = []


for _ in range(len(trajectory)):
    with torch.no_grad():
        action = policy_net(torch.tensor(obs, dtype=torch.float32)).numpy()
    obs, _, done, _, _ = env.step(action)
    positions.append(obs[:3])
    if done:
        break

# Plot
positions = np.array(positions)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'r--', label='Target')
ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', label='Tracked')
ax.legend()
plt.show()



# def train(env, policy, optimizer, episodes=1000, gamma=0.99):
#     # Decrease exploration noise gradually
#     exploration_decay = 0.995
#     exploration_std = 0.5
#     Old_Total_Reward=-1000000
#     sampling_attempts=5
#     for episode in range(episodes):
#         exploration_std *= exploration_decay  # Reduce std dev over time
#         obs, _ = env.reset()
#         log_probs, rewards = [], []
#         done = False
#         while not done:
#             obs_tensor = torch.tensor(obs, dtype=torch.float32)
#             action_probs = policy(obs_tensor)
#             dist = torch.distributions.Normal(action_probs, torch.ones_like(action_probs) * 0.1)
#             action = dist.sample()
#             log_prob = dist.log_prob(action).sum()
#             obs, reward, done, _, _ = env.step(action.detach().numpy())
#             log_probs.append(log_prob)
#             rewards.append(reward)
#
#         Current_total_reward=sum(rewards)
#         optimizer.zero_grad()
#         if Current_total_reward>Old_Total_Reward:
#             print(f"Episode {episode}, total reward: {sum(rewards):.2f}")
#             Old_Total_Reward=Current_total_reward
#             returns = [sum(gamma**t * r for t, r in enumerate(rewards[i:])) for i in range(len(rewards))]
#             returns = torch.tensor(returns, dtype=torch.float32)
#             #returns = (returns - returns.mean()) / (returns.std() + 1e-8)
#             #returns = torch.tensor(returns, dtype=torch.float32)
#             loss = -torch.stack(log_probs) @ returns
#             loss.backward()
#             optimizer.step()

# manipulator test
# for step in range(20):
# returns = (returns - returns.mean()) / (returns.std() + 1e-8)
# returns = torch.tensor(returns, dtype=torch.float32)