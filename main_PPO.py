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
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:2] = 0
        self.initial_position = self.calculate_initial_position()
        self.ball_pos = self.initial_position + np.array([0, 0, -STRING_LENGTH])
        return np.concatenate([[0.0, 0.0, 0.6], [0.0, 0.0, 0.6]]), {}


class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)

def compute_gae(rewards, values, masks, gamma=0.99, lam=0.95):
    values = values + [0]
    gae, returns = 0, []
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
        gae = delta + gamma * lam * masks[i] * gae
        returns.insert(0, gae + values[i])
    return returns

def ppo_update(policy, optimizer, observations, actions, log_probs_old, returns, advantages, clip_epsilon=0.2, epochs=10, batch_size=64):
    dataset = torch.utils.data.TensorDataset(observations, actions, log_probs_old, returns, advantages)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        for obs_batch, act_batch, logp_old_batch, ret_batch, adv_batch in loader:
            pi, value = policy(obs_batch)
            dist = torch.distributions.Normal(pi, 0.1)
            logp = dist.log_prob(act_batch).sum(axis=-1)
            ratio = torch.exp(logp - logp_old_batch)

            surr1 = ratio * adv_batch
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * adv_batch
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (ret_batch - value.squeeze()).pow(2).mean()

            optimizer.zero_grad()
            (policy_loss + 0.5 * value_loss).backward()
            optimizer.step()

def train_ppo(env, policy, optimizer, episodes=1000, gamma=0.99, lam=0.95):
    for episode in range(episodes):
        obs, _ = env.reset()
        log_probs, values, rewards, masks, actions, states = [], [], [], [], [], []

        done = False
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            states.append(obs_tensor)
            action_mean, value = policy(obs_tensor)
            dist = torch.distributions.Normal(action_mean, 0.1)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()

            obs, reward, done, _, _ = env.step(action.detach().numpy())
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value.item())
            rewards.append(reward)
            masks.append(1 - float(done))

        returns = compute_gae(rewards, values, masks, gamma, lam)
        advantages = torch.tensor(returns, dtype=torch.float32) - torch.tensor(values, dtype=torch.float32)

        observations = torch.stack(states)
        actions = torch.stack(actions)
        log_probs_old = torch.stack(log_probs)
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ppo_update(policy, optimizer, observations, actions, log_probs_old, returns, advantages)

        print(f"Episode {episode}, total reward: {sum(rewards):.2f}")


# Trajectory definition
num_steps = 250
time_steps = np.linspace(0, 1, num_steps)
start = np.array([0.0, 0.0, 0.6])
end = np.array([0.1, 0.0, 0.3])
trajectory = np.outer(time_steps, (end - start)) + start

# Setup
model_path = r"D:\research projects\Robotic-Reinforcement-Learning\manipulator.xml"
env = ManipulatorEnv(trajectory, model_path)
policy_net = ActorCritic(input_dim=6, output_dim=2)
# for name, param in policy_net.named_parameters():
#     if param.requires_grad:
#         print(name)
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)

# Train
train_ppo(env, policy_net, optimizer, episodes=100)

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
