import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend
import matplotlib.pyplot as plt
import os
import mujoco

# import sys
# if sys.version_info[0] == 3:
#     import tkinter as tk
# else:
#     import Tkinter as tk

# Define simulation parameters
STRING_LENGTH = 0.1  # meters
BALL_WEIGHT = 0.2  # kg
GRAVITY = 9.81  # m/s^2
SIMULATION_TIMESTEP = 1 / 50  # 50 Hz
bTheta = 0.1 # Damping Oscillation
bPhi = 0 # Damping Rotation

# Create environment for RL training.
class ManipulatorEnv(gym.Env):
    def __init__(self, trajectory, model_path):
        super(ManipulatorEnv, self).__init__()
        self.trajectory = trajectory
        self.current_step = 0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,),
                                       dtype=np.float32)  # Adjust to match the number of joints
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        print(self.data)

        # Initialize the end effector's position based on the initial joint angles
        self.initial_position = self.calculate_initial_position()
        self.ball_pos = self.initial_position + np.array([0, 0, -STRING_LENGTH])

    def calculate_initial_position(self):
        # Extract joint angles from the model (assumed initial positions)
        joint_angles = self.data.qpos[:2]  # Assuming the first two joints are relevant
        # Calculate the position of the end effector based on the joint angles
        base_height=0.2
        link1_length = 0.2  # Height of link 1
        link2_length = 0.2  # Height of link 2

        # Forward kinematics to determine end effector position
        x =0.0 #link1_length * np.cos(joint_angles[0]) + link2_length * np.cos(joint_angles[0] + joint_angles[1])
        y =0.0 #link1_length * np.sin(joint_angles[0]) + link2_length * np.sin(joint_angles[0] + joint_angles[1])
        z = base_height+link1_length+link2_length # Assuming base height

        return np.array([x, y, z])


    def step(self, action):
        # Apply joint velocity control
        self.data.ctrl[0] = action[0]  # Control for joint1
        self.data.ctrl[1] = action[1]  # Control for joint2
        mujoco.mj_step(self.model, self.data)

        # Get new end effector position
        end_effector_pos = self.data.site_xpos[0]
        # **Calculate the desired ball position based on the end effector position and string length**
        desired_ball_pos = end_effector_pos + np.array([0, 0, -STRING_LENGTH])

        # **Calculate the acceleration due to gravity and update the ball's position**
        acceleration_due_to_gravity = np.array([0, 0, -GRAVITY])
        # For the ball's new position, simulate its lagging response
        # Using simple physics update: position = previous_position + velocity * timestep + 0.5 * acceleration * timestep^2
        if self.current_step > 0:
            # Update position based on physics (considering string tension and ball's weight)
            tension = self.calculate_tension(desired_ball_pos, end_effector_pos)
            # Update ball's position based on the force acting on it (gravity and tension)
            self.ball_pos += (desired_ball_pos - self.ball_pos) * tension / (BALL_WEIGHT * GRAVITY)

        # Compute trajectory error
        target_pos = self.trajectory[self.current_step]
        #error = np.linalg.norm(np.array(self.ball_pos) - np.array(target_pos))
        error = np.linalg.norm(end_effector_pos- np.array(target_pos))
        reward = -error

        self.current_step += 1
        done = self.current_step >= len(self.trajectory)
        #obs = np.concatenate([self.ball_pos, end_effector_pos, target_pos])
        obs = np.concatenate([end_effector_pos, target_pos])

        return obs, reward, done, {}, {}

    def calculate_tension(self, desired_ball_pos, end_effector_pos):
        # Calculate the length of the string and tension
        distance_to_ball = np.linalg.norm(desired_ball_pos - end_effector_pos)
        if distance_to_ball > STRING_LENGTH:
            # If the distance exceeds the string length, it means tension is acting
            return (STRING_LENGTH / distance_to_ball)  # This is a simplification
        return 1.0  # Full tension if within length


    def reset(self, seed=None, options=None):
        self.current_step = 0
        mujoco.mj_resetData(self.model, self.data)

        # Set initial joint angles if needed
        self.data.qpos[:2] = np.array([0, 0])  # Reset to zero angles for example
        self.initial_position = self.calculate_initial_position()
        self.ball_pos = self.initial_position + np.array([0, 0, -STRING_LENGTH])  # Initialize ball position
        #return np.concatenate([self.ball_pos, [0,0,0.6], [0.0, 0.0, 0.5]]), {}
        return np.concatenate([[0, 0, 0.6], [0.0, 0.0, 0.6]]), {}


    def render(self, mode='human'):
        mujoco.mj_render(self.model, self.data)

# Function to visualize initial positions
def visualize_initial_positions(initial_position, string_length,trajectory):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Base position
    ax.scatter(initial_position[0], initial_position[1], initial_position[2], color='k', s=100, label='Base')

    # Link 1 position
    link1_end = np.array(initial_position) + np.array([0, 0, 0.2])  # Height of link 1
    ax.plot([initial_position[0], link1_end[0]],
            [initial_position[1], link1_end[1]],
            [initial_position[2], link1_end[2]], color='b', linewidth=5, label='Link 1')

    # Link 2 position
    link2_end = link1_end + np.array([0, 0, 0.2])  # Height of link 2
    ax.plot([link1_end[0], link2_end[0]],
            [link1_end[1], link2_end[1]],
            [link1_end[2], link2_end[2]], color='g', linewidth=5, label='Link 2')

    # End effector position
    end_effector_pos = link2_end + np.array([0, 0, 0.0])
    ax.scatter(end_effector_pos[0], end_effector_pos[1], end_effector_pos[2], color='r', s=100, label='End Effector')

    # Ball position
    ball_pos = end_effector_pos + np.array([0, 0, -string_length])  # Position of the ball
    ax.scatter(ball_pos[0], ball_pos[1], ball_pos[2], color='orange', s=100, label='Ball')

    # Draw string
    ax.plot([end_effector_pos[0], ball_pos[0]],
            [end_effector_pos[1], ball_pos[1]],
            [end_effector_pos[2], ball_pos[2]], color='purple', linestyle='--', label='String')

    # Desired trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'r--', label='Desired Trajectory')


    # Set labels and legend
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('Initial Pose of Manipulator, Ball, String, and Desired Trajectory')
    ax.legend()
    plt.show()


# # Generate a sample 3D trajectory
# time_steps = np.linspace(0, 5, 250)
# trajectory = np.vstack((0.05 * np.sin(time_steps), 0.05 * np.cos(time_steps), 0.5 * np.ones_like(time_steps))).T

# Generate time steps
num_steps = 250
time_steps = np.linspace(0, 1, num_steps)

start_point = np.array([0.0, 0.0, 0.5])  # Initial position of the ball
end_point = np.array([0.1, 0.0, 0.3])  # Final position

# Interpolate between start and end points
trajectory = np.outer(time_steps, (end_point - start_point)) + start_point


# Call the visualization function before training
#visualize_initial_positions(initial_position=[0, 0, 0.2], string_length=STRING_LENGTH,trajectory=trajectory)

# Train the RL policy
#model_path = r"C:\Users\hthh1\PycharmProjects\pythonProject\manipulator.xml"
model_path = r"/Users/haihangw/PycharmProjects/Robotic-Reinforcement-Learning/manipulator.xml"
env = ManipulatorEnv(trajectory, model_path)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Test the trained policy
obs, _ = env.reset()
positions = []
for _ in range(len(trajectory)):
    #print('current is:', obs[:3],'target is:',obs[6:9], 'end effector is:',obs[3:6])
    print('end effector  is:', obs[:3], 'target is:', obs[3:6])
    action, _ = model.predict(obs, deterministic=True) #obs = np.concatenate([self.ball_pos, end_effector_pos, target_pos])
    obs, _, done, _, _ = env.step(action)
    positions.append(obs[:3])  # Track ball positions
    if done:
        break

# Visualization of the trajectoryko
positions = np.array(positions)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'r--', label='Desired Trajectory')
ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', label='Tracked Trajectory')
ax.legend()
plt.show()


# def f(t, state, i):  # i?
#
#     alpha = state[0]  # Oscillation velocity
#     theta = state[1]  # Vertical angle
#     beta = 0  # Rotational velocity (ignored for now)
#     phi = state[3]  # Azimuth angle
#
#     gamma_dot = xPivot(i)[1]  # end effector position?
#     delta_dot = zPivot(i)[1]
#
#     alpha_dot = 1 / (BALL_WEIGHT * pow(STRING_LENGTH, 2)) * (  # l is the length of the string ?
#             -GRAVITY * STRING_LENGTH * BALL_WEIGHT * np.sin(theta) + 0.5 * pow(STRING_LENGTH, 2) * BALL_WEIGHT * np.sin(
#         2 * theta) * pow(beta,
#                          2) - bTheta * alpha - STRING_LENGTH * BALL_WEIGHT * np.cos(
#         phi) * np.cos(theta) * gamma_dot + STRING_LENGTH * BALL_WEIGHT * np.cos(theta) * np.sin(phi) * delta_dot)
#     theta_dot = alpha
#
#     beta_dot = 1 / (BALL_WEIGHT * pow(STRING_LENGTH, 2) * pow(np.sin(theta), 2)) * (
#             -bPhi * beta - 2 * pow(STRING_LENGTH, 2) * BALL_WEIGHT * np.cos(theta) * np.sin(
#         theta) * beta * alpha + STRING_LENGTH * BALL_WEIGHT * np.sin(
#         phi) * np.sin(theta) * gamma_dot + STRING_LENGTH * BALL_WEIGHT * np.cos(phi) * np.sin(theta) * delta_dot)
#     phi_dot = beta
#
#     # return the first order derivatives
#     return np.array([alpha_dot, theta_dot, beta_dot, phi_dot])
