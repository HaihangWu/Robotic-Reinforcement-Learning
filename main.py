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
SIMULATION_TIMESTEP = 0.002
NUM_TIME_STEPS = 200
bTheta = 0.1 # Oscillation (Damping)
bPhi = 0 # Rotation (Damping)
t_eval  = np.arange(0,NUM_TIME_STEPS*SIMULATION_TIMESTEP,SIMULATION_TIMESTEP)

####################pendulum dynamics######################################
class Pendulum_System():
    def __init__(self,trajectory):
        self.trajectory = trajectory
        self.vxPivot0 = 0 # Initial velocity of pivot in x
        self.axPivot0 = 0 # Initial acceleration of pivot in x
        self.vzPivot0 = 0 # Initial velocity of pivot in z
        self.azPivot0 = 0 # Initial acceleration of pivot in z
        self.xend_effector = 0.0
        self.yend_effector = 0.6
        self.zend_effector = 0.0

        self.xball = 0.0
        self.yball = 0.5
        self.zball = 0.0

        self.xPos = np.zeros(NUM_TIME_STEPS)
        self.zPos = np.zeros(NUM_TIME_STEPS)
        self.yPos = np.zeros(NUM_TIME_STEPS)
        self.vx = np.zeros(NUM_TIME_STEPS)
        self.ax = np.zeros(NUM_TIME_STEPS)
        self.vz = np.zeros(NUM_TIME_STEPS)
        self.az = np.zeros(NUM_TIME_STEPS)

        self.alpha = 0.0
        self.beta = 0.0

    def xPivot(self,counter):
        if counter == 0:
            self.vx[counter] = self.vxPivot0
            self.ax[counter] = self.axPivot0
            return self.vx[counter], self.ax[counter]

        else:
            self.vx[counter] = (self.xPos[counter] - self.xPos[counter - 1]) / SIMULATION_TIMESTEP  # backward difference

            self.ax[counter] = (self.vx[counter] - self.vx[counter - 1]) / SIMULATION_TIMESTEP  # backward difference
            return self.vx[counter], self.ax[counter]

    # Defining acceleration of pivot in z direction
    def zPivot(self,counter):
        if counter == 0:
            self.vz[counter] = self.vzPivot0
            self.az[counter] = self.azPivot0
            return self.vz[counter], self.az[counter]

        else:
            self.vz[counter] = (self.zPos[counter] - self.zPos[counter - 1]) / SIMULATION_TIMESTEP  # backward difference

            self.az[counter] = (self.vz[counter] - self.vz[counter - 1]) / SIMULATION_TIMESTEP  # backward difference

            return self.vz[counter], self.az[counter]

    # Defining dynamics of the rope
    def f(self, t, state, i):
        self.alpha = state[0]  # Oscillation velocity
        theta = state[1]  # Vertical angle
        self.beta = 0  # Rotational velocity (ignored for now)
        phi = state[3]  # Azimuth angle

        gamma_dot = self.xPivot(i)[1]
        delta_dot = self.zPivot(i)[1]

        alpha_dot = 1 / (BALL_WEIGHT * pow(STRING_LENGTH, 2)) * (
                    -GRAVITY * STRING_LENGTH * BALL_WEIGHT * np.sin(theta) + 0.5 * pow(STRING_LENGTH,
                                                                                       2) * BALL_WEIGHT * np.sin(
                2 * theta) * pow(self.beta,
                                 2) - bTheta * self.alpha - STRING_LENGTH * BALL_WEIGHT * np.cos(
                phi) * np.cos(theta) * gamma_dot + STRING_LENGTH * BALL_WEIGHT * np.cos(theta) * np.sin(
                phi) * delta_dot)
        theta_dot = self.alpha

        beta_dot = 1 / (BALL_WEIGHT * pow(STRING_LENGTH, 2) * pow(np.sin(theta), 2)) * (
                -bPhi * self.beta - 2 * pow(STRING_LENGTH, 2) * BALL_WEIGHT * np.cos(theta) * np.sin(
            theta) * self.beta * self.alpha + STRING_LENGTH * BALL_WEIGHT * np.sin(
            phi) * np.sin(theta) * gamma_dot + STRING_LENGTH * BALL_WEIGHT * np.cos(phi) * np.sin(theta) * delta_dot)
        phi_dot = self.beta

        # return the first order derivatives
        return np.array([alpha_dot, theta_dot, beta_dot, phi_dot])

    # Defining fixed timestep RK45 numerical integrator
    def rk45(self,t, state, h, counter):
        k1 = self.f(t, state, counter)
        k2 = self.f(t + h / 4, state + h / 4 * k1, counter)
        k3 = self.f(t + 3 / 8 * h, state + h * (3 / 32 * k1 + 9 / 32 * k2), counter)
        k4 = self.f(t + 12 / 13 * h, state + h * (1932 / 2197 * k1 - 7200 / 2197 * k2 + 7296 / 2197 * k3), counter)
        k5 = self.f(t + h, state + h * (439 / 216 * k1 - 8 * k2 + 3680 / 513 * k3 - 845 / 4104 * k4), counter)
        k6 = self.f(t + h / 2, state + h * (-8 / 27 * k1 + 2 * k2 - 3544 / 2565 * k3 + 1859 / 4104 * k4 - 11 / 40 * k5),
               counter)

        state4 = state + h * (25 / 216 * k1 + 0 * k2 + 1408 / 2565 * k3 + 2197 / 4104 * k4 - 1 / 5 * k5 + 0 * k6)

        self.alpha = state4[0]
        theta = state4[1]
        self.beta = state4[2]
        phi = state4[3]

        newState = [self.alpha, theta, self.beta, phi]

        # return 4th order approximation
        return newState

    def step(self,i, obs):
        self.xPos[i] = obs[0] if i > 0 else self.xend_effector
        self.zPos[i] = obs[1] if i > 0 else self.yend_effector
        self.yPos[i] = obs[2] if i > 0 else self.zend_effector
        # Distance between pivot current position and ball's current position
        xDelta = self.xball - self.xPos[i]
        yDelta = self.yball - self.yPos[i]
        zDelta = self.zball - self.zPos[i]

        # Finding angle between pivot and ball
        theta = np.atan2(np.sqrt(pow(xDelta, 2) + pow(zDelta, 2)), np.abs(yDelta))  # vertical angle
        phi = np.atan2(zDelta, xDelta)  # azimuth angle
        #print(theta, phi)

        if theta == 0:
            theta = 1e-6  # Prevent computational error for beta
        if ((xDelta < 0 or zDelta < 0)):  # Ensure that the ball oscillates
            phi += np.pi  # Change the azimuth angle to opposite angle for oscillation
            theta = theta * -1  # Change the vertical angle to opposite angle for oscillation
        phi = phi % (2 * np.pi)  # Ensure it's within -2pi to 2pi
        if phi < 0:
            phi += 2 * np.pi  # Ensure it's within 0 to 2pi

        # Ball's current state
        state = np.array([self.alpha, theta, self.beta, phi])
        # Finding ball's next state
        if i > 0:
            state = self.rk45(t_eval[i], state, SIMULATION_TIMESTEP, i)

        thetaNext = state[1]
        phiNext = state[3]
        thetaNext = thetaNext % (2 * np.pi)
        if thetaNext > np.pi:
            thetaNext = 2 * np.pi - thetaNext
        if (state[1] < 0):
            phiNext += np.pi
        phiNext = phiNext % (2 * np.pi)
        if (phiNext < 0):
            phiNext += 2 * np.pi

        #reward
        target = self.trajectory[i]
        error = np.linalg.norm(np.array([self.xball,self.zball, self.yball]) - target)
        reward = -error

        # Finding ball's next position
        self.xball = obs[0] + STRING_LENGTH * np.sin(thetaNext) * np.cos(phiNext)
        self.yball = obs[2] - STRING_LENGTH * np.cos(thetaNext)
        self.zball = obs[1] + STRING_LENGTH * np.sin(thetaNext) * np.sin(phiNext)
        self.alpha = state[0]  # Ball's next theta_dot
        self.beta = state[2]  # Ball's next phi_dot

        # Storing ball's next state
        # stateStore[:,i+1] = [alpha,state[1],beta,phiNext]
        # Storing ball's next position
        return [self.xball, self.zball, self.yball], reward

    def reset(self):
        self.xPos = np.zeros(NUM_TIME_STEPS)
        self.zPos = np.zeros(NUM_TIME_STEPS)
        self.yPos = np.zeros(NUM_TIME_STEPS)
        self.vx = np.zeros(NUM_TIME_STEPS)
        self.ax = np.zeros(NUM_TIME_STEPS)
        self.vz = np.zeros(NUM_TIME_STEPS)
        self.az = np.zeros(NUM_TIME_STEPS)

        self.xball = 0.0
        self.yball = 0.5
        self.zball = 0.0

        self.xend_effector = 0.0
        self.yend_effector = 0.6
        self.zend_effector = 0.0

        self.alpha = 0.0
        self.beta = 0.0

# Environment Definition
class ManipulatorEnv(gym.Env):
    def __init__(self, end_effector_trajectory, model_path):
        super().__init__()
        self.trajectory = end_effector_trajectory
        self.current_step = 0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = SIMULATION_TIMESTEP
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

        obs = np.concatenate([end_effector, target])
        self.current_step += 1
        done = self.current_step >= len(self.trajectory)
        return obs, done, {}, {}

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
def train(env, pendulum, policy, episodes=1000, gamma=0.99):
    # Decrease exploration noise gradually
    policy_attempt = PolicyNetwork(input_dim=6, output_dim=2)
    optimizer_attempt = optim.SGD(policy_attempt.parameters(), lr=1e-6)

    best_Total_Reward=-1000000
    network_initialization_attempts=1000 # if negative, there are no network re-initialization; if positive, the network is re-initialized network_initialization_attempts times
    for episode in range(episodes):
        #exploration_std *= exploration_decay  # Reduce std dev over time
        obs, _ = env.reset()
        pendulum.reset()

        if episode<network_initialization_attempts:
           reset_policy_weights(policy_attempt)
        if episode==network_initialization_attempts: # select the best model initialization
            policy_attempt = copy.deepcopy(policy)

        log_probs, rewards = [], []
        done = False
        iter_step=0
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            action_probs = policy_attempt(obs_tensor)
            dist = torch.distributions.Normal(action_probs, torch.ones_like(action_probs) * 0.5)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()
            obs, done, _, _ = env.step(action.detach().numpy())
            ball_position, reward = pendulum.step(iter_step, obs)
            log_probs.append(log_prob)
            rewards.append(reward)
            iter_step=iter_step+1

        Current_total_reward=sum(rewards)
        if Current_total_reward>best_Total_Reward: # avoid using the bad trajectory to  update the model
            print(f"Episode {episode}, total reward: {sum(rewards):.2f}")
            best_Total_Reward=Current_total_reward
            if episode > network_initialization_attempts:  # avoid bad model initialization
                final_returns = [sum(gamma**t * r for t, r in enumerate(rewards[i:])) for i in range(len(rewards))]
                final_returns = torch.tensor(final_returns, dtype=torch.float32)
                final_log_probs=log_probs

                optimizer_attempt.zero_grad()
                loss = -torch.stack(final_log_probs) @ final_returns
                loss.backward()
                optimizer_attempt.step()

            policy = copy.deepcopy(policy_attempt)  # Save best model

    return policy

# start = np.array([0.0, 0.0, 0.6])
# end = np.array([0.1, 0.0, 0.3])
end_effector_trajectory = np.array([[0. , 0. , 0.6], [-2.63794685e-05,  0.00000000e+00,  5.99999993e-01], [-7.91165821e-05,  0.00000000e+00,  5.99999947e-01], [-1.58188955e-04,  0.00000000e+00,  5.99999806e-01], [-2.63573592e-04,  0.00000000e+00,  5.99999487e-01], [-3.95246822e-04,  0.00000000e+00,  5.99998882e-01], [-5.53184203e-04,  0.00000000e+00,  5.99997858e-01], [-0.00073736,  0.        ,  0.59999626], [-0.00094775,  0.        ,  0.5999939 ], [-0.00118432,  0.        ,  0.59999057], [-0.00144705,  0.        ,  0.59998605], [-0.00173591,  0.        ,  0.59998007], [-0.00205086,  0.        ,  0.59997236], [-0.00239186,  0.        ,  0.59996262], [-0.00275889,  0.        ,  0.5999505 ], [-0.00315191,  0.        ,  0.59993567], [-0.00357086,  0.        ,  0.59991775], [-0.00401571,  0.        ,  0.59989634], [-0.00448641,  0.        ,  0.59987102], [-0.0049829 ,  0.        ,  0.59984136], [-0.00550513,  0.        ,  0.59980687], [-0.00605303,  0.        ,  0.59976708], [-0.00662654,  0.        ,  0.59972147], [-0.00722559,  0.        ,  0.59966953], [-0.00785009,  0.        ,  0.59961068], [-0.00849996,  0.        ,  0.59954437], [-0.00917511,  0.        ,  0.59946999], [-0.00987545,  0.        ,  0.59938694], [-0.01060085,  0.        ,  0.59929457], [-0.01135121,  0.        ,  0.59919224], [-0.0121264 ,  0.        ,  0.59907928], [-0.01292628,  0.        ,  0.59895498], [-0.01375072,  0.        ,  0.59881865], [-0.01459954,  0.        ,  0.59866956], [-0.0154726 ,  0.        ,  0.59850695], [-0.0163697 ,  0.        ,  0.59833008], [-0.01729065,  0.        ,  0.59813815], [-0.01823526,  0.        ,  0.59793038], [-0.0192033 ,  0.        ,  0.59770596], [-0.02019454,  0.        ,  0.59746407], [-0.02120872,  0.        ,  0.59720386], [-0.0222456,  0.       ,  0.5969245], [-0.02330488,  0.        ,  0.5966251 ], [-0.02438626,  0.        ,  0.59630481], [-0.02548945,  0.        ,  0.59596272], [-0.02661409,  0.        ,  0.59559796], [-0.02775985,  0.        ,  0.5952096 ], [-0.02892634,  0.        ,  0.59479674], [-0.03011318,  0.        ,  0.59435845], [-0.03131996,  0.        ,  0.59389381], [-0.03254625,  0.        ,  0.59340187], [-0.03379158,  0.        ,  0.59288171], [-0.03505549,  0.        ,  0.59233236], [-0.03633747,  0.        ,  0.5917529 ], [-0.03763701,  0.        ,  0.59114235], [-0.03895354,  0.        ,  0.59049979], [-0.04028651,  0.        ,  0.58982425], [-0.04163532,  0.        ,  0.58911479], [-0.04299933,  0.        ,  0.58837046], [-0.04437792,  0.        ,  0.58759032], [-0.04577039,  0.        ,  0.58677344], [-0.04717606,  0.        ,  0.58591887], [-0.04859419,  0.        ,  0.5850257 ], [-0.05002403,  0.        ,  0.58409301], [-0.05146479,  0.        ,  0.5831199 ], [-0.05291567,  0.        ,  0.58210547], [-0.05437582,  0.        ,  0.58104885], [-0.05584438,  0.        ,  0.57994916], [-0.05732045,  0.        ,  0.57880556], [-0.0588031,  0.       ,  0.5776172], [-0.06029139,  0.        ,  0.57638328], [-0.06178432,  0.        ,  0.57510299], [-0.06328089,  0.        ,  0.57377557], [-0.06478006,  0.        ,  0.57240025], [-0.06628074,  0.        ,  0.57097631], [-0.06778185,  0.        ,  0.56950303], [-0.06928226,  0.        ,  0.56797976], [-0.0707808 ,  0.        ,  0.56640583], [-0.0722763 ,  0.        ,  0.56478062], [-0.07376753,  0.        ,  0.56310354], [-0.07525325,  0.        ,  0.56137405], [-0.0767322,  0.       ,  0.5595916], [-0.07820308,  0.        ,  0.55775572], [-0.07966456,  0.        ,  0.55586595], [-0.08111529,  0.        ,  0.55392188], [-0.08255389,  0.        ,  0.55192313], [-0.08397896,  0.        ,  0.54986936], [-0.08538906,  0.        ,  0.54776029], [-0.08678276,  0.        ,  0.54559565], [-0.08815857,  0.        ,  0.54337525], [-0.08951498,  0.        ,  0.54109892], [-0.09085049,  0.        ,  0.53876655], [-0.09216354,  0.        ,  0.53637807], [-0.09345257,  0.        ,  0.53393346], [-0.09471599,  0.        ,  0.53143276], [-0.0959522 ,  0.        ,  0.52887604], [-0.09715958,  0.        ,  0.52626344], [-0.09833649,  0.        ,  0.52359516], [-0.09948127,  0.        ,  0.52087143], [-0.10059226,  0.        ,  0.51809256], [-0.10166778,  0.        ,  0.5152589 ], [-0.10270611,  0.        ,  0.51237086], [-0.10370557,  0.        ,  0.50942891], [-0.10466443,  0.        ,  0.50643359], [-0.10558096,  0.        ,  0.50338548], [-0.10645344,  0.        ,  0.50028525], [-0.10728013,  0.        ,  0.49713359], [-0.10805929,  0.        ,  0.49393129], [-0.10878916,  0.        ,  0.49067918], [-0.10946801,  0.        ,  0.48737817], [-0.1100941 ,  0.        ,  0.48402923], [-0.11066567,  0.        ,  0.48063339], [-0.11118101,  0.        ,  0.47719174], [-0.11163837,  0.        ,  0.47370546], [-0.11203604,  0.        ,  0.47017577], [-0.11237231,  0.        ,  0.46660398], [-0.11264548,  0.        ,  0.46299145], [-0.11285387,  0.        ,  0.45933962], [-0.11299582,  0.        ,  0.45564998], [-0.11306968,  0.        ,  0.45192411], [-0.11307383,  0.        ,  0.44816365], [-0.11300667,  0.        ,  0.44437031], [-0.11286663,  0.        ,  0.44054586], [-0.11265218,  0.        ,  0.43669215], [-0.1123618 ,  0.        ,  0.43281109], [-0.11199403,  0.        ,  0.42890466], [-0.11154743,  0.        ,  0.42497492], [-0.1110206 ,  0.        ,  0.42102397], [-0.1104122 ,  0.        ,  0.41705401], [-0.10972092,  0.        ,  0.41306727], [-0.10894551,  0.        ,  0.40906608], [-0.10808478,  0.        ,  0.4050528 ], [-0.10713757,  0.        ,  0.40102989], [-0.1061028 ,  0.        ,  0.39699984], [-0.10497945,  0.        ,  0.39296523], [-0.10376657,  0.        ,  0.38892867], [-0.10246326,  0.        ,  0.38489284], [-0.1010687 ,  0.        ,  0.38086049], [-0.09958216,  0.        ,  0.37683442], [-0.09800296,  0.        ,  0.37281746], [-0.09633053,  0.        ,  0.36881252], [-0.09456437,  0.        ,  0.36482254], [-0.09270406,  0.        ,  0.36085053], [-0.09074929,  0.        ,  0.35689951], [-0.08869982,  0.        ,  0.35297257], [-0.08655552,  0.        ,  0.34907283], [-0.08431637,  0.        ,  0.34520345], [-0.08198243,  0.        ,  0.34136761], [-0.07955387,  0.        ,  0.33756854], [-0.07703098,  0.        ,  0.33380948], [-0.07441415,  0.        ,  0.33009371], [-0.07170389,  0.        ,  0.32642452], [-0.0689008 ,  0.        ,  0.32280522], [-0.06600564,  0.        ,  0.31923913], [-0.06301924,  0.        ,  0.31572959], [-0.05994259,  0.        ,  0.31227994], [-0.05677679,  0.        ,  0.30889352], [-0.05352306,  0.        ,  0.30557367], [-0.05018274,  0.        ,  0.30232373], [-0.04675732,  0.        ,  0.29914701], [-0.04324839,  0.        ,  0.29604684], [-0.03965768,  0.        ,  0.29302651], [-0.03598708,  0.        ,  0.29008928], [-0.03223855,  0.        ,  0.2872384 ], [-0.02841424,  0.        ,  0.28447708], [-0.0245164 ,  0.        ,  0.28180849], [-0.02054743,  0.        ,  0.27923577], [-0.01650983,  0.        ,  0.27676201], [-0.01240627,  0.        ,  0.27439025], [-0.00823953,  0.        ,  0.27212347], [-0.00401253,  0.        ,  0.2699646 ], [0.00027169, 0.        , 0.26791651], [0.00460995, 0.        , 0.26598198], [0.00899894, 0.        , 0.26416373], [0.01343525, 0.        , 0.26246441], [0.0179153 , 0.        , 0.26088658], [0.02243543, 0.        , 0.25943271], [0.02699184, 0.        , 0.25810517], [0.0315806 , 0.        , 0.25690624], [0.03619767, 0.        , 0.25583811], [0.04083892, 0.        , 0.25490284], [0.04550007, 0.        , 0.2541024 ], [0.05017676, 0.        , 0.25343861], [0.05486451, 0.        , 0.25291321], [0.05955875, 0.        , 0.25252777], [0.0642548 , 0.        , 0.25228376], [0.06894791, 0.        , 0.2521825 ], [0.07363321, 0.        , 0.25222516], [0.07830578, 0.        , 0.25241278], [0.08296058, 0.        , 0.25274623], [0.08759255, 0.        , 0.25322624], [0.09219651, 0.        , 0.25385336], [0.09676726, 0.        , 0.25462798], [0.10129952, 0.        , 0.25555033], [0.10578798, 0.        , 0.25662044], [0.11022728, 0.        , 0.25783818], [0.11461202, 0.        , 0.25920322], [0.11893679, 0.        , 0.26071507], [0.12319615, 0.        , 0.262373  ], [0.12738466, 0.        , 0.26417612]])
ball_trajectory=[]
for end_effector_position in end_effector_trajectory:
    ball_position=end_effector_position-np.array([0., 0., STRING_LENGTH])
    ball_trajectory.append(ball_position)
ball_trajectory = np.array(ball_trajectory)
model_path = r"D:\research projects\Robotic-Reinforcement-Learning\manipulator.xml"
env = ManipulatorEnv(end_effector_trajectory, model_path)
pendulum=Pendulum_System(ball_trajectory)
policy_net = PolicyNetwork(input_dim=6, output_dim=2)

# Train RL polcy
policy_net=train(env, pendulum, policy_net, episodes=2000)

# Evaluate the learned policy
obs, _ = env.reset()
pendulum.reset()
end_effect_positions = []
ball_positions=[]
#stateStore = []


for i in range(len(ball_trajectory)):
    with torch.no_grad():
        action = policy_net(torch.tensor(obs, dtype=torch.float32)).numpy()
    obs, done, _, _ = env.step(action)
    end_effect_positions.append(obs[:3])
    ball_position, _ =pendulum.step(i, obs)
    ball_positions.append(ball_position)
    ########################################################################
    if done:
        break

# Plot
end_effect_positions = np.array(end_effect_positions)
ball_positions = np.array(ball_positions)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(ball_trajectory[:, 0], ball_trajectory[:, 1], ball_trajectory[:, 2], 'r--', label='Target')
ax.plot(ball_positions[:, 0], ball_positions[:, 1], ball_positions[:, 2], 'b-', label='Tracked')
ax.legend()
plt.show()


