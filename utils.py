import torch
import copy
import numpy as np
from torch import nn
from random import choices



activationDict = {"ReLU": nn.ReLU(), "Tanh": nn.Tanh(), "sigmoid": nn.Sigmoid()}

def build_NN(input_size, output_size, hidden_layers, activation="ReLU", endActivation=nn.Identity()):
    """Helper function to build variable size neural networks"""
    layers = []
    for i, layer in enumerate(hidden_layers):
        if i == 0:
            layers.append(nn.Linear(input_size, layer))
            layers.append(activationDict[activation])
        else:
            layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            layers.append(activationDict[activation])
    layers.append(nn.Linear(hidden_layers[i], output_size))
    layers.append(endActivation)
    return nn.Sequential(*layers)
            
class Path(object):
    """Object to store one transition in enviroment MDP (state, action, next_state, reward, terminal)"""
    def __init__(self) -> None:
        self.obs = []
        self.action = []
        self.next_obs = []
        self.reward = []
        self.done = []
        self.value = []
        self.action_logprob = []
    
    def add(self, obs, action, next_obs, reward, done, action_logprob, value):
        """Adds state, action, next_state, reward, terminal to path object"""
        self.obs.append(obs)
        self.action.append(action.numpy())
        self.next_obs.append(next_obs)
        self.reward.append(np.array(reward))
        self.done.append(np.array([done]))
        self.action_logprob.append(np.array([action_logprob]))
        self.value.append(np.array([value]))


class ReplayBuffer(object):
    """The experience buffer that holds path objects (single transitions) and has helper functions
        to sample from the all transitions in the experience buffer"""
    def __init__(self, max_buffer = 100000):
        self.max_buffer = max_buffer
        self.paths = []
        self.actions = None
        self.obs = None
        self.next_obs = None
        self.rewards = None
        self.terminals = None
        self.count = 0

    def add(self, path):
        """"Adds a experience to the experience buffer via taking in a single transition (path)"""
        self.count += 1
        self.paths.append(path)
        observations = self.obs
        actions = self.actions
        next_observation = self.next_obs
        rewards = self.rewards
        terminals = self.terminals
        breakpoint()
        if self.obs is None:
            observations = torch.tensor(path.obs, dtype=torch.float32)
            actions = torch.tensor(path.action, dtype=torch.float32)
            next_observation = torch.tensor(path.next_obs, dtype=torch.float32)
            rewards = torch.tensor(path.reward, dtype=torch.float32)
            terminals = torch.tensor(path.done, dtype=torch.float32)
        else:
            observations = np.concatenate([observations, torch.tensor(path.obs, dtype=torch.float32)], axis=0)
            actions = np.concatenate([actions, torch.tensor(path.action, dtype=torch.float32)], axis=0)
            next_observation = np.concatenate([next_observation, torch.tensor(path.next_obs, dtype=torch.float32)], axis=0)
            rewards = np.concatenate([rewards, torch.tensor(path.reward, dtype=torch.float32)], axis=0)
            terminals = np.concatenate([terminals, torch.tensor(path.done, dtype=torch.float32)], axis=0)
        if self.obs is None:
            self.obs = observations[-self.max_buffer:]
            self.actions = actions[-self.max_buffer:]
            self.next_obs = next_observation[-self.max_buffer:]
            self.rewards = rewards[-self.max_buffer:]
            self.terminals = terminals[-self.max_buffer:]
        else:
            self.obs = observations[-self.max_buffer:]
            self.actions = actions[-self.max_buffer:]
            self.next_obs = next_observation[-self.max_buffer:]
            self.rewards = rewards[-self.max_buffer:]
            self.terminals = terminals[-self.max_buffer:]

    def sample(self, batch_size=64):
        """Samples random transitions from the experience buffer based on the buffer size"""
        observations = choices(self.obs, k=batch_size)
        actions = choices(self.actions, k=batch_size)
        next_observation = choices(self.next_obs, k=batch_size)
        rewards = choices(self.rewards, k=batch_size)
        terminals = choices(self.terminals, k=batch_size)
        return observations, actions, next_observation, rewards, terminals

    def sample_recent(self, transitions=1):
        return self.paths[-1:]
    
    def __len__(self):
        if self.obs is None:
            return 0
        else:
            return len(self.obs)

#note: adpated from OpenAI noise helper function
# https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class OrnsteinUhlenbeckActionNoise(object):
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None, final_scale=0.02, scale_timestep=40000):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.scale = 1.0
        self.t = 0
        self.final_scale = 0.02
        self.scale_timestep = scale_timestep
        self.reset()

    def noise(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        self.t += 1
        self.scaleFn()
        x = x * self.scale
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def scaleFn(self):
        self.scale = max(self.final_scale, (self.scale_timestep - self.t)/self.scale_timestep)

def Qvalue(rewards, critic):
    
    pass

def discountReward(rewards):
    pass

def Advantage(reward, q_vals):
    pass