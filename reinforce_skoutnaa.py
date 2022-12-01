import torch
import torch.nn as nn
import torch.utils.data as data
from torch import distributions
import torch.nn.utils as utils
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import LinearLR
from utils import build_NN
from torch import optim
import numpy as np
import itertools


class REINFORCEAGENT(nn.Module):
    """REINFOCE AGENT that builds a neural network"""
    def __init__(self, env, input_dim, output_dim, layers, size, learning_rate = 10e-3, activation = "ReLU") -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = layers
        self.size = size
        self.network = build_NN(self.input_dim, self.output_dim, self.layers, self.size, activation).float()
        self.std = nn.Parameter(torch.zeros(self.output_dim, dtype=torch.float32))
        self.loss = nn.MSELoss()
        self.lr = learning_rate
        self.optim = optim.Adam([self.std] +list(self.network.parameters()) , self.lr)
        self.env = env
        self.steps = 0

    def forward(self, obs):
        """Agent action function approximator
        """
        if type(obs) == torch.Tensor: 
            x = self.network(obs.float())
        else:
            x = self.network(torch.from_numpy(obs).float())
        sigma = torch.exp(self.std) #std of noise
        action_distribution = distributions.MultivariateNormal(x, torch.diag(sigma)) #noise addition to action to impose exploration from agent
        return action_distribution #a MultivariateNormal sample distribution for agent actions


    def update(self, obs, actions, rewards):
        """Update function for REINFORCE Agent
            updates the agent network weights by calulating the log probability of policy action and sampled action
            from their the action probability is increased of decreased based on the reward"""
        obs = torch.from_numpy(np.array(obs))
        actions = torch.tensor(actions)
        rewards = torch.from_numpy(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 10e-6) #normalizing the batch reward signal 
        action = self.forward(obs).sample() #sampling a action from a normal distrubution based on the policy
        log_prob = self.forward(obs).log_prob(action) #calculating the log probability of policy action
        loss = torch.sum(torch.neg(torch.mul(log_prob, rewards))) #calculating the log probability of current action times the reward to approximate the policy gradient
        self.optim.zero_grad() #remove previous gradient in memory
        loss.backward() #backpropagate through loss with optimizer and find gradients
        utils.clip_grad_norm(self.network.parameters(), 10) #clip large gradients
        self.optim.step() #update wieghts in REINFORCE agent network by the gradient amount found from backpropagation
        return loss
    