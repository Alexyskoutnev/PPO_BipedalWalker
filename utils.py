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