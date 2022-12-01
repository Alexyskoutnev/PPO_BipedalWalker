import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from torch import optim
from torch import normal
from utils import build_NN
from torch.nn import utils
from torch.nn.utils import clip_grad_norm_

activationDic = {"Relu": nn.ReLU(), "Tanh": nn.Tanh()}
ReLU = nn.ReLU()
Tanh = nn.Tanh()


def init_weights_helper(size):
    """Helper function to create network weights with small values to help slow down network convergence 
    and stop exploding gradient"""
    fanin = size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    """The policy of the DDPG agent that inherits from torch.nn module"""
    def __init__(self, input_dim, output_dim, hidden_layer_1=256, hidden_layer_2=256, init_w=3e-3, activation = "Tanh"):
        super().__init__()
        "creates a linear neural network with one hidden layer"
        self.fc1 = nn.Linear(input_dim, hidden_layer_1)
        self.fc2 = nn.Linear(hidden_layer_1, hidden_layer_2)
        self.fc3 = nn.Linear(hidden_layer_2, output_dim)
        self.init_weights(init_w) #helper function to keep initial network weight small
        
    def init_weights(self, weights):
        """Helper function to change network weight to small values (helps reduce possible wieght explosions)"""
        self.fc1.weight.data = init_weights_helper(self.fc1.weight.data.size())
        self.fc2.weight.data = init_weights_helper(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-weights, weights)
    
    def forward(self, obs):
        """Policy action function that recieves oberservation tensor and outputs an action tensor"""
        out = self.fc1(obs)
        out = ReLU(out) #nonlinear activation function
        out = self.fc2(out)
        out = ReLU(out) #nonlinear activation function 
        out = self.fc3(out)
        out = Tanh(out) #nonlinear activation function 
        return out


class Q_Critic(nn.Module):
    """The q-value function approximator for the agent state (the Critic)"""
    def __init__(self, input_dim, output_dim, hidden_layer_1=400, hidden_layer_2=300, init_w=3e-3, activation = "ReLU"):
        super().__init__()
        self.fc1 = nn.Linear(input_dim+output_dim, hidden_layer_1) #we combine the state and action values together to approximate the q-value 
        self.fc2 = nn.Linear(hidden_layer_1, hidden_layer_2)
        self.fc3 = nn.Linear(hidden_layer_2, 1)
        self.init_weights(init_w) #helper function to keep initial network weight small

    def init_weights(self, weights):
        """Helper function to change network weight to small values (helps reduce possible wieght explosions)"""
        self.fc1.weight.data = init_weights_helper(self.fc1.weight.data.size())
        self.fc2.weight.data = init_weights_helper(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-weights, weights)
    
    def forward(self, obs, action):
        """Q-value function approximator with input of state, action pairs 
        and outputs coresponding q-value"""
        if len(action.shape) == 1:
            action = torch.unsqueeze(action, 1)
        out = torch.cat([obs, action], 1) #combine the action and state together for input of q-value function approximator
        out = self.fc1(out)
        out = ReLU(out) #nonlinear activation function 
        out = self.fc2(out)
        out = ReLU(out) #nonlinear activation function 
        out = self.fc3(out) #outputs the approximate q-value
        return out

class DDPG(object):
    """Main DDPG agent that has a critic and actor function approximator"""
    def __init__(self, env, params, actor_learning_rate = 10e-4, critic_learning_rate = 10e-3, activationF = "ReLU", gamma=0.99, polyak = 0.995):
        self.input_dim = params['input_dim']
        self.output_dim = params['output_dim'] 
        self.pi = Actor(self.input_dim, self.output_dim, params['actor_dim'][0], params['actor_dim'][1]).float()
        self.q = Q_Critic(self.input_dim, self.output_dim, params['critic_dim'][0], params['critic_dim'][1]).float()
        self.pi_target = deepcopy(self.pi) #copy the exact weights and bias of the parent actor
        self.q_target = deepcopy(self.q) #copy the exact weights and bias of the parent critic
        self.actor_lr = params['actor_learning_rate']
        self.critic_lr = params['critic_learning_rate']
        self.gamma = params['gamma']
        self.polyak = params['polyak'] #The delay factor in updating the target actor and critic weights
        self.optim_pi = optim.Adam(list(self.pi.parameters()), lr=self.actor_lr, eps=1e-7, weight_decay=0.000001) #actor optimizer to backpropagate through the NN networks and find coresponding gradients
        self.optim_q = optim.Adam(list(self.q.parameters()), self.critic_lr, weight_decay=0.000001) #critic optimizer to backpropagate through the NN networks and find coresponding gradients
        self.env = env
        self.grad_clip_ac = params["ac_grad_clip"]
        self.grad_clip_cr = params["cr_grad_clip"]

    def forward(self, obs, noise_scale=None):
        """Agent step function,
        input: current state that the agent is in (torch.tensor)
        output: action taken by agent with induced noise if not testing (torch.tensor)
        """
        if type(obs) == torch.Tensor: 
            self.pi.eval()
            action = self.pi(obs.float()) #action selection from actor policy
            self.pi.train()
            if noise_scale is not None:
                noise = torch.tensor(noise_scale.noise()) #OU noise addition to action (promote agent exploration)
                action += noise
        else:
            self.pi.eval()
            action = self.pi(torch.from_numpy(obs).float())  #action selection from actor policy
            self.pi.train()
            if noise_scale is not None:
                noise = torch.tensor(noise_scale.noise()) #OU noise addition to action (promote agent exploration)
                action += noise
        return action.clamp(self.env.action_space.low[0], self.env.action_space.high[0]) #clip actions depending on enviroment action bounds

    def update(self, obs, actions, rewards, next_obs, done):
        """Agent update function that updates the weights in the actor and critic networks
        input: state (torch.tensor), action (torch.tensor), reward (torch.tensor), next state (torch.tensor), terminal (torch.tensor)
        outputs: actor loss and critic loss
        """

        for p in self.q_target.parameters(): #helps reduce computation cost for optumizer
            p.requires_grad = False
        gradient_clip_ac = self.grad_clip_ac #clip actor gradients if too large
        gradient_clip_cr = self.grad_clip_cr #clip critic gradients if too large

        #optimizing Q value function
        loss_q = self.lossQ(obs, actions, rewards, next_obs, done) #find the critic loss function
        self.optim_q.zero_grad() #remove previous gradient in memory
        loss_q.backward() #backpropagate with optimizer and find gradient
        clip_grad_norm_(self.q.parameters(), gradient_clip_cr) #clip large gradients
        self.optim_q.step() #update wieghts in critic network by the gradient amount found from backpropagation

        #optimizing PI policy
        for p in self.q.parameters(): #helps reduce computation cost for optumizer
            p.requires_grad = False
        self.optim_pi.zero_grad() #remove previous gradient in memory
        loss_pi = self.lossPi(obs) #find the actor loss function
        loss_pi.backward() #backpropagate with optimizer and find gradient
        clip_grad_norm_(self.pi.parameters(), gradient_clip_ac) #clip large gradients
        self.optim_pi.step() #update wieghts in critic network by the gradient amount found from backpropagation
        for p in self.q.parameters(): #helps reduce computation cost for optumizer
            p.requires_grad = True

        #perform soft update on target networks by a factor of the polyak metric
        with torch.no_grad():
            for p_target, p in zip(self.q_target.parameters(), self.q.parameters()):
                p_target.data.copy_(self.polyak * p_target.data + (1 - self.polyak) * p.data)
            for p_target, p in zip(self.pi_target.parameters(), self.pi.parameters()):
                p_target.data.copy_(self.polyak * p_target.data + (1 - self.polyak) * p.data)
        return loss_q, loss_pi


    def lossQ(self, obs, actions, rewards, next_obs, done):
        """Helper function to calculate the critic loss by approximating the
           MSE Bellman error between the next q-value and current q-value"""
        obs = torch.tensor(obs, dtype=torch.float32)
        next_obs = torch.tensor(next_obs, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        with torch.no_grad(): #no gradients are needed
            q_vals_targ = self.q_target.forward(next_obs, torch.squeeze(self.pi_target(next_obs))) #next state q_value via target critic network and target actor policy
            target_q = rewards + self.gamma * torch.mul(q_vals_targ, (torch.ones(done.shape[0]) - done)) #reward + discounted constant * next state q-values
        q_vals = self.q.forward(obs, actions) #current state, action q-value
        loss_q = F.mse_loss(q_vals, target_q) #difference between current state, and action q-value and next state, and action q-value
        return loss_q

    def lossPi(self, obs):
        """Helper function to calculated the actor loss by taking a gradient ascent based on the 
           current q-value of state, and action (from parent policy)"""
        obs = torch.tensor(obs, dtype=torch.float32, requires_grad=True)
        q_val = torch.tensor(self.q.forward(obs, torch.squeeze(self.pi(obs))), dtype=torch.float32, requires_grad=True) #recieve current action from parent policy and estimate the current state, action q-value
        pi_loss = -torch.mean(q_val) #take the mean of the q-values from the batch and make them negative because we are trying to increase the mean q-value of the agent
        return pi_loss



        