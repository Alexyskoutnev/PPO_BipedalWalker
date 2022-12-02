import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torch import optim
from torch import distributions
import numpy as np


from utils import build_NN

#AddBias module
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))
    
    def forward(self, x):
        bias = self._bias.t().view(1, -1)
        return x + bias

class Critic(nn.Module):
    def __init__(self, input_dim, output_dim, layers, activation):
        super().__init__()
        self.critic = build_NN(input_dim, 1, layers, activation)

    def forward(self, obs):
        value = self.critic(obs)
        return value[:,0]

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, layers, activation):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.pi = build_NN(input_dim, output_dim, layers, activation)
        self.logstd = AddBias(torch.zeros(output_dim))

    def forward(self, state, noise=True):
        action = self.pi(state)
        logstd = self.logstd(torch.zeros_like(action))
        if noise:
            action_distribution = distributions.Normal(action, logstd.exp())
            action = action_distribution.sample()
            log_prob = action_distribution.log_prob(action).sum(-1)
        else:
            action = action.detach()
            action_distribution = 0 
            log_prob = 0 
        action_logprob = log_prob
        return action, action_logprob, action_distribution

    def evaluate(self, state, action):
        logstd = self.logstd(torch.zeros_like(action))
        feature = self.pi(state)
        action_distribution = distributions.Normal(feature, logstd.exp())
        return action_distribution.log_prob(action).sum(-1), action_distribution.entropy()
    

        

class PPOAgent(nn.Module):
    def __init__(self, env, args, input_dim, output_dim, hidden_layers=[128, 128, 128], gamma = 0.99, esp_clip=0.2, std=0.5, activation = "ReLU", endActivation=nn.Tanh(), policy_update_steps = 5, critic_update_steps=1, actor_lr = 1e-4, critic_lr = 1e-4):
        super().__init__()
        self.input_dim = args['input_dim']
        self.output_dim = args['output_dim']
        # self.pi = build_NN(input_dim, output_dim, hidden_layers, args['activation'])
        # self.pi = build_NN(input_dim, output_dim, args['layers'], args['activation'])
        # self.pi = Actor(input_dim, output_dim)
        self.pi = Actor(input_dim, output_dim, args['layers'], args['activation'])
        self.pi_old = Actor(input_dim, output_dim, args['layers'], args['activation'])
        # self.pi_old = build_NN(input_dim, output_dim, args['layers'], args['activation'])
        # self.pi_old = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        # self.dist = DiagGaussian(128, args['output_dim'])
        
        # self.pi_old = PolicyNet(self.input_dim, self.output_dim)
        # breakpoint()

        # self.pi_old = build_NN(input_dim, output_dim, hidden_layers, args['activation'])
        self.pi_old.load_state_dict(self.pi.state_dict())
        # self.critic = build_NN(input_dim, 1, hidden_layers, args['activation'])
        # self.critic = ValueNet(self.input_dim)
        self.critic = Critic(input_dim, output_dim, args['layers'], args['activation'])
        # breakpoint()
        # self.critic = nn.Sequential(nn.Linear(input_dim, 64), nn.Tanh(), nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))

        self.gamma = args['gamma']
        self.eps_clip = args['esp_clip']
        self.std = torch.tensor([args['std']])
        self.b_logstd = AddBias(torch.zeros(args['output_dim']))
        self.min_std = torch.tensor([args['std_min']])
        self.std_decay = torch.tensor([args['std_decay']])
        self.entropy = args['entropy']
        self.args = args
        self.timesteps = 0
        self.batch = args['batch']
        self.grad_update_clip = args['grad_clip']

        self.policy_update_steps = args['policy_updates']
        self.critic_update_steps = critic_update_steps


        self.critic_optim = optim.Adam(self.critic.parameters(), lr=args['critic_lr'])
        self.actor_optim = optim.Adam(self.pi.parameters(), lr=args['actor_lr'])



    def forward(self, obs, noise=True):
        """Agent step function,
        input: current state that the agent is in (torch.tensor)
        output: action taken by agent with induced noise if not testing (torch.tensor)
        """
        action, log_prob, dist = self.pi.forward(obs, noise)
        return action, log_prob, dist

    
    def evaluate(self, obs, actions):
        """
        """
        log_prob, entropy = self.pi.evaluate(obs, actions)
        values = self.critic(obs)
        return log_prob, entropy, values

    def decaySTD(self):
        if self.std > self.min_std:
            self.std -= self.std_decay
        else:
            self.std = self.min_std

    def update(self, obs, actions, rewards, next_obs, terminals, action_logprob, value):
        """Agent update function that updates the weights in the actor and critic networks
        input: state (torch.tensor), action (torch.tensor), reward (torch.tensor), next state (torch.tensor), terminal (torch.tensor)
        outputs: actor loss and critic loss
        """
        obs = torch.from_numpy(np.array(obs))
        actions = torch.squeeze(torch.from_numpy(np.array(actions)))
        returns = torch.from_numpy(self.discount_rewards(np.array(rewards), np.array(terminals)))
        next_obs = torch.from_numpy(np.array(next_obs))
        terminals = torch.from_numpy(np.array(terminals))
        old_action_logprob = torch.squeeze(torch.from_numpy(np.array(action_logprob)))
        old_state_value = torch.squeeze(torch.from_numpy(np.array(value)))
        advantage = returns - old_state_value
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)
        transition_length = len(obs)
        rand_idx = np.arange(transition_length)
        sample_n_mb = transition_length // self.batch
        if sample_n_mb <= 0:
            sample_mb_size = transition_length
            sample_n_mb = 1
        else:
            sample_mb_size = self.batch
        for i in range(self.policy_update_steps):
            np.random.shuffle(rand_idx)
            # breakpoint()
            for j in range(sample_n_mb):
                sample_idx = rand_idx[j * sample_mb_size : (j+1)*sample_mb_size]
                sample_obs = obs[sample_idx]
                sample_actions = actions[sample_idx]
                sample_old_values = old_state_value[sample_idx]
                sample_returns = returns[sample_idx]
                sample_old_action_logprob = old_action_logprob[sample_idx]
                sample_advantages = advantage[sample_idx]
                sample_action_logprob, sample_entropy, sample_values = self.evaluate(sample_obs, sample_actions)
                
                ratio = (sample_action_logprob - sample_old_action_logprob).exp().requires_grad_()
                print(f"ratio {ratio[:10]}")
                actor_loss1 = -sample_advantages * ratio
                actor_loss1 = -sample_advantages * torch.clamp(ratio, 1.0-self.eps_clip, 1.0+self.eps_clip)
                pi_loss = torch.max(actor_loss1, actor_loss1).mean() - self.entropy*sample_entropy.mean()
                self.actor_optim.zero_grad()
                pi_loss.backward()
                nn.utils.clip_grad_norm_(self.pi.parameters(), self.grad_update_clip)
                self.actor_optim.step()

                #Train critic
                critic_loss = ((sample_returns - sample_values)**2).mean()
                self.critic_optim.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_update_clip)
                self.critic_optim.step()

        return pi_loss, critic_loss

    
    def pi_loss(self, obs, actions, rewards, advantage, logp_a_old):
        """
        Critic loss calculation
        """
        advantage = torch.from_numpy(advantage)
        logp_a, val, entropy = self.evaluate(obs, actions)
        ratio = torch.exp(logp_a - logp_a_old.detach()).requires_grad_()
        print(f"ratio: {ratio[1:20]}")
        clipped_advantage = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
        print(f"clipped advantage: {clipped_advantage[1:20]}")
        loss_pi = (-torch.min(ratio * advantage, clipped_advantage) + 0.5 * torch.mean((torch.squeeze(val.float()) - rewards.float())**2) - self.entropy * entropy).requires_grad_().float()
        # breakpoint()
        return loss_pi

    def critic_loss(self, obs, rew):
        loss_v = torch.mean((self.critic(obs) - rew)**2)
        print(f"critic val {self.critic(obs)[1:10]}")
        return loss_v

    def discount_rewards(self, rewards, terminals):
        discount_rewards = np.zeros_like(rewards)
        discount_reward = 0
        # breakpoint()
        for i, (reward, term) in enumerate(zip(reversed(rewards), reversed(terminals))):
            # breakpoint()
            if term:
                discount_reward = 0
            discount_reward = reward + self.gamma * discount_reward
            discount_rewards[i] = discount_reward
        discount_rewards = np.flip(discount_rewards).copy()
        return discount_rewards
    
