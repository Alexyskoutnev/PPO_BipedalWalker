import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torch import optim
from torch import distributions
import numpy as np

from utils import build_NN

class AddBias(nn.Module):
    #Bias Layer to help exploration (makes PPO converge alot faster)
    # adapted from {https://medium.com/analytics-vidhya/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8}
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))
    
    def forward(self, x):
        """
        Bias addition to the action from normal distribution
        Input: agent action (torch.tensor)
        Output: biased agent action (torch.tensor)
        """
        bias = self._bias.t().view(1, -1)
        return x + bias

class Critic(nn.Module):
    """
    Critic class for PPO algorithm that has the value function for the agent
    """
    def __init__(self, input_dim, output_dim, layers, activation):
        super().__init__()
        self.critic = build_NN(input_dim, 1, layers, activation)

    def forward(self, obs):
        """
        Function that predicts the agent value based on the agent observation
        Input: agent oberservation (torch.tensor)
        Output: state value (torch.tensor) 
        """
        value = self.critic(obs)
        return value[:,0]

class Actor(nn.Module):
    """
    Actor class for the PPO algorithm that has state-action policy
    """
    def __init__(self, input_dim, output_dim, layers, activation):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.pi = build_NN(input_dim, output_dim, layers, activation)
        self.logstd = AddBias(torch.zeros(output_dim))

    def forward(self, state, noise=True):
        """
        Function that predicts the agent's next action occcuring to the agent state
        Input: agent state (torch.tensor)
        Output: Action (torch.tensor)
        """
        action = self.pi(state) #policy of PPO actor
        logstd = self.logstd(torch.zeros_like(action)) #varying std to induce noise into policy
        if noise:
            action_distribution = distributions.Normal(action, logstd.exp()) #sample action from multivariate normal with noisy std
            action = action_distribution.sample() #sample a single action
            log_prob = action_distribution.log_prob(action).sum(-1) #calculate the total log probability of sampled action
        else:
            action = action.detach() #sample actio without noise
            action_distribution = 0 
            log_prob = 0 
        action_logprob = log_prob
        return action, action_logprob, action_distribution

    def evaluate(self, state, action):
        """
        Evaluation function to calculate the action log probability, distribution entropy
        Input: agent state, action (torch.tensor)
        Output: agent log probability, distribution entropy
        """
        logstd = self.logstd(torch.zeros_like(action)) #changing std parameters to bring noise into action
        action_pred = self.pi(state) #predicted action from sampled state
        action_distribution = distributions.Normal(action_pred, logstd.exp())
        return action_distribution.log_prob(action).sum(-1), action_distribution.entropy()

class PPOAgent(nn.Module):
    """
    Interface class for PPO agent that contains the actor and critic network to process the PPO algorithm
    """
    def __init__(self, env, args, input_dim, output_dim, hidden_layers=[128, 128, 128], gamma = 0.99, esp_clip=0.2, activation = "ReLU", endActivation=nn.Tanh(), policy_update_steps = 5, critic_update_steps=1, actor_lr = 1e-4, critic_lr = 1e-4):
        super().__init__()
        self.input_dim = args['input_dim']
        self.output_dim = args['output_dim']
        self.pi = Actor(input_dim, output_dim, args['layers'], args['activation'])
        self.pi_old = Actor(input_dim, output_dim, args['layers'], args['activation'])
        self.pi_old.load_state_dict(self.pi.state_dict())
        self.critic = Critic(input_dim, output_dim, args['layers'], args['activation'])
        self.gamma = args['gamma']
        self.eps_clip = args['esp_clip']
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
        Input: current state that the agent is in (torch.tensor)
        Output: action taken by agent with induced noise if not testing (torch.tensor)
        """
        action, log_prob, dist = self.pi.forward(obs, noise)
        return action, log_prob, dist

    
    def evaluate(self, obs, actions):
        """
        Evaluation function for the PPO agent, returns the entropy and log probability of the a sampled action along its value
        Input: sampled state and action (torch.tensor)
        Output: action log probability, distribution entropy, state values (torch.tensor)
        """
        log_prob, entropy = self.pi.evaluate(obs, actions)
        values = self.critic(obs) #sample value function
        return log_prob, entropy, values

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
        advantage = returns - old_state_value #calulation of advantage for agent trajectory
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)
        transition_length = len(obs)
        rand_idx = np.arange(transition_length) #index each trajectory point
        number_of_strides = transition_length // self.batch #calculate the number of batch strides
        if transition_length <= self.batch:
            sample_length = transition_length
            number_of_strides = 1
            prob_vec = [1/transition_length for i in range(transition_length)]
        else:
            sample_length = self.batch 
            prob_vec = [1/sample_length for i in range(sample_length)]
        for i in range(self.policy_update_steps):
            np.random.shuffle(rand_idx)
            for j in range(number_of_strides):
                #sampling random indexes in trajectory to induce more variance in sampling
                sample_idx = rand_idx[j * sample_length : (j+1)*sample_length]
                sample_obs = obs[sample_idx]
                sample_actions = actions[sample_idx]
                sample_old_values = old_state_value[sample_idx]
                sample_returns = returns[sample_idx]
                sample_old_action_logprob = old_action_logprob[sample_idx]
                sample_advantages = advantage[sample_idx]
                sample_action_logprob, sample_entropy, sample_values = self.evaluate(sample_obs, sample_actions) #evaluation of sampled results
                
                #train actor
                ratio = (sample_action_logprob - sample_old_action_logprob).exp().requires_grad_() #compared the ratio to new policy compared to old policy
                actor_loss1 = -sample_advantages * ratio #calulate the actor loss without clipping the sample_advantages * ratio
                actor_loss1 = -sample_advantages * torch.clamp(ratio, 1.0-self.eps_clip, 1.0+self.eps_clip) #calulate the actor loss with clipping the sample_advantages * ratio to policy from diverging too much
                pi_loss = torch.max(actor_loss1, actor_loss1).mean() - self.entropy*sample_entropy.mean() #calculates the actor loss based on a clipped policy and trajectory advantages
                self.actor_optim.zero_grad() #remove previous stored gradients 
                pi_loss.backward() #perform backpropagation to obtain next update gradients based on actor loss
                nn.utils.clip_grad_norm_(self.pi.parameters(), self.grad_update_clip) #clip exploding gradient to a max of self.grad_update_clip 
                self.actor_optim.step() #updated the weights in the NN with update gradient vector

                #train critic
                value_prediction = sample_old_values + torch.clamp(sample_values - sample_old_values, -self.eps_clip, self.eps_clip) #Helps control the value prediction so ratio values don't jump
                critic_loss1 = ((sample_returns - value_prediction)**2) #find the loss from sample returns and value prediction
                critic_loss2 = ((sample_returns - sample_values)**2) #find loss from sample returns and sampled values from the critic NN
                critic_loss = torch.max(critic_loss1, critic_loss2).mean() #Take the max loss of the two critic errors
                self.critic_optim.zero_grad() #remove previous stored gradients 
                critic_loss.backward() #perform backpropagation to obtain next update gradients based on critic loss
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_update_clip) #clip exploding gradient to a max of self.grad_update_clip 
                self.critic_optim.step() #updated the weights in the NN with update gradient vector

        return pi_loss, critic_loss

    def discount_rewards(self, rewards, terminals):
        """
        Helper function to find the discount total reward from reward trajectory
        Input: reward trajectory (np.array())
        Ouput: cumulative discounted reward trajectory (np.array())
        """
        discount_rewards = np.zeros_like(rewards)
        discount_reward = 0
        for i, (reward, term) in enumerate(zip(reversed(rewards), reversed(terminals))):
            if term:
                discount_reward = 0
            discount_reward = reward + self.gamma * discount_reward
            discount_rewards[i] = discount_reward
        discount_rewards = np.flip(discount_rewards).copy()
        return discount_rewards
    
