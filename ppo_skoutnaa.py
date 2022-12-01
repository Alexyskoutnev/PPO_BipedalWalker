import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torch import optim
from torch import distributions
import numpy as np


from utils import build_NN

class ValueNet(nn.Module):
    #Constructor
    def __init__(self, s_dim):
        super(ValueNet, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(s_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    #Forward pass
    def forward(self, state):
        # breakpoint()
        return self.main(state)[:,0]



#AddBias module
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))
    
    def forward(self, x):
        bias = self._bias.t().view(1, -1)
        return x + bias

#Gaussian distribution with given mean & std.
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, x):
        return super().log_prob(x).sum(-1)
    
    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean

#Diagonal Gaussian module
class DiagGaussian(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(DiagGaussian, self).__init__()
        self.fc_mean = nn.Linear(inp_dim, out_dim)
        self.b_logstd = AddBias(torch.zeros(out_dim))
    
    def forward(self, x):
        mean = self.fc_mean(x)
        logstd = self.b_logstd(torch.zeros_like(mean))
        # print("logstd ->", logstd)
        # print("logstd exp() ->", logstd.exp())
        return FixedNormal(mean, logstd.exp())

#Policy Network
class PolicyNet(nn.Module):
    #Constructor
    def __init__(self, s_dim, a_dim):
        super(PolicyNet, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(s_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.dist = DiagGaussian(128, a_dim)
    
    #Forward pass
    def forward(self, state, noise_scale=False):
        feature = self.main(state)
        dist = self.dist(feature)

        if noise_scale:
            action = dist.mode()
        else:
            action = dist.sample()
        
        return action, dist.log_probs(action)
    
    #Choose an action (stochastically or deterministically)
    def choose_action(self, state, noise_scale=False):
        feature = self.main(state)
        dist = self.dist(feature)

        if noise_scale:
            return dist.mode()

        return dist.sample()
    
    #Evaluate a state-action pair (output log-prob. & entropy)
    def evaluate(self, state, action):
        feature = self.main(state)
        dist = self.dist(feature)
        # breakpoint()
        return dist.log_probs(action), dist.entropy()


class PPOAgent(nn.Module):
    def __init__(self, env, args, input_dim, output_dim, hidden_layers=[128, 128, 128], gamma = 0.99, esp_clip=0.2, std=0.5, activation = "ReLU", endActivation=nn.Tanh(), policy_update_steps = 5, critic_update_steps=1, actor_lr = 1e-4, critic_lr = 1e-4):
        super().__init__()
        self.input_dim = args['input_dim']
        self.output_dim = args['output_dim']
        # self.pi = build_NN(input_dim, output_dim, hidden_layers, args['activation'])
        self.pi = PolicyNet(self.input_dim, self.output_dim)
        self.pi_old = PolicyNet(self.input_dim, self.output_dim)

        # self.pi_old = build_NN(input_dim, output_dim, hidden_layers, args['activation'])
        self.pi_old.load_state_dict(self.pi.state_dict())
        # self.critic = build_NN(input_dim, 1, hidden_layers, args['activation'])
        self.critic = ValueNet(self.input_dim)
        # self.critic = nn.Sequential(nn.Linear(input_dim, 64), nn.Tanh(), nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))

        self.gamma = args['gamma']
        self.eps_clip = args['esp_clip']
        self.std = torch.tensor([args['std']])
        self.min_std = torch.tensor([args['std_min']])
        self.std_decay = torch.tensor([args['std_decay']])
        self.entropy = args['entropy']
        self.mseloss = nn.MSELoss()
        self.args = args
        self.timesteps = 0
        self.batch = args['batch']
        self.grad_update_clip = 0.5

        self.policy_update_steps = args['policy_updates']
        self.critic_update_steps = critic_update_steps


        self.critic_optim = optim.Adam(self.critic.parameters(), lr=args['critic_lr'])
        self.actor_optim = optim.Adam(self.pi.parameters(), lr=args['actor_lr'])

    def forward(self, obs, noise_scale=False):
        # breakpoint()
        feature = self.pi.main(obs)
        dist = self.pi.dist(feature)

        if noise_scale:
            action = dist.mode().detach()
        else:
            action = dist.sample().detach()
        # breakpoint()
        # action = torch.squeeze(action)
        return action, dist.log_probs(action), dist
    
    def evaluate(self, state, action):
        feature = self.pi.main(state)
        dist = self.pi.dist(feature)
        values = self.critic(state)
        # breakpoint()
        return dist.log_probs(action), dist.entropy(), values


    # def forward(self, obs, noise_scale=True):
    #     """Agent step function,
    #     input: current state that the agent is in (torch.tensor)
    #     output: action taken by agent with induced noise if not testing (torch.tensor)
    #     """
    #     action_logprob = None
    #     action = None
    #     action_distribution = None
    #     with torch.no_grad():
    #         action = self.pi(torch.from_numpy(obs))
    #     if noise_scale == True:
    #         sigma = self.std.expand(self.output_dim) #std of noise
    #         action_distribution = distributions.MultivariateNormal(action, torch.diag(sigma))
    #         action = action_distribution.sample().detach()
    #         action_logprob = action_distribution.log_prob(action)
    #     else:
    #         pass
    #         # sigma = self.std.expand(self.output_dim) #std of noise
    #         # action_distribution = distributions.MultivariateNormal(action, torch.diag(sigma))
    #         # action_logprob = action_distribution.log_prob(action)
    #     return action, action_logprob, action_distribution
    
    # def evaluate(self, obs, actions):
    #     """
    #     """
    #     action, log_prob, distribution = self.forward(obs.numpy())
    #     values = self.critic(obs)
    #     entropy = distribution.entropy()
    #     # breakpoint()
    #     return log_prob, entropy, values

    def decaySTD(self):
        print(f"STD decay -> {self.std}")
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
        # breakpoint()
        actions = torch.squeeze(torch.from_numpy(np.array(actions)))
        # actions = torch.from_numpy(np.array(actions))q
        # breakpoint()
        returns = torch.from_numpy(self.discount_rewards(np.array(rewards), np.array(terminals)))
        # returns = torch.from_numpy(self.discount_rewards1(np.array(rewards), obs))
        # breakpoint()
        rewards = torch.from_numpy(np.array(rewards))
        rewards = torch.from_numpy(np.array(rewards))
        rewards = (rewards - rewards.mean() / rewards.std() + 1e-6)
        next_obs = torch.from_numpy(np.array(next_obs))
        terminals = torch.from_numpy(np.array(terminals))
        old_action_logprob = torch.squeeze(torch.from_numpy(np.array(action_logprob)))
        old_state_value = torch.squeeze(torch.from_numpy(np.array(value)))
        advantage = returns - old_state_value
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)
        # breakpoint()
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
                print(f"j {j}")
                print(f"sample_mb {sample_n_mb}")
                sample_idx = rand_idx[j * sample_mb_size : (j+1)*sample_mb_size]
                sample_obs = obs[sample_idx]
                sample_actions = actions[sample_idx]
                sample_old_values = old_state_value[sample_idx]
                sample_returns = returns[sample_idx]
                sample_old_action_logprob = old_action_logprob[sample_idx]
                sample_advantages = advantage[sample_idx]
                sample_action_logprob, sample_entropy, sample_values = self.evaluate(sample_obs, sample_actions)
                ent = sample_entropy.mean()

                

                 #Compute value loss
                v_pred_clip = sample_old_values + torch.clamp(sample_values - sample_old_values, -self.eps_clip, self.eps_clip)
                print(f"return {sample_returns[:10]}")
                print(f"value {sample_values[:10]}")
                print(f"advantage {sample_advantages[:10]}")
                v_loss1 = (sample_returns - sample_values).pow(2)
                v_loss2 = (sample_returns - v_pred_clip).pow(2)
                v_loss = torch.max(v_loss1, v_loss2).mean()

                # breakpoint()
                ratio = (sample_action_logprob - sample_old_action_logprob).exp()
                print(f"ratio {ratio[:10]}")
                pg_loss1 = -sample_advantages * ratio
                
                # print(f"pg_loss1 {pg_loss1[:10]}")
                pg_loss2 = -sample_advantages * torch.clamp(ratio, 1.0-self.eps_clip, 1.0+self.eps_clip)
                # print(f"pg_loss2 {pg_loss2[:10]}")
                # pg_loss2 = -sample_advs * torch.clamp(ratio, 1.0-self.clip_val, 1.0+self.clip_val)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean() - self.entropy*ent
                # breakpoint()
                #Train actor
                self.actor_optim.zero_grad()
                pg_loss.backward()
                nn.utils.clip_grad_norm_(self.pi.parameters(), self.grad_update_clip)
                self.actor_optim.step()

                #Train critic
                self.critic_optim.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_update_clip)
                self.critic_optim.step()

                # breakpoint()
                # print(f"returns {sample_returns[:10]}")
                # print(f"value {sample_values[:10]}")
                # print(f"advantage {sample_advantage[:10]}")

                # #actor loss
                # # ratio = torch.exp(sample_action_logprob - sample_old_action_logprob.detach()).requires_grad_()
                # ratio =(sample_action_logprob - sample_old_action_logprob).exp()
                # print(f"sample logprob {sample_action_logprob[:10]}")
                # print(f"old logprob {sample_old_action_logprob[:10]}")
                # print(f"ratio {ratio[:10]}")
                # a_loss1 = -torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * sample_advantage
                # a_loss1_ult = -sample_advantage * torch.clamp(ratio, 1.0-self.eps_clip, 1.0+self.eps_clip)
                # print("a_loss1 ", a_loss1[10:])
                # print("a_loss1_ult ", a_loss1_ult[10:])
                # a_loss2 = -ratio * sample_advantage
                # a_loss2_ult = -sample_advantage * ratio
                # print("a_loss2 ", a_loss2[10:])
                # print("a_loss2_ult ", a_loss2_ult[10:])
                # # breakpoint()
                # loss_pi = (torch.max(a_loss1, a_loss2).mean() - self.entropy * sample_entropy.mean()).requires_grad_().float()
                # self.actor_optim.zero_grad()
                # loss_pi.backward()
                # nn.utils.clip_grad_norm_(self.pi.parameters(), self.grad_update_clip)
                # self.actor_optim.step()
                # # print(f"actor loss {loss_pi}")
                # # breakpoint()

                # #critic
                # # value_clip = sample_old_values + torch.clamp(sample_values - sample_old_values, -self.eps_clip, self.eps_clip)
                # # v_loss1 = (sample_returns - sample_values).pow(2)
                # # v_loss2 = (sample_returns - value_clip).pow(2)
                # # critic_loss = torch.max(v_loss1, v_loss2).mean()
                # critic_loss = torch.mean((returns - self.critic(obs))**2)
                # self.critic_optim.zero_grad()
                # critic_loss.backward()
                # nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_update_clip)
                # self.critic_optim.step()
                # # print(f"critic loss {critic_loss}")

                # breakpoint()




        # advantage = self.advantageEstimation(obs, next_obs, rewards, terminals)
        # advantage_1 = self.advantageEstimation1(returns, obs)
        # breakpoint()

        # logp_a_old, val_old, entropy_old = self.evaluate(obs, actions)
        # # breakpoint()
        # for i in range(self.policy_update_steps):
        #     # print(i)
        #     # print(f"old logp {logp_a_old}")
        #     loss_pi = self.pi_loss(obs, actions, returns, advantage, logp_a_old).mean()
        #     self.actor_optim.zero_grad()
        #     loss_pi.backward()
        #     self.actor_optim.step()
        #     # print(f"pi loss {loss_pi}")

        # for j in range(self.critic_update_steps):
        #     critic_loss = self.critic_loss(obs, returns)
        #     self.critic_optim.zero_grad()
        #     # critic_loss = critic_loss.retain_grad()
        #     critic_loss.backward()
        #     # print(f"critic_loss grad {critic_loss.grad}")
        #     self.critic_optim.step()

        #     # print(f"critic loss {critic_loss}")

        return pg_loss, v_loss

    
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

    def advantageEstimation(self, obs, next_obs, rewards, ternimals):
        """
            Helper function to estimate the advantage of transition
        """
        value_current = self.critic(obs)
        value_next = self.critic(next_obs)
        q_val = rewards + self.gamma * torch.squeeze(value_next)
        advantage_val = q_val - torch.squeeze(value_current)
        advantage_val = advantage_val.detach().numpy()
        return advantage_val

    def advantageEstimation1(self, returns, obs):
        """
            Helper function to estimate the advantage of transition
        """
        # breakpoint()
        with torch.no_grad():
            values = self.critic(obs)
            advantages = returns - torch.squeeze(values)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
        # breakpoint()
        return advantages



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
    

    def discount_rewards1(self, rewards, obs):
        returns = np.zeros_like(rewards)
        n_step = len(rewards)
        # breakpoint()
        for t in reversed(range(n_step)):
            if t == n_step - 1:
                # breakpoint()
                returns[t] = rewards[t] + self.gamma * self.critic(torch.from_numpy(np.expand_dims(obs[t], axis=0)))
                print(f"last values {self.critic(torch.from_numpy(np.expand_dims(obs[t], axis=0)))}")
            else:
                returns[t] = rewards[t] + self.gamma * returns[t+1]

        # breakpoint()
        return returns

