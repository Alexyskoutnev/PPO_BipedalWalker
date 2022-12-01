import argparse
import gym
import torch
import numpy as np
import time
import os
import datetime
from tqdm import tqdm
from utils import Path, ReplayBuffer, OrnsteinUhlenbeckActionNoise
from ppo_skoutnaa import PPOAgent

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter(log_dir="./runs/BipedalWaler-v2" + time.strftime("%Y_%m_%d_%H_%M"))




STEPS = 1000

def setup(args):
    env = gym.make("BipedalWalker-v3")
    obs_shape = env.observation_space.shape[0]
    action_shape = env.action_space.shape[0]
    args['input_dim'] = obs_shape
    args['output_dim'] = action_shape
    agent = PPOAgent(env, args, args['input_dim'], args['output_dim'])
    agent.pi.train()
    agent.critic.train()
    # breakpoint()
    return env, agent


def train(args):
    env, agent = setup(args)
    training_time = time.time()
    cum_rewards = list()
    average_reward_100 = 0
    successfulTrain = False
    eval_rewards = list()
    eval_rewards_mean = 0
    trainDDPG = False
    # noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(args['output_dim']), sigma=args['OU_sigma'] * np.ones(args['output_dim']), theta=args['OU_theta'], scale_timestep=args['OU_scaletimesteps'], final_scale=args['OU_final_scale'])
    cum_rewards, policy_losses, value_losses, mean_test_rewards = [], [], [], []
    training_time = time.time()
    average_reward_100 = 0
    successfulTrain = False
    for epoch in tqdm(range(args['epochs'])):
        path = Path()
        done = False
        obs = env.reset()
        # noise.reset()
        i = 0
        epoch_return = 0
        while(not done and i < args["steps"]): 
            state = torch.tensor(np.expand_dims(obs, axis=0), dtype=torch.float32)
            action, action_logprob, _ = agent.forward(state)
            # value = agent.critic(torch.from_numpy(state))
            value = agent.critic(state)
            # breakpoint()
            # next_obs, reward, done, _ = env.step(action)
            next_obs, reward, done, _ = env.step(action.numpy()[0])
            epoch_return += reward
            path.add(obs, action, next_obs, reward, done, action_logprob.item(), value.item())
            if done == True:
                break
            obs = next_obs
            agent.timesteps += 1
            if agent.timesteps % args['decay_timestep'] == 0:
                # breakpoint()
                print("decay")
                agent.decaySTD()
            i += 1

        print(f"total epc steps {i}")
        print(f"agent timestep {agent.timesteps}")
        # breakpoint()
        cum_rewards.append(epoch_return)
        loss_pi, loss_critic = agent.update(path.obs, path.action, path.reward, path.next_obs, path.done, path.action_logprob, path.value)
        print(f"[{epoch}]: loss_pi {loss_pi} \t loss_critic {loss_critic}")
        print(f"[{epoch}]: epoch_return: {epoch_return}")

   #      writer.add_scalar('training_rewards', epoch_return, epoch)
     

  
    #         writer.add_scalar('loss/train', loss, epoch)
    #         writer.add_scalar('discounted_rewards/train', discounted_rewards.sum(), epoch)
    #         agent.steps += 1
    #         print(f"{epoch}: reinforce loss {loss}")
    #         print(f"{epoch}: epoch reward {discounted_rewards.sum()}")
         
        if epoch % args['test_every_epoch'] and args['algorithm'] == 'reinforce':
            eval_rewards = eval(env, agent, args)
            # writer.add_scalar('test_rewards', eval_rewards, epoch)
        if epoch % args['test_every_epoch'] == 0:
            eval_rewards = eval(env, agent, args)
    #         writer.add_scalar('test_rewards', eval_rewards, epoch)
        if len(cum_rewards) > 100:
            average_reward_100 = np.array(cum_rewards[-100:]).mean()
            print(f"{epoch}: average_100 rewards {average_reward_100}")
    #         writer.add_scalar('mean_100_rewards/train', average_reward_100, epoch)
    #     if average_reward_100 > 200 or epoch >= args['epochs'] - 1:
    #         successfulTrain = True
    #         break
        



    # end_train_time = time.time() - training_time
    # if successfulTrain:
    #     if args['algorithm'] == 'reinforce':
    #         os.makedirs(args['save_dir'], exist_ok=True)
    #         print("MODEL HAS BEEN SAVE")
    #         print(f"total train time -> {end_train_time}")
    #         NOWTIMES = datetime.datetime.now()
    #         curr_time = NOWTIMES.strftime("%y%m%d_%H%M%S")
    #         filenameREINFORCE = "reinforce_skoutaa_" + str(curr_time) + ".pt"
    #         pathREINFORCE = os.path.join(args['save_dir'], filenameREINFORCE)
    #         torch.save(agent.network.state_dict(), pathREINFORCE)
    #         print(f"saved REINFORCE model weights path: {pathREINFORCE}")
    #     if args['algorithm'] == 'ddpg':
    #         os.makedirs(args['save_dir'], exist_ok=True)
    #         print("MODEL HAS BEEN SAVE")
    #         print(f"total train time -> {end_train_time}")
    #         NOWTIMES = datetime.datetime.now()
    #         curr_time = NOWTIMES.strftime("%y%m%d_%H%M%S")
    #         filenameActor = "DDPG_MoutainCarContinuous_actor_skoutaa_" + str(curr_time) + ".pt"
    #         filenameCritic = "DDPG_MoutainCarContinuous_critic_skoutaa_" + str(curr_time) + ".pt"
    #         filenameCombined = "DDPG_MoutainCarContinuous_critic_actor_skoutaa_" + str(curr_time) + ".pt"
    #         pathActor = os.path.join(args['save_dir'], filenameActor)
    #         pathCritic = os.path.join(args['save_dir'], filenameCritic)
    #         pathCombined = os.path.join(args['save_dir'], filenameCombined)
    #         params = save_params(agent)
    #         torch.save(agent.pi.state_dict(), pathActor)
    #         torch.save(agent.q.state_dict(), pathCritic)
    #         torch.save(params, pathCombined)
    #         print(f"combined actor-critic model weights path: {pathCombined}")
    env.close()
    # writer.close()


def save_params(agent):
    params = {
        "actor_state_dict" : agent.pi.state_dict(),
        "critic_state_dict" : agent.q.state_dict()
    }
    return params

def eval(env, agent, args):
    result = []
    steps = 0
    dones = 0
    for epoch in range(args['eval_epoch']):
        obs = env.reset()
        done = False
        epoch_return = 0.0
        done = False
        while not done and args['eval_steps'] > steps:
            if args['render']:
                env.render()
            state = torch.tensor(np.expand_dims(obs, axis=0), dtype=torch.float32)
            action, _, _ = agent.forward(state)
            # breakpoint()
            # next_state, reward, done, _ = env.step(action)
            next_obs, reward, done, _ = env.step(action.numpy()[0])
            epoch_return += reward
            obs = next_obs
            steps += 1
            if done == True:
                dones += 1
        result.append(epoch_return)
        if dones > 1:
            break
    result = np.array(result).reshape(-1,1)
    print(f"!!!!!!!EVAL Mean ->>{np.mean(result)}")
    return np.mean(result)










if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algorithm', type=str, help='agent algorithm', default="ppo")
    parser.add_argument('-env', type=str, default='BipedalWaler-v2')
    parser.add_argument('-act', "--activation", type=str, default="ReLU")
    parser.add_argument('-l', '--layers', type=list, default=[64, 64, 64])
    parser.add_argument('-lr','--learning_rate', type=float, default=10e-3)
    parser.add_argument('-ep', '--epochs', type=int, default=2000)
    parser.add_argument('-st', '--steps', type=int, default=2048)
    parser.add_argument('-g', '--gamma', type=float, default=0.99)
    parser.add_argument('-b', '--batch', type=int, default=64)
    parser.add_argument('-n', '--noise', type=float, default=0.1)
    parser.add_argument('-tc', '--test_cycles', type=int, default=2000)
    parser.add_argument('-ti', '--test_every_epoch', type=int, default=25)
    parser.add_argument('-eval_ep', '--eval_epoch', type=int, default=5)
    parser.add_argument('-eval_st', '--eval_steps', type=int, default=500)
    parser.add_argument("--off-render", dest="render", action="store_false", help="turn off rendering")
    parser.add_argument("--save_dir", type=str, default='./models/')
    parser.add_argument("-std", "--std", type=float, default=0.1)
    parser.add_argument("-stdD", "--std_decay", type=float, default=0.01)
    parser.add_argument("-stdmin", "--std_min", type=float, default=0.001)
    parser.add_argument("-decay_t", "--decay_timestep", type=int, default=10000)



    parser.add_argument("-entr", "--entropy", type=float, default=0.01)
    parser.add_argument("-k", "--policy_updates", type=int, default=4)
    parser.add_argument("-a_lr", "--actor_lr", type=float, default=10e-4)
    parser.add_argument("-c_lr", "--critic_lr", type=float, default=10e-4)
    parser.add_argument("-clip", "--esp_clip", type=float, default=0.2)
    args = vars(parser.parse_args())
    train(args)