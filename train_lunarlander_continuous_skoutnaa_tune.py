import argparse
import gym
import torch
import numpy as np
import time
import os
import datetime
from tqdm import tqdm
# from reinforce_skoutnaa import REINFORCEAGENT
import matplotlib.pyplot as plt
from ddpg_skoutnaa import DDPG
from utils import Path, ReplayBuffer
import time
from torch.utils.tensorboard import SummaryWriter




# from noise import OrnsteinUhlenbeckActionNoise
from utils import OrnsteinUhlenbeckActionNoise

writer = SummaryWriter(log_dir="./runs/lunarlander_testing_" + time.strftime("%Y_%m_%d_%H_%M"))

def setup(args):
    env = gym.make("LunarLanderContinuous-v2")
    obs_shape = env.observation_space
    args['input_dim'] = obs_shape.shape[0]
    action_space = env.action_space
    args['output_dim'] = action_space.shape[0]
    agent = None
    if args['algorithm'] == 'ddpg':
        agent = DDPG(env, args, args['actor_learning_rate'], args['critic_learning_rate'], args['activation'], args['gamma'], polyak=args['polyak'])
        agent.q.train()
        agent.pi.train()
        agent.q_target.train()
        agent.pi_target.train()
    elif args['third']:
        agent = DDPG(config=config)
    return env, agent


def train(args):
    env, agent = setup(args)
    rollouts = list()
    done = False
    show = False
    trainDDPG = False
    replayBuff = ReplayBuffer(int(args['maxBuffer']))
    noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(args['output_dim']), sigma=args['OU_sigma']* np.ones(args['output_dim']), theta=args['OU_theta'])
    cum_rewards, policy_losses, value_losses, mean_test_rewards = [], [], [], []
    training_time = time.time()
    average_reward_100 = 0
    successfulTrain = False


    for epoch in tqdm(range(args['epochs'])):
        path = Path()
        done = False
        obs = env.reset()
        noise.reset()
        t = 0
        epoch_return = 0
        while(not done and t < args["steps"]): 
            if args['algorithm'] == 'ddpg':
                if trainDDPG == False:
                    action = torch.tensor(env.action_space.sample())
                    action = action.detach().numpy()
                else:
                    action = agent.forward(obs, noise)
                    action = action.detach().numpy()
            next_obs, reward, done, _ = env.step(action)
            epoch_return += reward
            path.add(obs, action, next_obs, reward, done)
            obs = next_obs
            t += 1
            if len(replayBuff) > args['batch'] and trainDDPG == True:
                observations, actions, next_observation, rewards, terminals = replayBuff.sample(args['batch'])
                loss_q, loss_pi = agent.update(observations, actions, rewards, next_observation, terminals)
                writer.add_scalar('loss_q/train', loss_q, epoch)
                writer.add_scalar('loss_pi/train', loss_pi, epoch)
            if done == True:
                break

        
        cum_rewards.append(epoch_return)
        writer.add_scalar('training_rewards', epoch_return, epoch)

        if args['algorithm'] == 'ddpg':
            replayBuff.add(path)
            if len(replayBuff) >= args['DDPG_Start'] and trainDDPG is not True:
                trainDDPG = True
            elif trainDDPG:
                print(f"{epoch} epoch_return: {epoch_return}")
                print(f"{epoch} loss_q: {loss_q}, loss_pi {loss_pi}")
                print(f"buff: {len(replayBuff)}")

        if epoch % args['test_every_epoch'] == 0 and len(replayBuff) > args['DDPG_Start']:
            eval_rewards = eval(env, agent, args)
            writer.add_scalar('test_rewards', eval_rewards, epoch)
        if len(cum_rewards) > 100:
            average_reward_100 = np.array(cum_rewards[-100:]).mean()
            print(f"{_}: average_100 rewards {average_reward_100}")
            writer.add_scalar('mean_100_rewards/train', average_reward_100, epoch)
        if average_reward_100 > 200 or epoch >= args['epochs'] - 1:
            successfulTrain = True
            break

    end_train_time = time.time() - training_time
    if successfulTrain:
        os.makedirs(args['save_dir'], exist_ok=True)
        print("MODEL HAS BEEN SAVE")
        print(f"total train time -> {end_train_time}")
        NOWTIMES = datetime.datetime.now()
        curr_time = NOWTIMES.strftime("%y%m%d_%H%M%S")
        filenameActor = "DDPG_lunarlander_actor_skoutaa_" + str(curr_time) + ".pt"
        filenameCritic = "DDPG_lunarlander_critic_skoutaa_" + str(curr_time) + ".pt"
        filenameCombined = "DDPG_lunarlander_critic_actor_skoutaa_" + str(curr_time) + ".pt"
        pathActor = os.path.join(args['save_dir'], filenameActor)
        pathCritic = os.path.join(args['save_dir'], filenameCritic)
        pathCombined = os.path.join(args['save_dir'], filenameCombined)
        params = save_params(agent)
        torch.save(agent.pi.state_dict(), pathActor)
        torch.save(agent.q.state_dict(), pathCritic)
        torch.save(params, pathCombined)
        print(f"combined actor-critic model weights path: {pathCombined}")
    env.close()
    writer.close()

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
    for epoch in range(args['eval_epc']):
        state = env.reset()
        done = False
        epoch_return = 0.0
        done = False
        while not done and args['eval_steps'] > steps:
            if args['render']:
                env.render()
            action = agent.forward(state).detach().numpy()
            next_state, reward, done, _ = env.step(action)
            epoch_return += reward
            state = next_state
            steps += 1
            if done == True:
                dones += 1
        result.append(epoch_return)
        if dones > 5:
            break
    result = np.array(result).reshape(-1,1)
    print(f"!!!!!!!EVAL Mean ->>{np.mean(result)}")
    return np.mean(result)









if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algorithm', type=str, help='agent algorithm', default="ddpg")
    parser.add_argument('-env', type=str, default='LunarLanderContinuous-v2')
    parser.add_argument('--input_dim', type=int, default=2)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('-l', '--layers', type=int, default=2)
    parser.add_argument('-s','--size', type=int, default=128)
    parser.add_argument('-lr','--learning_rate', type=float, default=10e-2)
    parser.add_argument('-A_lr','--actor_learning_rate', type=float, default=0.003)
    parser.add_argument('-C_lr','--critic_learning_rate', type=float, default=0.01)
    parser.add_argument('-act','--activation', type=str, default='ReLU')
    parser.add_argument('-ep', '--epochs', type=int, default=1000)
    parser.add_argument('-st', '--steps', type=int, default=1000)
    parser.add_argument('-g', '--gamma', type=float, default=0.99)
    parser.add_argument('-ta', '--tau', type=float, default=0.005)
    parser.add_argument('-buf', '--maxBuffer', type=int, default=10e6)
    parser.add_argument('-DDPG_S', '--DDPG_Start', type=int, default=20000)
    parser.add_argument('-DDPG_mean', '--DDPG_mean', type=float, default=0)
    parser.add_argument('-DDPG_std', '--DDPG_std', type=float, default=0.50)
    parser.add_argument('-DDPG_a_hs1', '--DDPG_a_hs1', type=int, default=256)
    parser.add_argument('-DDPG_a_hs2', '--DDPG_a_hs2', type=int, default=256)
    parser.add_argument('-DDPG_c_hs1', '--DDPG_c_hs1', type=int, default=256)
    parser.add_argument('-DDPG_c_hs2', '--DDPG_c_hs2', type=int, default=256)
    parser.add_argument('-ac_grad_clip', '--ac_grad_clip', type=float, default=0.5)
    parser.add_argument('-cr_grad_clip', '--cr_grad_clip', type=float, default=1.0)
    parser.add_argument('-b', '--batch', type=int, default=100)
    parser.add_argument('-n', '--noise', type=float, default=1)
    parser.add_argument('-OU_sigma', '--OU_sigma', type=float, default=0.20)
    parser.add_argument('-OU_mu', '--OU_mu', type=float, default=0)
    parser.add_argument('-OU_theta', '--OU_theta', type=float, default=0.15)
    parser.add_argument('-p', '--polyak', type=float, default=0.999)
    parser.add_argument('-tc', '--test_cycles', type=int, default=5000)
    parser.add_argument('-ti', '--test_every_epoch', type=int, default=10)
    parser.add_argument('-eval_epc', '--eval_epc', type=int, default=5)
    parser.add_argument('-eval_steps', '--eval_steps', type=int, default=1000)
    parser.add_argument("--off-render", dest="render", action="store_false", help="turn off rendering")
    parser.add_argument("--save_dir", type=str, default='./models/')

    args = vars(parser.parse_args())
    train(args)