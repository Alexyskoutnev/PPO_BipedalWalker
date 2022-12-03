import argparse
import gym
import torch
import numpy as np
import time
import datetime

from ppo_skoutnaa import PPOAgent

def setup(args):
    env = gym.make(args['env'])
    obs_shape = env.observation_space.shape[0]
    action_shape = env.action_space.shape[0]
    args['input_dim'] = obs_shape
    args['output_dim'] = action_shape
    agent = PPOAgent(env, args, args['input_dim'], args['output_dim'])
    params = torch.load(args['path'])
    actor_weights = params['actor_state_dict']
    critic_weights = params['critic_state_dict']
    agent.pi.load_state_dict(actor_weights)
    agent.critic.load_state_dict(critic_weights)
    agent.pi.eval()
    agent.critic.eval()
    return env, agent

def test(args):
    env, agent = setup(args)
    t = 0
    dones = 0
    for epoch in range(args['eval_epoch']):
        obs = env.reset()
        done = False
        epoch_return = 0.0
        done = False
        steps = 0
        while not done and args['eval_steps'] > steps:
            if args['render']:
                env.render()
            state = torch.tensor(np.expand_dims(obs, axis=0), dtype=torch.float32)
            action, _, _ = agent.forward(state, noise=False)
            next_obs, reward, done, _ = env.step(action.numpy()[0])
            epoch_return += reward
            obs = next_obs
            steps += 1
            if done == True:
                dones += 1
        print(f"test epoch reward: {epoch_return}")
        if dones > 10:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algorithm', type=str, help='agent algorithm', default="ppo")
    ### Please use combined actor/critic weights .pt file to run the agent test -> python test_BipedalWalker_skoutnaa.py -p ./models/bipedalwalker_weights_skoutaa_221202_191507.pt ###
    parser.add_argument('-p', '--path', type=str, default='./models/bipedalwalker_weights_skoutaa_221202_191507.pt')
    parser.add_argument('-env', type=str, default='BipedalWalker-v3')
    parser.add_argument('-act', "--activation", type=str, default="ReLU")
    parser.add_argument('-l', '--layers', type=list, default=[128, 128])
    parser.add_argument('-lr','--learning_rate', type=float, default=10e-3)
    parser.add_argument('-ep', '--epochs', type=int, default=5000)
    parser.add_argument('-st', '--steps', type=int, default=2048)
    parser.add_argument('-g', '--gamma', type=float, default=0.99)
    parser.add_argument('-b', '--batch', type=int, default=64)
    parser.add_argument('-n', '--noise', type=float, default=0.1)
    parser.add_argument('-tc', '--test_cycles', type=int, default=2000)
    parser.add_argument('-ti', '--test_every_epoch', type=int, default=25)
    parser.add_argument('-eval_ep', '--eval_epoch', type=int, default=10)
    parser.add_argument('-eval_st', '--eval_steps', type=int, default=10000)
    parser.add_argument("--off-render", dest="render", action="store_false", help="turn off rendering")
    parser.add_argument("--save_dir", type=str, default='./models/')
    parser.add_argument("-std", "--std", type=float, default=0.1)
    parser.add_argument("-stdD", "--std_decay", type=float, default=0.01)
    parser.add_argument("-stdmin", "--std_min", type=float, default=0.001)
    parser.add_argument("-decay_t", "--decay_timestep", type=int, default=10000)
    parser.add_argument("-entr", "--entropy", type=float, default=0.01)
    parser.add_argument("-k", "--policy_updates", type=int, default=10)
    parser.add_argument("-a_lr", "--actor_lr", type=float, default=10e-4)
    parser.add_argument("-c_lr", "--critic_lr", type=float, default=10e-4)
    parser.add_argument("-clip", "--esp_clip", type=float, default=0.2)
    parser.add_argument("-grad_c", "--grad_clip", type=float, default=1.0)

    args = vars(parser.parse_args())
    test(args)