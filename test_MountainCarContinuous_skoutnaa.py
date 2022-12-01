import argparse
import gym
import torch
import numpy as np
import time
import datetime

from ddpg_skoutnaa import DDPG
from reinforce_skoutnaa import REINFORCEAGENT

def setup(args):
    env = gym.make(args['env'])
    agent = None
    if args['algorithm'] == 'ddpg':
        agent = DDPG(env, args)
        agent.q.eval()
        agent.pi.eval()
        agent.q_target.eval()
        agent.pi_target.eval()
        params = torch.load(args['path'])
        actor_weights = params['actor_state_dict']
        critic_weights = params['critic_state_dict']
        agent.pi.load_state_dict(actor_weights)
        agent.q.load_state_dict(critic_weights)
    elif args['algorithm'] == 'reinforce':
        agent = REINFORCEAGENT(env, args['input_dim'], args['output_dim'], args['layers'], args['size'], args['learning_rate'], args['activation'])
        agent.network.eval()
    return env, agent


def test(args):
    env, agent = setup(args)
    t = 0
    dones = 0
    while args['test_steps'] > t and dones is not True:
        state = env.reset()
        done = False
        epoch_return = 0.
        while not done and args['test_steps'] > t:
            if args['render']:
                env.render()
                if args['algorithm'] == 'reinforce':   
                    action = agent.forward(state).sample()
                elif args['algorithm'] == 'ddpg':
                    action = agent.forward(state)
                    action = action.detach().numpy()
                next_state, reward, done, _ = env.step(action)
                epoch_return += reward
                state = next_state
            t += 1
            if done == True:
                dones += 1
            if dones >= 10:
                dones = True
                break
        print(f"epoch reward: {epoch_return}")
                
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algorithm', type=str, help='agent algorithm', default="reinforce")
    parser.add_argument('-env', type=str, default='MountainCarContinuous-v0')
    parser.add_argument('--input_dim', type=int, default=2)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument("--actor_dim", type=list, default=[32, 64])
    parser.add_argument('--critic_dim', type=list, default=[64,64])
    parser.add_argument('-l', '--layers', type=int, default=2)
    parser.add_argument('-s','--size', type=int, default=128)
    parser.add_argument('-lr','--learning_rate', type=float, default=10e-2)
    parser.add_argument('-A_lr','--actor_learning_rate', type=float, default=10e-4)
    parser.add_argument('-C_lr','--critic_learning_rate', type=float, default=10e-4)
    parser.add_argument('-act','--activation', type=str, default='ReLU')
    parser.add_argument('-ep', '--epochs', type=int, default=1000)
    parser.add_argument('-st', '--steps', type=int, default=1000)
    parser.add_argument('-g', '--gamma', type=float, default=0.99)
    parser.add_argument('-buf', '--maxBuffer', type=int, default=50000)
    parser.add_argument('-DDPG_S', '--DDPG_Start', type=int, default=1000)
    parser.add_argument('-DDPG_mean', '--DDPG_mean', type=float, default=0)
    parser.add_argument('-DDPG_std', '--DDPG_std', type=float, default=0.50)
    parser.add_argument('-b', '--batch', type=int, default=64)
    parser.add_argument('-n', '--noise', type=float, default=1)
    parser.add_argument('-tc', '--test_cycles', type=int, default=2000)
    parser.add_argument('-ti', '--test_every_epoch', type=int, default=25)
    parser.add_argument('-eval_epc', '--eval_epc', type=int, default=5)
    parser.add_argument('-eval_steps', '--eval_steps', type=int, default=1000)
    parser.add_argument('-OU_sigma', '--OU_sigma', type=float, default=0.2)
    parser.add_argument('-OU_mu', '--OU_mu', type=float, default=0)
    parser.add_argument('-OU_theta', '--OU_theta', type=float, default=0.15)
    parser.add_argument('-OU_scaletimesteps', '--OU_scaletimesteps', type=int, default=40000)
    parser.add_argument('-OU_final_scale', '--OU_final_scale', type=int, default=0.02)
    parser.add_argument('-polay', '--polyak', type=float, default=0.99)
    parser.add_argument('-ac_grad_clip', '--ac_grad_clip', type=float, default=0.5)
    parser.add_argument('-cr_grad_clip', '--cr_grad_clip', type=float, default=1.0)
    parser.add_argument("--off-render", dest="render", action="store_false", help="turn off rendering")
    parser.add_argument("--save_dir", type=str, default='./models/')
    parser.add_argument('-t', '--test_steps', type=int, default=5000)

    """Please use the combined actor and critic .pt files for the test to work, to run ->
       python test_MountainCarContinuous_skoutnaa.py -a ddpg -p ./mountaincar_ddpg_actor_critic_weights_skoutaa.pt
    """
    parser.add_argument("-p", "--path", type=str, default="./mountaincar_ddpg_actor_critic_weights_skoutaa.pt")

    args = vars(parser.parse_args())
    test(args)