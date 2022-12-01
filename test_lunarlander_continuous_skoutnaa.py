import argparse
import gym
import torch
import numpy as np
import time
import datetime


from ddpg_skoutnaa import DDPG


"""third party that require a different version of python, gym, etc to run (not compatible with Tune)
    uncomment below once different conda enviroment is switched
"""
# from rl_algorithms import build_agent
# import rl_algorithms.common.env.utils as env_utils
# import rl_algorithms.common.helper_functions as common_utils
# from rl_algorithms.utils import YamlConfig
# from rl_algorithms.common.helper_functions import numpy2floattensor

def setup(args):
    env = gym.make(args['env'])
    agent = None
    if args['third_party']:
        """YOU have to use a different conda enviroment to run thirdparty agent that solves this enviroment, once done
            uncomment the lines of code below,
            python test_lunarlander_continuous_skoutnaa.py -p ./models/DDPG_lunarlander_third_party_save_e_223_221112_223245.pt
            will run the well performing agent
        """
        # env, max_episode_steps = env_utils.set_env(env, args['max_episode_steps'])
        # common_utils.set_random_seed(args['seed'], env)
        # NOWTIMES = datetime.datetime.now()
        # curr_time = NOWTIMES.strftime("%y%m%d_%H%M%S")
        # cfg = YamlConfig(dict(agent=args['cfg_path'])).get_config_dict()
        # env_info = dict(
        #     name=env.spec.id,
        #     observation_space=env.observation_space,
        #     action_space=env.action_space,
        #     is_atari=False,
        # )
        # log_cfg = dict(agent=cfg.agent.type, curr_time=curr_time, cfg_path=args['cfg_path'])
        # build_args = dict(
        #     env=env,
        #     env_info=env_info,
        #     log_cfg=log_cfg,
        #     is_test=args['test'],
        #     load_from=args['path'],
        #     is_render=args['render'],
        #     render_after=args['render_after'],
        #     is_log=args['log'],
        #     save_period=args['save_period'],
        #     episode_num=1000,
        #     max_episode_steps=max_episode_steps,
        #     interim_test_num=args['interim_test_num'],
        # )
        # agent = build_agent(cfg.agent, build_args)
        # agent.learner.load_params(args['path'])
        pass
    elif not args['third_party']:
        obs_shape = env.observation_space
        args['input_dim'] = obs_shape.shape[0]
        action_space = env.action_space
        args['output_dim'] = action_space.shape[0]
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
    return env, agent


def test(args):
    env, agent = setup(args)
    t = 0
    dones = 0
    while args['test_steps'] > t and dones is not True:
        state = env.reset()
        done = False
        epoch_return = 0.
        done = False
        while not done and args['test_steps'] > t:
            if args['render']:
                env.render()
            if args['third_party']:
                action = agent.select_action(state)
                next_state, reward, done, _ = agent.step(action)
                epoch_return += reward
                state = next_state
            else:
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
    parser.add_argument('-a', '--algorithm', type=str, help='agent algorithm', default="ddpg")
    parser.add_argument('-env', type=str, default='LunarLanderContinuous-v2')
    parser.add_argument("--cfg-path", type=str, default="./rl_algorithms/configs/lunarlander_continuous_v2/ddpg.yaml",help="config path")
    parser.add_argument("--third_party", dest="third_party", action="store_true", help="use third party DDPG agent")

    """Please use the combined critic, actor weights when running the test such ./models/DDPG_lunarlander_critic_actor_skoutaa_221113_213622.pt or ./lunarlander_continuous_skoutaa_combined.pt (if have third party agent installed)"""
    parser.add_argument("-p", "--path", type=str, default="./models/DDPG_lunarlander_critic_actor_skoutaa_221113_213622.pt")

    parser.add_argument("--test", dest="test", action="store_false", help="test mode (no training)")
    parser.add_argument("--load-from", type=str, default=None, help="load the saved model and optimizer at the beginning")
    parser.add_argument("--off-render", dest="render", action="store_false", help="turn off rendering")
    parser.add_argument( "--render-after", type=int, default=0, help="start rendering after the input number of episode")
    parser.add_argument("--log", dest="log", action="store_true", help="turn on logging")
    parser.add_argument("--save-period", type=int, default=100, help="save model period")
    parser.add_argument("--max-episode-steps", type=int, default=1000, help="max episode step")
    parser.add_argument("--interim-test-num", type=int, default=10,)
    parser.add_argument('-seed', '--seed', type=int, default=1)
    parser.add_argument('-t', '--test_steps', type=int, default=5000)
    parser.add_argument('--input_dim', type=int, default=2)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('-l', '--layers', type=int, default=2)
    parser.add_argument('-s','--size', type=int, default=128)
    parser.add_argument('-lr','--learning_rate', type=float, default=10e-2)
    parser.add_argument('-A_lr','--actor_learning_rate', type=float, default=0.003)
    parser.add_argument('-C_lr','--critic_learning_rate', type=float, default=0.003)
    parser.add_argument('-act','--activation', type=str, default='ReLU')
    parser.add_argument('-ep', '--epochs', type=int, default=1000)
    parser.add_argument('-st', '--steps', type=int, default=1000)
    parser.add_argument('-g', '--gamma', type=float, default=0.99)
    parser.add_argument('-ta', '--tau', type=float, default=0.005)
    parser.add_argument('-polay', '--polyak', type=float, default=0.999)
    parser.add_argument('-buf', '--maxBuffer', type=int, default=50000)
    parser.add_argument('-ac_grad_clip', '--ac_grad_clip', type=float, default=0.5)
    parser.add_argument('-cr_grad_clip', '--cr_grad_clip', type=float, default=1.0)
    parser.add_argument('-b', '--batch', type=int, default=100)
    parser.add_argument("--actor_dim", type=list, default=[256, 256])
    parser.add_argument('--critic_dim', type=list, default=[256,256])
    args = vars(parser.parse_args())
    test(args)