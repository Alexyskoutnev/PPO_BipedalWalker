import argparse
import gym
import torch
import numpy as np
import time
import datetime
import os
from tqdm import tqdm
from ddpg_skoutnaa import DDPG
from utils import Path, ReplayBuffer
import time
from torch.utils.tensorboard import SummaryWriter

#third party modules
from rl_algorithms import build_agent
import rl_algorithms.common.env.utils as env_utils
import rl_algorithms.common.helper_functions as common_utils
from rl_algorithms.utils import YamlConfig
from rl_algorithms.common.helper_functions import numpy2floattensor




STEPS = 1000
writer = SummaryWriter(log_dir="./runs/lunarlander_third_party_" + time.strftime("%Y_%m_%d_%H_%M"))
layout = {"mean-100_and_train_rewards": {
        "rewards": ["Multiline", ["mean_training_100", "training_rewards"]],
    },}
writer.add_custom_scalars(layout)

def setup(args):
    # env_name = "LunarLanderContinuous-v2"
    env = gym.make(args.env)
    env, max_episode_steps = env_utils.set_env(env, args.max_episode_steps)
    common_utils.set_random_seed(args.seed, env)
    NOWTIMES = datetime.datetime.now()
    curr_time = NOWTIMES.strftime("%y%m%d_%H%M%S")
    cfg = YamlConfig(dict(agent=args.cfg_path)).get_config_dict()
    env_info = dict(
        name=env.spec.id,
        observation_space=env.observation_space,
        action_space=env.action_space,
        is_atari=False,
    )
    log_cfg = dict(agent=cfg.agent.type, curr_time=curr_time, cfg_path=args.cfg_path)
    build_args = dict(
        env=env,
        env_info=env_info,
        log_cfg=log_cfg,
        is_test=args.test,
        load_from=args.load_from,
        is_render=args.render,
        render_after=args.render_after,
        is_log=args.log,
        save_period=args.save_period,
        episode_num=args.episode_num,
        max_episode_steps=max_episode_steps,
        interim_test_num=args.interim_test_num,
    )
    agent = build_agent(cfg.agent, build_args)
    return env, agent


def train(args):
    env, agent = setup(args)
    training_time = time.time()
    cum_rewards = list()
    average_reward_100 = 0
    successfulTrain = False
    eval_rewards = list()
    eval_rewards_mean = 0

    for epoch in tqdm(range(args.epochs)):
        state = env.reset()
        done = False
        epoch_return = 0
        agent.episode_step = 0
        losses = list()
        t_begin = time.time()
        path = Path()
        while not done:
            if agent.is_render and agent.i_episode >= agent.render_after:
                env.render()
            action = agent.select_action(state)
            next_state, reward, done, _ = agent.step(action)
            path.add(state, action, next_state, reward, done)
            agent.total_step += 1
            agent.episode_step += 1
            epoch_return += reward
            state = next_state
            if len(agent.memory) >= args.batch:
                for _ in range(agent.hyper_params.multiple_update):
                    experience = agent.memory.sample()
                    experience = numpy2floattensor(experience, agent.learner.device)
                    loss_pi, loss_q = agent.learner.update_model(experience)
                writer.add_scalar('loss_q/train', loss_q, epoch)
                writer.add_scalar('loss_pi/train', loss_pi, epoch)
        print(f"{epoch} epoch_return: {epoch_return}")
        print(f"{epoch} loss_q: {loss_q}, loss_pi {loss_pi}")
        t_end = time.time()
        avg_time_cost = (t_end - t_begin) / agent.episode_step
        writer.add_scalar('training_rewards', epoch_return, epoch)
        cum_rewards.append(epoch_return)

        if epoch % args.test_every_epoch == 0 and epoch != 0:
            eval_reward = eval(env, agent, args)
            writer.add_scalar('test_rewards', eval_reward, epoch)
            eval_rewards.append(eval_reward)
            if len(eval_rewards) > 5:
                eval_rewards_mean = sum(eval_rewards[-5:])/5.0
                print(f"eval_rewards_mean: {eval_rewards_mean}")

        if len(cum_rewards) > 100:
            average_reward_100 = np.array(cum_rewards[-100:]).mean()
            print(f"{_}: average_100 rewards {average_reward_100}")
            writer.add_scalar('mean_100_rewards/train', average_reward_100, epoch)
            # writer.add_scalar('rewards/merged', {'mean_100': average_reward_100, "training_rewards": np.array([epoch_return])}, epoch)
        if average_reward_100 > 200 and eval_rewards_mean > 190:
            successfulTrain = True
            break
    
    end_train_time = time.time() - training_time
    if successfulTrain:
        os.makedirs(args.save_dir, exist_ok=True)
        print("MODEL HAS BEEN SAVE")
        print(f"total train time -> {end_train_time}")
        NOWTIMES = datetime.datetime.now()
        curr_time = NOWTIMES.strftime("%y%m%d_%H%M%S")
        filenameActor = "./models/DDPG_lunarlander_actor-" + str(curr_time) + ".pt"
        filenameCritic = "./models/DDPG_lunarlander_critic-" + str(curr_time) + ".pt"
        filenameThirdParty = "DDPG_lunarlander_third_party_save_" + "e_" + str(agent.episode_step) + "_" + str(curr_time) + ".pt"
        pathThirdParty = os.path.join(args.save_dir, filenameThirdParty)
        params = save_weights(agent)
        f_actor = open(filenameActor, "w+")
        f_critic  = open(filenameCritic, "w+")
        torch.save(agent.learner.actor.state_dict(), filenameActor)
        torch.save(agent.learner.critic.state_dict(), filenameCritic)
        torch.save(params, pathThirdParty)
        print(f"third-party model weights path: {pathThirdParty}")
    env.close()
    writer.close()

def save_weights(agent):
    params = {
            "actor_state_dict": agent.learner.actor.state_dict(),
            "actor_target_state_dict": agent.learner.actor_target.state_dict(),
            "critic_state_dict": agent.learner.critic.state_dict(),
            "critic_target_state_dict": agent.learner.critic_target.state_dict(),
            "actor_optim_state_dict": agent.learner.actor_optim.state_dict(),
            "critic_optim_state_dict": agent.learner.critic_optim.state_dict(),
    }
    return params

def eval(env, agent, args):
    result = []
    steps = 0
    dones = 0
    for epoch in range(args.eval_epc):
        state = env.reset()
        done = False
        epoch_return = 0.
        done = False
        while not done and args.eval_steps > steps:
            if agent.is_render:
                env.render()
            action = agent.select_action(state)
            next_state, reward, done, _ = agent.step(action)
            epoch_return += reward
            state = next_state
            steps += 1
            if done == True:
                dones += 1
        result.append(epoch_return)
        if dones >= 10:
            break
    result = np.array(result).reshape(-1,1)
    print(f"!!!!!!!EVAL Mean ->>{np.mean(result)}")
    return np.mean(result)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algorithm', type=str, help='agent algorithm', default="ddpg")
    parser.add_argument('-env', type=str, default='LunarLanderContinuous-v2')
    parser.add_argument('-ep', '--epochs', type=int, default=1000)
    parser.add_argument('-b', '--batch', type=int, default=64)
    parser.add_argument('-tc', '--test_cycles', type=int, default=5000)
    parser.add_argument('-ti', '--test_every_epoch', type=int, default=10)
    parser.add_argument('-eval_epc', '--eval_epc', type=int, default=5)
    parser.add_argument('-eval_steps', '--eval_steps', type=int, default=1000)
    parser.add_argument('-seed', '--seed', type=int, default=1)
    parser.add_argument("--cfg-path", type=str, default="./rl_algorithms/configs/lunarlander_continuous_v2/ddpg.yaml",help="config path")
    parser.add_argument("--test", dest="test", action="store_true", help="test mode (no training)")
    parser.add_argument("--load-from", type=str, default=None, help="load the saved model and optimizer at the beginning")
    parser.add_argument("--off-render", dest="render", action="store_false", help="turn off rendering")
    parser.add_argument( "--render-after", type=int, default=10, help="start rendering after the input number of episode")
    parser.add_argument("--log", dest="log", action="store_true", help="turn on logging")
    parser.add_argument("--save-period", type=int, default=100, help="save model period")
    parser.add_argument("--episode-num", type=int, default=1500, help="total episode num")
    parser.add_argument("--max-episode-steps", type=int, default=10000, help="max episode step")
    parser.add_argument("--interim-test-num", type=int, default=10,)
    parser.add_argument("--save_dir", type=str, default='./models/')

    args = parser.parse_args()
    train(args)