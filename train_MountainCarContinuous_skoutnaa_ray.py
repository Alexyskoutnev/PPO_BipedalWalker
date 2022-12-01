import argparse
# import gym
import torch
import numpy as np
import time
import os
from tqdm import tqdm
import datetime

from collections import OrderedDict


from ray.rllib.algorithms.ddpg import DDPG
from ray.rllib.algorithms.ddpg.ddpg import DDPGConfig
from ray import air
from ray import tune
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="./runs/MountainCarContinuous_third_party" + time.strftime("%Y_%m_%d_%H_%M"))

# configfile = {
#     # Environment (RLlib understands openAI gym registered strings).
#     "env": "MountainCarContinuous-v0",
#     # Use 2 environment workers (aka "rollout workers") that parallelly
#     # collect samples from their own environment clone(s).
#     "num_workers": 5,
#     # Change this to "framework: torch", if you are using PyTorch.
#     # Also, use "framework: tf2" for tf2.x eager execution.
#     "framework": "torch",
#     # Tweak the default model provided automatically by RLlib,
#     # given the environment's observation- and action spaces.
#     "model": {
#         "fcnet_hiddens": [64, 64],
#         "fcnet_activation": "relu",
#     },
#     # Set up a separate evaluation worker set for the
#     # `algo.evaluate()` call after training (see below).
#     "evaluation_num_workers": 1,
#     "evaluation_duration":10,
#     # Only for evaluation runs, render the env.
#     "evaluation_config": {
#         "render_env": False,
#     },
# }


config_file = {
        # Works for both tor
        # ch and tf.
        "env": "MountainCarContinuous-v0",
        "num_workers": 5,
        'framework': 'torch',
        # === Model ===
        'actor_hiddens': [32, 64],
        'critic_hiddens': [64, 64],
        'n_step': 3,
        'model': {},
        'gamma': 0.99,
        'env_config': {},

        # === Exploration ===
        'exploration_config':{
            'initial_scale': 1.0,
            'final_scale': 0.02,
            'scale_timesteps': 40000,
            'ou_base_scale': 0.75,
            'ou_theta': 0.15,
            'ou_sigma': 0.2
        },

        'min_sample_timesteps_per_iteration': 1000,

        'target_network_update_freq': 0,
        'tau': 0.01,

        # === Replay buffer ===
        'replay_buffer_config':{
          'type': 'MultiAgentPrioritizedReplayBuffer',
          'capacity': 50000,
          'prioritized_replay_alpha': 0.6,
          'prioritized_replay_beta': 0.4,
          'prioritized_replay_eps': 0.000001,
          'worker_side_prioritization': False
        },
        'num_steps_sampled_before_learning_starts': 1000,
        'clip_rewards': False,

        # === Optimization ===
        'actor_lr': 0.001,
        'critic_lr': 0.001,
        'use_huber': False,
        'huber_threshold': 1.0,
        'l2_reg': 0.00001,
        'rollout_fragment_length': 1,
        'train_batch_size': 64,

        # === Parallelism ===
        'num_workers': 4,
        'num_gpus_per_worker': 0,

        # === Evaluation ===
        'evaluation_interval': 5,
        'evaluation_duration': 10,
        "evaluation_config": {
         "render_env": False}
    }


# config_file = DDPGConfig()

def setup(args):
    # env = gym.make("MountainCarContinuous-v0")
    if args['tuning']:
        config = DDPGConfig()
        config = config.training(actor_lr=tune.loguniform(1e-2, 1e-5),critic_lr=tune.loguniform(1e-2, 1e-5), gamma=tune.uniform(0.8,0.999), train_batch_size=tune.choice([32, 64, 128, 256]), tau=tune.grid_search([0.0001, 0.001, 0.01]), actor_hiddens=tune.choice([[32,32], [64,64], [128, 128]]), critic_hiddens=tune.choice([[32,32], [64,64], [128, 128]]))
        config = config.environment(env=args['env'])
        results = tune.Tuner("DDPG", run_config=air.RunConfig(stop={"episode_reward_mean": 90}), param_space=config.to_dict()).fit()
        dfs = {result.log_dir: result.metrics_dataframe for result in results}
        print(dfs)
    else:
        # config_file['render_env'] = args['render']
        agent = DDPG(config=config_file)
        # breakpoint()
        
    return agent


def train(args):
    agent = setup(args)
    training_time = time.time()
    cum_rewards = list()
    average_reward_100 = 0
    successfulTrain = False
    eval_rewards = list()
    eval_rewards_mean = 0
    for epoch in tqdm(range(args['epochs'])):
        i = 0
        params = agent.train()
        epoch_reward = params['sampler_results']['episode_reward_mean']
        writer.add_scalar('training_rewards', epoch_reward, epoch)
        cum_rewards.append(params['episode_reward_mean'])
        print(f"{epoch} epoch_return: {epoch_reward}")

        if epoch % args['test_every_epoch'] == 0 and epoch != 0:
            params_eval = agent.evaluate()
            eval_mean_reward = params_eval['evaluation']['episode_reward_mean']
            print(f"{epoch} eval mean reward: {eval_mean_reward}")
            writer.add_scalar('test_rewards', eval_mean_reward, epoch)
            eval_rewards.append(eval_mean_reward)
            if len(eval_rewards) > 5:
                eval_rewards_mean = sum(eval_rewards[-5:])/5.0
                print(f"eval_rewards_mean: {eval_rewards_mean}")

        if len(cum_rewards) > 90:
            average_reward_100 = np.array(cum_rewards[-100:]).mean()
            print(f"{epoch}: average_100 rewards {average_reward_100}")
            writer.add_scalar('mean_100_rewards/train', average_reward_100, epoch)
        
        if average_reward_100 > 90 and eval_rewards_mean > 90:
            successfulTrain = True
            break
        # successfulTrain = True
        # break

    end_train_time = time.time() - training_time
    if successfulTrain:
        os.makedirs(args['save_dir'], exist_ok=True)
        print("MODEL HAS BEEN SAVE")
        print(f"total train time -> {end_train_time}")
        NOWTIMES = datetime.datetime.now()
        curr_time = NOWTIMES.strftime("%y%m%d_%H%M%S")
        filenameActor = "DDPG_MountainCarContinous_third_party_Actor_" + str(curr_time) + ".pt"
        filenameCritic = "DDPG_MountainCarContinous_third_party_Critic_" + str(curr_time) + ".pt"
        filenameThirdParty = "DDPG_MountainCarContinous_third_party_Combined_" + str(curr_time) + ".pt"
        pathThirdParty = os.path.join(args['save_dir'], filenameThirdParty)
        pathActor = os.path.join(args['save_dir'], filenameActor) 
        pathCritic = os.path.join(args['save_dir'], filenameCritic) 
        checkpoint = agent.get_policy().get_weights()
        paramsActor, paramsCritic, params = save_weights(checkpoint)
        torch.save(paramsActor, pathActor)
        torch.save(paramsCritic, pathCritic)
        torch.save(params, pathThirdParty)
        print(f"third-party model weights path: {pathThirdParty}")
    writer.close()
            
def save_weights(checkpoint):
    paramsActor = OrderedDict()
    paramsActor['fc1.weight'] = torch.tensor(checkpoint['policy_model.action_0._model.0.weight'])
    paramsActor['fc1.bias'] = torch.tensor(checkpoint['policy_model.action_0._model.0.bias'])
    paramsActor['fc2.weight'] = torch.tensor(checkpoint['policy_model.action_1._model.0.weight'])
    paramsActor['fc2.bias'] = torch.tensor(checkpoint['policy_model.action_1._model.0.bias'])
    paramsActor['fc3.weight'] = torch.tensor(checkpoint['policy_model.action_out._model.0.weight'])
    paramsActor['fc3.bias'] = torch.tensor(checkpoint['policy_model.action_out._model.0.bias'])
    paramsCritic = OrderedDict()
    paramsCritic['fc1.weight'] = torch.tensor(checkpoint['q_model.q_hidden_0._model.0.weight'])
    paramsCritic['fc1.bias'] = torch.tensor(checkpoint['q_model.q_hidden_0._model.0.bias'])
    paramsCritic['fc2.weight'] = torch.tensor(checkpoint['q_model.q_hidden_1._model.0.weight'])
    paramsCritic['fc2.bias'] = torch.tensor(checkpoint['q_model.q_hidden_1._model.0.bias'])
    paramsCritic['fc3.weight'] = torch.tensor(checkpoint['q_model.q_out._model.0.weight'])
    paramsCritic['fc3.bias'] = torch.tensor(checkpoint['q_model.q_out._model.0.bias'])
    params = {
         "actor_state_dict" : paramsActor,
         "critic_state_dict" : paramsCritic
    }
    return paramsActor, paramsCritic, params
       







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algorithm', type=str, help='agent algorithm', default="reinforce")
    parser.add_argument('-env', type=str, default='MountainCarContinuous-v0')
    parser.add_argument('--input_dim', type=int, default=2)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('-l', '--layers', type=int, default=2)
    parser.add_argument('-s','--size', type=int, default=128)
    parser.add_argument('-lr','--learning_rate', type=float, default=10e-2)
    parser.add_argument('-A_lr','--actor_learning_rate', type=float, default=10e-4)
    parser.add_argument('-C_lr','--critic_learning_rate', type=float, default=10e-5)
    parser.add_argument('-act','--activation', type=str, default='ReLU')
    parser.add_argument('-ep', '--epochs', type=int, default=500)
    parser.add_argument('-st', '--steps', type=int, default=1000)
    parser.add_argument('-g', '--gamma', type=float, default=0.99)
    parser.add_argument('-buf', '--maxBuffer', type=int, default=10e6)
    parser.add_argument('-DDPG_S', '--DDPG_Start', type=int, default=50000)
    parser.add_argument('-DDPG_mean', '--DDPG_mean', type=float, default=0)
    parser.add_argument('-DDPG_std', '--DDPG_std', type=float, default=0.50)
    parser.add_argument('-DDPG_a_hs1', '--DDPG_a_hs1', type=int, default=400)
    parser.add_argument('-DDPG_a_hs2', '--DDPG_a_hs2', type=int, default=400)
    parser.add_argument('-DDPG_c_hs1', '--DDPG_c_hs1', type=int, default=400)
    parser.add_argument('-DDPG_c_hs2', '--DDPG_c_hs2', type=int, default=300)
    parser.add_argument('-b', '--batch', type=int, default=32)
    parser.add_argument('-n', '--noise', type=float, default=1)
    parser.add_argument('-tc', '--test_cycles', type=int, default=2000)
    parser.add_argument('-ti', '--test_every_epoch', type=int, default=1)
    parser.add_argument('-eval_epc', '--eval_epc', type=int, default=5)
    parser.add_argument('-eval_steps', '--eval_steps', type=int, default=1000)
    parser.add_argument("--tuning_on", dest="tuning", action="store_true", help="turn on tuning")
    parser.add_argument("--off-render", dest="render", action="store_true", help="turn off rendering")
    parser.add_argument("--save_dir", type=str, default='./models/')

    args = vars(parser.parse_args())
    train(args)