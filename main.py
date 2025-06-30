

import os

import time
import gymnasium as gym
import gym_envs
import numpy as np
import pandas as pd
import logging
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList, StopTrainingOnMaxEpisodes
from stable_baselines3.common.utils import set_random_seed

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from gym_envs.wrappers.reward_shaping import RewardWrapper1, RewardWrapper2, RewardWrapper3

from util.RuntimeLogger import RuntimeLogger
from util.common import save_params
import torch


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve',
                            description='Training RL models with reward shaping for strategy transfer')
    parser.add_argument('--exp_id',
                        default='test',
                        help='Experiment identifier (used for saving and logging data)',
                        required=True)
    parser.add_argument('--env',
                        default='SimplePacman-v0',
                        choices=['SimplePacman-v0',
                                 'DungeonCrawler-v0',
                                 'DungeonCrawler-v1',
                                 'SimpleBankHeist-v0',
                                 'SimpleMinecraft-v0'],
                        help='Name of the environment',
                        required=True)
    parser.add_argument('--env_map',
                        default='map.txt',
                        choices=['map.txt',
                                 'map_simple.txt'],
                        help='Configuration file for the environment',
                        required=False)
    parser.add_argument('--shaping',
                        type=int,
                        default=None,
                        choices=[1, 2, 3, 4],
                        help='Reward shaping method')
    parser.add_argument('--stop_shaping',
                        '-stop',
                        action="store_true",
                        help='Number of timesteps after which to stop reward shaping',
                        required=False)
    parser.add_argument('--stop_shaping_n',
                        type=int,
                        help='Number of timesteps after which to stop applying reward shaping',
                        required=False)
    parser.add_argument('--timesteps',
                        type=int,
                        default=1000000,
                        choices=[1000000, 2000000, 3000000],
                        help='Number of training steps')
    parser.add_argument('--decay',
                        '-d',
                        action="store_true",
                        help='Flag to apply decay to the reward shaping (applies to methods 1 and 4)')
    parser.add_argument('--decay_param',
                        type=float,
                        help='Parameter to control the decay rate of the reward shaping',
                        required=False)
    parser.add_argument('--decay_n',
                        type=int,
                        help='Number of timesteps after which decay should start',
                        required=False)

    parser.add_argument('--seed',
                        type=int,
                        default=None,
                        help='Random seed')

    args = parser.parse_args()

    # Argument error handling
    if (not args.decay) and args.decay_n:
        parser.error('--decay_n can only be set when using flag --decay.')
    elif (not args.decay) and args.decay_param:
        parser.error('--decay_param can only be set when using flag --decay.')
    elif (not args.shaping) and args.stop_shaping:
        parser.error(
            'The flag --stop_shaping can only be used if a shaping method is selected with --shaping.')
    elif (not args.stop_shaping) and args.stop_shaping_n:
        parser.error(
            '--stop_shaping_n can only be set when using flag --stop_shaping.')

    return args


def mask_fn(env: gym.Env) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.unwrapped.valid_action_mask()


class TensorboardCallback(BaseCallback):
    def __init__(self, env: gym.Env, eval_env: gym.Env, seed: int = 0, eval_interval: int = 10000, n_eval_episodes: int = 1000, deterministic: bool = True, stop_shaping: bool = False, decay: bool = False):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard
        """
        super().__init__()
        self._env = env
        self._eval_env = eval_env
        self._seed = seed
        self._eval_interval = eval_interval
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic
        self._mean_episode_rewards = []
        self._decay = decay
        self._stop_shaping = stop_shaping
        # keep track of the mean_episode_reward (mes) obtained after each evaluation period
        # revert to a previous saved model if the new mes is worse.
        self._prev_mes = None
        self._last_saved_model = None

    def _on_step(self) -> bool:
        # Periodically evaluate the model
        if self.num_timesteps % self._eval_interval == 0:
            # Calculate the average reward over a fixed number of episodes
            mean_episode_reward = 0

            for n in range(0, self._n_eval_episodes):
                observation, info = self._eval_env.reset(
                    seed=self._seed, options={})

                # render the environment at the beginning of each episode
                done = False
                score = 0

                # play a certain number of moves for each episode
                while not done:
                    action, _ = self.model.predict(
                        observation, action_masks=mask_fn(self._eval_env))

                    observation, reward, terminated, truncated, info = self._eval_env.step(
                        action)

                    done = terminated or truncated

                    score += reward

                mean_episode_reward += score

            mean_episode_reward /= self._n_eval_episodes

            # # ############## FOR SMOOTHING ################

            # if self._prev_mes == None or mean_episode_reward >= self._prev_mes:
            #     self._prev_mes = mean_episode_reward
            #     self._last_saved_model = self.model
            # elif mean_episode_reward < (self._prev_mes*0.5):
            #     # If the MES is less than the previous evaluation, revert to the previous model
            #     # and plot the previous score
            #     self.model = self._last_saved_model
            #     mean_episode_reward = self._prev_mes

            # # ############################################

            self.logger.record("mean_episode_reward", mean_episode_reward)

            if self._stop_shaping or self._decay:
                # check if decay_n or shaping_n has already been set
                if (self._stop_shaping and self._env.stop_shaping_n != None) or (self._decay and self._env.decay_n != None):
                    print("No shaping stop or decay start calculation needed")
                    self._decay = False
                    self._stop_shaping = False
                else:
                    # if not, calculate when to stop shaping or start decay
                    threshold = 0
                    plateau_start = self.calc_slope(
                        self._mean_episode_rewards, threshold)
                    if plateau_start and self._stop_shaping:
                        self._env.set_stop_shaping_n(self.num_timesteps)
                        self._stop_shaping = False
                    elif plateau_start and self._decay:
                        self._env.set_decay_n(self.num_timesteps)
                        self.decay = False

            self._mean_episode_rewards.append(mean_episode_reward)

        return True

    def calc_slope(self, ydata, threshold=1):
        if len(ydata) >= 10:
            sum = 0
            for j in range(-10, -1, -1):
                sum += abs(ydata[j] - ydata[-1])
            av_diff = sum/10
            if av_diff < threshold:
                return True

        return False


def train(env, eval_env, env_name, seed, log_dir, stop_shaping=False, decay=False, max_episodes=10, max_total_step_num=1e10):
    starttime = time.time()

    # Check if GPU is available else use CPU
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # If existing model at location, load model else
    # define model
    model_save_path = f"{log_dir}/PPO_Model_" + env_name
    if os.path.isfile(model_save_path+".zip"):
        print("Loading existing model...")
        model = MaskablePPO.load(model_save_path, env=env, device=device)
    else:
        print("No existing model found. Creating a new model...")

        model = MaskablePPO(MaskableActorCriticPolicy, env,
                            tensorboard_log=f"{log_dir}", device=device)

    callback = TensorboardCallback(
        env, eval_env, seed=seed, stop_shaping=stop_shaping, decay=decay)
    model.learn(max_total_step_num, callback=callback)

    model.save(model_save_path)

    dt = time.time()-starttime
    print("Calculation took %g hr %g min %g s" %
          (dt//3600, (dt//60) % 60, dt % 60))


def init(env_name, map_file, transferred_strategies, log_dir, stop_shaping=False, stop_shaping_n=False, reward_wrapper=None, decay=False, decay_param=None, decay_n=None, seed=None):
    if seed is None:
        seed = 100

    # Instantiate the environment using training configuration settings (if required)
    if env_name == "SimpleMinecraft-v0":
        env = gym.make(env_name)
    else:
        env = gym.make(env_name, map_file=map_file,
                       config_file="config_train.json")

    check_env(env.unwrapped)
    if (reward_wrapper is not None):
        if decay == True:
            if reward_wrapper == 1:
                env = RewardWrapper1(
                    env, transferred_strategies, decay=decay, decay_param=decay_param, decay_n=decay_n)
            elif reward_wrapper == 2:
                env = RewardWrapper2(
                    env, transferred_strategies, decay=decay, decay_param=decay_param, decay_n=decay_n)
            elif reward_wrapper == 3:
                env = RewardWrapper3(
                    env, transferred_strategies, decay=decay, decay_param=decay_param, decay_n=decay_n)
        elif stop_shaping == True:
            if reward_wrapper == 1:
                env = RewardWrapper1(
                    env, transferred_strategies, stop_shaping_n=stop_shaping_n)
            elif reward_wrapper == 2:
                env = RewardWrapper2(
                    env, transferred_strategies, stop_shaping_n=stop_shaping_n)
            elif reward_wrapper == 3:
                env = RewardWrapper3(
                    env, transferred_strategies, stop_shaping_n=stop_shaping_n)
        else:
            if reward_wrapper == 1:
                env = RewardWrapper1(
                    env, transferred_strategies)
            elif reward_wrapper == 2:
                env = RewardWrapper2(env, transferred_strategies)
            elif reward_wrapper == 3:
                env = RewardWrapper3(env, transferred_strategies)

    env = ActionMasker(env, mask_fn)  # Wrap to enable masking

    env = Monitor(env, log_dir)

    # Define an evaluation environment using testing configuration settings (if required)
    if env_name == "SimpleMinecraft-v0":
        eval_env = gym.make(env_name)
    else:
        eval_env = gym.make(env_name, map_file=map_file,
                            config_file="config_test.json")

    eval_env = ActionMasker(eval_env, mask_fn)  # Wrap to enable masking

    if seed != None:
        # train on the same environment
        env.reset(seed=seed, options={})
        # evaluate on the same environment
        eval_env.reset(seed=seed, options={})
        set_random_seed(seed)  # stable_baselines3 seed function
    return env, eval_env


def main(args):

    # read json file with transferred strategies
    strategies_df = pd.read_json('strategies.json')
    env_strategies = strategies_df[args.env].to_dict()

    transferred_strategies = env_strategies['strategies']

    # Log directory
    data_dir = f'data/{args.exp_id}'
    log_dir = f'{data_dir}/log_{args.seed}'
    os.makedirs(os.path.dirname(data_dir), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    save_params(args, log_dir)  # save the experiment parameters in a text file

    logger = RuntimeLogger(data_dir)
    logger.start()

    if args.decay == True:
        env, eval_env = init(env_name=args.env,
                             map_file=args.env_map,
                             transferred_strategies=transferred_strategies,
                             log_dir=log_dir,
                             reward_wrapper=args.shaping,
                             decay=args.decay,
                             decay_n=args.decay_n,
                             decay_param=args.decay_param,
                             seed=args.seed
                             )
    elif args.stop_shaping == True:
        env, eval_env = init(env_name=args.env,
                             map_file=args.env_map,
                             transferred_strategies=transferred_strategies,
                             stop_shaping=args.stop_shaping,
                             stop_shaping_n=args.stop_shaping_n,
                             log_dir=log_dir,
                             reward_wrapper=args.shaping,
                             seed=args.seed
                             )
    else:
        env, eval_env = init(env_name=args.env,
                             map_file=args.env_map,
                             transferred_strategies=transferred_strategies,
                             log_dir=log_dir,
                             reward_wrapper=args.shaping,
                             seed=args.seed
                             )

    train(env=env,
          eval_env=eval_env,
          env_name=args.env,
          seed=args.seed,
          max_total_step_num=args.timesteps,
          log_dir=log_dir,
          stop_shaping=args.stop_shaping,
          decay=args.decay)

    logger.log_experiment_end()


def more_main():
    main(parse_args())


def debug_main():
    test_args = {'exp_id': "1-1.debug",
                 'env': 'SimplePacman-v0',
                 'timesteps': 1000000,
                 'env_map': "map.txt",
                 'shaping': 1,
                 'decay': False,
                 'decay_param': 0.8,
                 'decay_n': 0,
                 'seed': 12
                 }
    main(dotdict(test_args))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    more_main()
    # debug_main()
