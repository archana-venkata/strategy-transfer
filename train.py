
import gym
import gym_envs
import pandas as pd

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from wrappers.reward_shaping import RewardWrapper1, RewardWrapper2, RewardWrapper3, RewardWrapper4

class EvalCallback(BaseCallback):
    def __init__(self, env: gym.Env, eval_env: gym.Env, eval_interval: int = 10000, n_eval_episodes: int = 1000):
        super().__init__()
        self._env = env
        self._eval_env = eval_env
        self._eval_interval = eval_interval
        self._n_eval_episodes = n_eval_episodes
        self._mean_episode_scores = []

    def _on_step(self) -> bool:
        # Periodically evaluate the model
        if self.num_timesteps % self._eval_interval == 0:
            # Calculate the average score over a fixed number of episodes
            mean_episode_score = 0

            for n in range(0, self._n_eval_episodes):
                observation = self._eval_env.reset()

                # render the environment at the beginning of each episode
                done = False
                score = 0

                # play a certain number of moves for each episode
                while not done:
                    action, _ = self.model.predict(
                        observation, action_masks=self.env.valid_action_mask())
                    observation, reward, done, info = self._eval_env.step(
                        action)
                    score += reward

                mean_episode_score += score

            mean_episode_score /= self._n_eval_episodes

            self.logger.record("mean_episode_score", mean_episode_score)
            self._mean_episode_scores.append(mean_episode_score)

        return True
    
def train(env_name, transferred_strategies, seed, log_dir, max_total_step_num=1e10):
    
    # read json file with transferred strategies
    strategies_df = pd.read_json('strategies.json')
    env_strategies = strategies_df[env_name].to_dict()

    transferred_strategies = env_strategies['strategies']
    
    # Define main and evaluation environments
    env = gym.make(env_name, config_file="config_train.json")
    # Reward shaping with R_{action}
    env = RewardWrapper1(env, transferred_strategies)
    eval_env = gym.make(env_name, config_file="config_test.json")
    
    # Wrap to enable masking
    env = ActionMasker(env, env.valid_action_mask())  
    eval_env = ActionMasker(eval_env, env.valid_action_mask())
    
    env.seed(seed)  # train on the same environment
    eval_env.seed(seed)  # evaluate on the same environment
    set_random_seed(seed)  # stable_baselines3 seed function
    
    # Define model
    model = MaskablePPO(MaskableActorCriticPolicy, env,
                        verbose=1, tensorboard_log=f"{log_dir}")
    model_save_path = "./Saved Models/PPO_Model_" + env_name

    callback = EvalCallback(env, eval_env)
    
    # Train
    model.learn(max_total_step_num, callback=callback)

    # Save model
    model.save(model_save_path)
    