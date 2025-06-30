import gymnasium as gym
from util.common import is_subsequence


class RewardWrapper(gym.Wrapper):
    def __init__(self, env, strategies, stop_shaping_n=None, decay=False, decay_param=None, decay_n=None):
        super().__init__(env)
        self.strategies = strategies
        self.step_count = 0
        self.trajectory_reward = 0

        # If the additional reward should stop being included after some amount of time
        # This may be specified by the user otherwise it will be determined as the model learns
        self.stop_shaping_n = stop_shaping_n

        # This will be set to True when the number of timesteps exceeds stop_shaping_n
        self.stop_shaping = False

        # If the additional reward will decay over time,
        # specify the decay rate
        if decay:
            self.decay_param = decay_param

        # After how many timesteps will decay occur
        # This may be specified by the user otherwise it will be determined as the model learns
        # If 0: the decay starts at 0 timesteps
        # By default: None
        self.decay_n = decay_n

        # This will be set to True when the number of timesteps exceeds decay_n
        self.decay = False

    def reset(self, **kwargs):
        self.obs, info = self.env.reset(**kwargs)
        self.action_history = []
        self.trajectory_reward = 0
        return self.obs, info

    def set_decay_n(self, decay_n):
        if self.decay_n is not None:
            raise (ValueError(
                'Attempted to override decay_n when it has already been set.'))
        elif self.decay:
            raise (ValueError(
                'Attempted to override decay_n when the decay option was not set.'))
        else:
            self.decay_n = decay_n

    def set_stop_shaping_n(self, stop_shaping_n):
        if self.stop_shaping_n is not None:
            raise (ValueError(
                'Attempted to override stop_shaping_n when it has already been set.'))
        elif self.stop_shaping:
            raise (ValueError(
                'Attempted to override stop_shaping_n when the decay option was not set.'))
        else:
            self.stop_shaping_n = stop_shaping_n

# Additional reward = % of the current action reward
# If DECAY, Decay reward shaping at DECAY_PARAM rate after DECAY_N timesteps


class RewardWrapper1(RewardWrapper):

    def step(self, action):
        self.step_count += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        new_reward = reward
        for i in info["result_of_action"]:
            self.action_history.append(i)

        # start to decay the reward shaping after some threshold number of steps
        if self.decay_n is not None and self.step_count > self.decay_n:
            self.decay = True

        if self.stop_shaping_n is not None and self.step_count > self.stop_shaping_n:
            self.stop_shaping = True

        # if the strategy sequence is part of the current trajectory
        for strategy in self.strategies:
            # compute how much of the strategy is being followed
            x, y = is_subsequence(strategy['strategy'], self.action_history)
            # If the agent's trajectory indicates that one (or more) strategies have been partially followed
            if x:
                # If the option to start decaying the reward shaping has been set and decay_n timesteps has elapsed
                # or if the option to stop reward shaping has not been set, apply an additional reward
                if self.decay:
                    if self.decay_param != 0:
                        # Determine the value of the additional reward based on the specified decay rate
                        new_reward += (y*reward*self.decay_param)
                elif not self.stop_shaping:
                    # additional reward is equal to a percentage of the reward for the current action
                    new_reward += (y*reward)

        if self.decay:
            self.decay_param *= self.decay_param

        return obs, new_reward, terminated, truncated, info

# Additional reward = % of the goal reward


class RewardWrapper2(RewardWrapper):

    def step(self, action):
        self.step_count += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        new_reward = reward
        for i in info["result_of_action"]:
            self.action_history.append(i)

        # start to decay the reward shaping after some threshold number of steps
        if self.decay_n is not None and self.step_count > self.decay_n:
            self.decay = True

        if self.stop_shaping_n is not None and self.step_count > self.stop_shaping_n:
            self.stop_shaping = True

        # if the strategy sequence is part of the current trajectory
        for strategy in self.strategies:
            # compute how much of the strategy is being followed
            x, y = is_subsequence(strategy['strategy'], self.action_history)
            # If the agent's trajectory indicates that one (or more) strategies have been partially followed
            if x:
                # If the option to start decaying the reward shaping has been set and decay_n timesteps has elapsed
                # or if the option to stop reward shaping has not been set, apply an additional reward
                if self.decay:
                    if self.decay_param != 0:
                        # Determine the value of the additional reward based on the specified decay rate
                        new_reward += (y*strategy['reward']*self.decay_param)
                elif not self.stop_shaping:
                    # reward is influenced by how much of the strategy is being followed
                    # additional reward is equal to a percentage of the goal reward for the strategy
                    new_reward += (y*strategy['reward'])

        if self.decay:
            self.decay_param *= self.decay_param

        return obs, new_reward, terminated, truncated, info


# Additional reward = % of the current trajectory reward
class RewardWrapper3(RewardWrapper):

    def step(self, action):
        self.step_count += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        new_reward = reward
        self.trajectory_reward += reward
        for i in info["result_of_action"]:
            self.action_history.append(i)

        # start to decay the reward shaping after some threshold number of steps
        if self.decay_n is not None and self.step_count > self.decay_n:
            self.decay = True

        if self.stop_shaping_n is not None and self.step_count > self.stop_shaping_n:
            self.stop_shaping = True

        # if the strategy sequence is part of the current trajectory
        for strategy in self.strategies:
            # compute how much of the strategy is being followed
            x, y = is_subsequence(strategy['strategy'], self.action_history)
            # If the agent's trajectory indicates that one (or more) strategies have been partially followed
            if x:
                # If the option to start decaying the reward shaping has been set and decay_n timesteps has elapsed
                # or if the option to stop reward shaping has not been set, apply an additional reward
                if self.decay:
                    if self.decay_param != 0:
                        # Determine the value of the additional reward based on the specified decay rate
                        new_reward += (y*self.trajectory_reward *
                                       self.decay_param)
                elif not self.stop_shaping:
                    # reward is influenced by how much of the strategy is being followed
                    # additional reward is equal to a percentage of the trajectory reward (so far)
                    new_reward += (y*self.trajectory_reward)

        if self.decay:
            self.decay_param *= self.decay_param

        return obs, new_reward, terminated, truncated, info
