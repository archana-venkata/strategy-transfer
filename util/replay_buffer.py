import random
import numpy as np

from util.Move import Move
from util.common import create_trajectory
import time


class ReplayBuffer:
    def __init__(self, buffer_size) -> None:
        self._buffer_size = buffer_size
        self._buffer = []
        self._next_idx = 0
        self._episode_idxes = []

    def __len__(self):
        len(self._buffer)

    def new_episode(self):
        self._episode_idxes.append(self._next_idx)

    def add(self, state, action, reward, next_state, done, info):
        data = (state, action, reward, next_state, done, info)

        if self._next_idx >= len(self._buffer):
            self._buffer.append(data)
        else:
            self._buffer[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._buffer_size

    def _encode_sample(self, idxes):
        moves, states, actions, rewards, next_states, dones, infos = [], [], [], [], [], [], []
        for i in idxes:
            data = self._buffer[i]
            state, action, reward, next_state, done, info = data
            move = Move(state, action, reward, next_state, done, info)
            moves.append(move)
            states.append(np.array(state, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            next_states.append(np.array(next_state, copy=False))
            dones.append(done)
            infos.append(info)

        return np.array(moves), np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones), np.array(infos)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._buffer) - 1)
                 for _ in range(batch_size)]
        return self._encode_sample(idxes)

    def sample_n_last(self, n=None):
        if n == None:
            return self._buffer[self._episode_idxes[-1]:]
        return self._buffer[-n:]

    def sample_index(self, index):
        return self._buffer[index]

    def sample_random_episodes(self, n, max_runtime=6000, condition=None, not_condition=False):
        batch = []
        start = time.time()
        runtime = time.time()
        while len(batch) < n:
            # if (runtime - start) > max_runtime:
            #     break

            idx, start = random.choice(list(enumerate(self._episode_idxes)))

            if idx != len(self._episode_idxes) - 1:
                end = self._episode_idxes[idx+1]
                moves = self._buffer[start:end]
            else:
                moves = self._buffer[start:]

            new_trajectory = create_trajectory(moves)
            actions = new_trajectory.get_moves_info()

            # if a condition has been specified;
            # a condition can either indicate what must or must not be included in the buffer selections
            if condition != None:
                # if the condition is NOT to be included
                if not_condition and condition not in actions:
                    batch.append(new_trajectory)
                # else if the condition MUST be included
                elif not not_condition and condition in actions:
                    batch.append(new_trajectory)

            runtime = time.time()

        return batch

    def sample_trajectories(self, batch_size, len_of_trajectory=None):
        """Sample a batch of trajectories (retains temporal ordering of moves)

        Args:
            batch_size (int): How many trajectories to return
            len_of_trajectory (int): How many moves per trajectory to sample

        Returns:
            batch (list of Trajectory): list of trajectories
        """
        idxes = [random.randint(0, len(self._buffer) - 1)
                 for _ in range(batch_size)]
        batch = []
        for index in idxes:
            moves = []
            if len_of_trajectory == None:
                if index in self._episode_idxes:
                    index -= 1

                episode_start = max(
                    v for v in self._episode_idxes if v < index)

                for i in range(episode_start, index):
                    moves.append(self.sample_index(i))
            else:
                for i in range(index-len_of_trajectory, index):
                    moves.append(self.sample_index(i))
            batch.append(create_trajectory(moves))

        return batch
