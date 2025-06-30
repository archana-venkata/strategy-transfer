# Code adapted from source provided from the authors of:

# @inproceedings{Muller2023incompleteplans,
#   title        = "Using incomplete and incorrect plans to shape reinforcement
#                   learning in long-sequence sparse-reward tasks",
#   author       = "M{\"u}ller, Henrik and Berg, Lukas and Kudenko, Daniel",
#   booktitle    = {Proceedings of the Adaptive and Learning Agents Workshop},
#   conference   = {The 22nd International Conference on Autonomous Agents and Multiagent Systems},
#   year         =  2023,
#   address      = "London"
# }

import logging
import cv2
import gymnasium as gym
from addict import addict
from gymnasium import Env, spaces
import numpy as np
import plotly.colors as colors
from stable_baselines3.common.env_checker import check_env

# Direction constants
LEFT = 2
DOWN = 1
RIGHT = 3
UP = 0
actions_str = ["LEFT", "DOWN", "RIGHT", "UP"]


class Grid_Renderer:
    def __init__(self, grid_size, color_map, background_color=(0, 0, 0)):
        self.grid_size = int(grid_size)
        self.background_color = background_color
        self.color_map = color_map

    def render_nd_grid(self, grid, flip=False):
        """
        The grid is supposed to be in this format (h, w, n_objects), the channel idx is supposed to be the object id
        """
        if flip:
            grid = np.swapaxes(grid, 0, 1)
            grid = np.flip(grid, axis=0)
        g_s = self.grid_size
        h, w = grid.shape[0], grid.shape[1]
        img = np.zeros(shape=(int(g_s * h), int(g_s * w), 3))
        for x in range(h):
            for y in range(w):
                if grid[x, y, :].any():
                    object_id = grid[x, y, :].argmax()
                    if object_id in self.color_map:
                        img[x * g_s:(x + 1) * g_s, y * g_s:(y + 1)
                            * g_s, :] = self.color_map[object_id]
        return img.astype(np.uint8)

    def render_2d_grid(self, grid, flip=False):
        if flip:
            grid = np.swapaxes(grid, 0, 1)
            grid = np.flip(grid, axis=0)
        g_s = self.grid_size
        h, w = grid.shape[0], grid.shape[1]
        img = np.zeros(shape=(int(g_s * h), int(g_s * w), 3))
        for x in range(h):
            for y in range(w):
                object_id = grid[x, y]
                if object_id in self.color_map:
                    img[x * g_s:(x + 1) * g_s, y * g_s:(y + 1) *
                        g_s, :] = self.color_map[object_id]
        return img.astype(np.uint8)


class MinecraftEnv(Env):
    # noinspection PySetFunctionToLiteral

    action_desc = ['wall()',
                   'collect(wood)',
                   'at(workbench)',
                   'get(wood_pickaxe)',
                   'collect(stone)',
                   'get(stone_pickaxe)',
                   'collect(ironore)',
                   'at(furnace)',
                   'get(iron_ingot)',
                   'get(iron_pickaxe)',
                   'collect(diamond)',
                   'died(lava)',
                   'out_of_moves()',
                   'out_of_wood()']

    def __init__(self, success_reward=0):
        # actions: up, down, left, right
        super(MinecraftEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(151,),
                                            dtype=np.int16)

        self.obs_type = np.int16
        # game config
        self.success_reward = success_reward
        self.height = 12
        self.width = 12
        self.objects = addict.Dict({
            'agent': {
                'id': 1,
                'location': (1, 1),
                'color': colors.hex_to_rgb(colors.qualitative.Plotly[1])  # red
            },
            'wood': {
                'id': 2,
                'locations': [(1, 7), (5, 2), (7, 8), (7, 3)],
                'color': colors.hex_to_rgb(colors.qualitative.D3[2])  # green
            },
            'stone': {
                'id': 3,
                'locations': [(10, 2)],
                # dark gray
                'color': colors.hex_to_rgb(colors.qualitative.D3[7])
            },
            'iron': {
                'id': 4,
                'locations': [(10, 5)],
                # light gray
                'color': colors.hex_to_rgb(colors.qualitative.T10[9])
            },
            'diamond': {
                'id': 5,
                'locations': [(9, 9)],
                # light green
                'color': colors.hex_to_rgb(colors.qualitative.Plotly[2])
            },
            'workbench': {  # wood -> plank
                'id': 6,
                'location': (8, 5),
                # yellow
                'color': colors.hex_to_rgb(colors.qualitative.Plotly[9])
            },
            'furnace': {  # (wood, iron ore) -> iron ingot
                'id': 9,
                'location': (4, 8),
                'color': colors.hex_to_rgb(colors.qualitative.D3[1])  # orange
            },
            'wall': {  # only keep outline
                'id': 10,
                'locations': [(i, 0) for i in range(self.height)] +
                             [(i, self.width-1) for i in range(self.height)] +
                             [(self.height - 1, i) for i in range(self.width)] +
                             [(0, i) for i in range(self.width)],
                # dark brown
                'color': colors.hex_to_rgb(colors.qualitative.D3[5])
            },
            'lava': {
                'id': 11,
                'locations': [(1, 9), (2, 9), (1, 10), (2, 10), (3, 10)],
                # light blue
                'color': colors.hex_to_rgb(colors.qualitative.D3[6])
            },
        })
        # finalize the coordinates of wall
        self.wall_positions = set(self.objects.wall['locations'])

        # init
        self._init()
        # init renderer
        color_map = {
            self.objects[obj].id: self.objects[obj].color for obj in self.objects}
        self.renderer = Grid_Renderer(grid_size=20, color_map=color_map)

    def _init(self):
        # update flags and state info
        self.has_wood = False
        self.has_stone = False
        self.has_iron_ore = False
        self.has_iron_ingot = False
        self.has_diamond = False
        self.obtained_pickaxes = set()
        self.wood_pickaxe = False  # start with a pickaxe
        self.stone_pickaxe = False
        self.iron_pickaxe = False

        self.wood_counter = 0

        self.at_destination = False
        self.failed = False

        self.score = 0
        self.steps = 0
        self.max_allowed_steps = 200
        self.max_steps = 200
        self.default_reward = 0

        # state
        self.info = dict()
        self.agent_pos = self.objects.agent.location
        # map
        self.grid = np.zeros(shape=(self.height, self.width))
        # init wall
        for wall_pos in self.wall_positions:
            self.grid[wall_pos[0], wall_pos[1]] = self.objects.wall.id
        # place the agent
        self.grid[self.agent_pos[0], self.agent_pos[1]] = self.objects.agent.id
        # place ressources
        for wood_pos in self.objects.wood.locations:
            self.grid[wood_pos[0], wood_pos[1]] = self.objects.wood.id
        for stone_pos in self.objects.stone.locations:
            self.grid[stone_pos[0], stone_pos[1]] = self.objects.stone.id
        for iron_pos in self.objects.iron.locations:
            self.grid[iron_pos[0], iron_pos[1]] = self.objects.iron.id
        for diamond_pos in self.objects.diamond.locations:
            self.grid[diamond_pos[0], diamond_pos[1]] = self.objects.diamond.id
        # place crafting stations
        workbench_pos = self.objects.workbench.location
        self.grid[workbench_pos[0], workbench_pos[1]
                  ] = self.objects.workbench.id
        furnace_pos = self.objects.furnace.location
        self.grid[furnace_pos[0], furnace_pos[1]] = self.objects.furnace.id
        for lava_pos in self.objects.lava.locations:
            self.grid[lava_pos[0], lava_pos[1]] = self.objects.lava.id

    def valid_action_mask(self):
        row = self.agent_pos[0]
        col = self.agent_pos[1]
        possible_grid_ids = [self.grid[row-1, col],
                             self.grid[row+1, col], self.grid[row, col-1], self.grid[row, col+1]]
        return [int(grid_id != self.objects.wall.id) for grid_id in possible_grid_ids]

    def make_observation(self):
        grid_obs = np.copy(self.grid).flatten()
        grid_obs[grid_obs != 0] = 1
        obs = np.append(grid_obs, [int(self.has_wood), int(self.has_stone), int(self.has_iron_ore), int(
            self.has_iron_ingot), int(self.wood_pickaxe), int(self.stone_pickaxe), int(self.iron_pickaxe)])

        return obs.astype(self.obs_type)

    def make_info(self, result_of_action=[], rewards_per_action=[]):
        info = {
            "result_of_action": result_of_action,
            "rewards_per_action": rewards_per_action,
            "score": self.score
        }
        return info

    def render(self, mode='rgb_array'):
        return self.renderer.render_2d_grid(self.grid)

    def step(self, action):
        def _go_to(x, y):
            _old_pos = tuple(self.agent_pos)
            self.agent_pos = (x, y)
            self.grid[x, y] = self.objects.agent.id
            # you only move into spaces you are allowed to interact with
            # resources are picked up upon entry and do not have to be replaced after moving to another space
            # crafting benches have to be replaced upon exit
            if self.agent_pos != _old_pos:
                if _old_pos == self.objects.workbench.location:
                    self.grid[_old_pos[0], _old_pos[1]
                              ] = self.objects.workbench.id
                elif _old_pos == self.objects.furnace.location:
                    self.grid[_old_pos[0], _old_pos[1]
                              ] = self.objects.furnace.id
                else:
                    self.grid[_old_pos[0], _old_pos[1]] = 0

        assert action <= 3
        reward = 0
        rewards_per_action = []
        results_of_action = []

        old_pos = tuple(self.agent_pos)
        next_x, next_y = self.agent_pos[0] + \
            self.actions[action][0], self.agent_pos[1] + \
            self.actions[action][1]

        # update info
        old_info = self.info.copy()

        # update grid
        # check if the agent will hit into walls, then make no change
        if self.grid[next_x, next_y] == self.objects.wall.id:
            results_of_action.append(self.action_desc[0])
            pass
        # ressource pickup (if required tool obtained)
        elif self.grid[next_x, next_y] == self.objects.wood.id:
            if not self.has_wood:
                self.has_wood = True
            if self.score == 0:
                reward += 1
            else:
                reward += 1
            self.wood_counter += 1
            results_of_action.append(self.action_desc[1])
            rewards_per_action.append(self.default_reward)
            _go_to(next_x, next_y)
        elif self.grid[next_x, next_y] == self.objects.stone.id:
            if not self.has_stone and self.wood_pickaxe:
                results_of_action.append(self.action_desc[4])
                rewards_per_action.append(self.default_reward)

                self.has_stone = True

                reward += 1

                _go_to(next_x, next_y)
        elif self.grid[next_x, next_y] == self.objects.iron.id:
            if not self.has_iron_ore and self.stone_pickaxe:
                results_of_action.append(self.action_desc[6])
                rewards_per_action.append(self.default_reward)

                self.has_iron_ore = True

                reward += 1
                _go_to(next_x, next_y)
        elif self.grid[next_x, next_y] == self.objects.diamond.id:
            if not self.has_diamond and self.iron_pickaxe:
                results_of_action.append(self.action_desc[10])
                rewards_per_action.append(self.default_reward)

                self.has_diamond = True
                self.at_destination = True
                reward += 50
                _go_to(next_x, next_y)
        # use collected resource to craft if at a crafting station
        elif self.grid[next_x, next_y] == self.objects.workbench.id:
            # self.info['at_workbench'] = True
            results_of_action.append(self.action_desc[2])
            rewards_per_action.append(self.default_reward)
            # wood -> plank
            if not self.wood_pickaxe:
                if self.has_wood and self.wood_counter == 2:
                    self.has_wood = False
                    self.wood_counter = 0
                    results_of_action.append(self.action_desc[3])
                    rewards_per_action.append(self.default_reward)

                    self.wood_pickaxe = True
                    reward += 1

            elif not self.stone_pickaxe:
                if self.has_wood and self.has_stone:
                    self.has_wood = False
                    self.has_stone = False
                    results_of_action.append(self.action_desc[5])
                    rewards_per_action.append(self.default_reward)

                    self.stone_pickaxe = True
                    reward += 1
            elif not self.iron_pickaxe:
                if self.has_iron_ingot:
                    self.has_iron_ingot = False
                    results_of_action.append(self.action_desc[9])
                    rewards_per_action.append(self.default_reward)

                    self.iron_pickaxe = True
                    reward += 1

            if self.has_wood:
                # fail if no more wood available but no iron ingot smelted yet
                if not self.has_iron_ingot:
                    self.failed = True
                    for wood_location in self.objects.wood.locations:
                        if self.grid[wood_location[0], wood_location[1]] == self.objects.wood.id:
                            self.failed = False
                            break
                    if self.failed:
                        reward -= 50
                        results_of_action.append(self.action_desc[13])
                        rewards_per_action.append(self.default_reward)

                        logging.debug(
                            "no more wood available but no iron ingot smelted yet")

            _go_to(next_x, next_y)
        elif self.grid[next_x, next_y] == self.objects.furnace.id:
            self.info['at_furnace'] = True
            results_of_action.append(self.action_desc[7])
            rewards_per_action.append(self.default_reward)

            # use wood to smelt the ore to get the ingot
            if self.has_wood and self.has_iron_ore:
                self.has_wood = False
                self.has_iron_ore = False
                results_of_action.append(self.action_desc[8])
                rewards_per_action.append(self.default_reward)

                self.has_iron_ingot = True
                reward += 1
            _go_to(next_x, next_y)
        elif self.grid[next_x, next_y] == 0:
            _go_to(next_x, next_y)
        elif self.grid[next_x, next_y] == self.objects.lava.id:
            # die in lava
            self.failed = True
            reward -= 50
            results_of_action.append(self.action_desc[11])
            rewards_per_action.append(self.default_reward)

            logging.debug('died in lava')
            _go_to(next_x, next_y)

        for i in range(1, len(self.objects.wood.locations) + 1):
            # sets has_wood2=True while holding the second piece of wood
            self.info[f'has_wood{i}'] = (
                self.has_wood and self.wood_counter == i)
        self.info['has_stone'] = self.has_stone
        self.info['has_iron_ore'] = self.has_iron_ore
        self.info['has_iron_ingot'] = self.has_iron_ingot
        self.info['has_diamond'] = self.has_diamond
        self.info['wood_pickaxe'] = self.wood_pickaxe
        self.info['stone_pickaxe'] = self.stone_pickaxe
        self.info['iron_pickaxe'] = self.iron_pickaxe
        self.info['at_destination'] = self.at_destination

        # Agent has a maximum number of steps
        # Apply penalty if the agent exceeds the maximum number of steps allowed
        # End the game if the number of steps taken is too large
        self.steps += 1
        if self.steps >= self.max_allowed_steps:
            reward -= 1
            results_of_action.append(self.action_desc[12])
            rewards_per_action.append(self.default_reward)
            logging.debug("max steps reached")
            if self.steps >= self.max_steps:
                self.failed = True

        # get info
        self.score += reward

        # return info
        terminated = self.at_destination or self.failed
        info = self.make_info(results_of_action, rewards_per_action)

        # reward = self.success_reward if self.at_destination else 0
        return self.make_observation(), float(reward), terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init()
        return self.make_observation(), self.make_info()


def self_play():
    logging.debug('starting craftworld')
    import cv2
    env = MinecraftEnv()

    check_env(env)

    obs = env.reset()
    # logging.debug(f'[INFO] observation shape: {obs.shape}')
    terminated = False
    info = None
    prev_reward = 0
    steps = 0
    score = 0
    trajectory = []
    while not terminated:
        grid_img = env.render()
        cv2.imshow('grid render', cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR))
        a = cv2.waitKey(0)
        # actions: up, down, left, right
        if a == ord('q'):
            break
        elif a == ord('w'):
            a = 0
        elif a == ord('a'):
            a = 2
        elif a == ord('s'):
            a = 1
        elif a == ord('d'):
            a = 3
        elif a == ord('e'):
            a = 4

        mask = env.unwrapped.valid_action_mask()

        obs, reward, terminated, info = env.step(int(a))

        if reward != prev_reward:
            logging.debug(f"Reward: {reward}")

        if len(info['result_of_action']) != 0:
            trajectory.append(info['result_of_action'])

        # for item in info['result_of_action']:
        #     trajectory.append(item)

        prev_reward = reward
        score += reward

        print(f"Reward: {reward}, Score: {score}")

        steps += 1
        if terminated:
            print(f"Episode score: {score}")
            print(trajectory)
            logging.debug(f"{info}")
            logging.debug(f'Total steps: {steps}')
            logging.debug("terminated")


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.debug('Start game')

    env = MinecraftEnv()
    check_env(env)
    num_episodes = 1

    # testing the environment by playing a number of episodes
    for episode in range(num_episodes):
        observation = env.reset()

        terminated = False
        score = 0

        # play a certain number of moves for each episode
        while not terminated:
            observation, reward, terminated, info = env.step(
                env.action_space.sample())
            if terminated:
                logging.debug("terminated")

            if reward > 0:
                logging.debug(f"Observation: {observation}")
                logging.debug(f'{info}')
                logging.debug(f'{reward}')

            score += reward

            env.render()

        logging.debug('Episode: {}, Score: {}'.format(episode, score))
        logging.debug('')


if __name__ == "__main__":
    self_play()
