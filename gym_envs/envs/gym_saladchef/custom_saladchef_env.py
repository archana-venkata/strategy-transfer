from gymnasium import Env, spaces
import numpy as np
from stable_baselines3.common.env_checker import check_env
from addict import addict
import plotly.colors as colors
import logging
import cv2
import random


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


class SaladChefEnv(Env):

    action_desc = ['wall()',
                   'at(garden)',
                   'collect(ingredient)',
                   'at(cutting_station)',
                   'get(cut_ingredients)',
                   'at(mixing_station)',
                   'make(dish)',
                   'at(plating station)',
                   'plate(dish)',
                   'at(serving_station)',
                   'serve(dish)',
                   'out_of_moves()',
                   'served_all_dishes()',
                   'collect(plate)']

    def __init__(self, success_reward=0):
        # actions: up, down, left, right
        super(SaladChefEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        self.obs_type = np.int16
        self.observation_space = self.get_observation_space()

        # game config
        self.success_reward = success_reward
        self.height = 12
        self.width = 12

        self.garden_width = 6
        self.garden_height = 5

        self.objects = addict.Dict({
            'agent': {
                'id': 1,
                'location': (6, 6),
                'color': colors.hex_to_rgb(colors.qualitative.Plotly[1])  # red

            },
            'garden': {
                'id': 2,
                'locations': [(i, 1) for i in range(1, self.garden_height)] +
                             [(i, self.garden_width-1) for i in range(1, self.garden_height)] +
                             [(self.garden_height - 1, i) for i in range(1, self.garden_width)] +
                             [(1, i) for i in range(1, self.garden_width)],
                'color': colors.hex_to_rgb(colors.qualitative.D3[2])  # green

            },
            'cutting_station': {
                'id': 3,
                'locations': [(10, 3), (10, 4)],
                # light blue
                'color': colors.hex_to_rgb(colors.qualitative.D3[6])

            },
            'mixing_station': {
                'id': 4,
                'locations': [(10, 7), (10, 8)],
                # yellow
                'color': colors.hex_to_rgb(colors.qualitative.Plotly[9])

            },
            'plating_station': {
                'id': 5,
                'locations': [(8, 10), (7, 10)],
                # light green
                'color': colors.hex_to_rgb(colors.qualitative.Plotly[2])

            },
            'serving_station': {
                'id': 6,
                'locations': [(1, 9), (1, 10)],
                'color': colors.hex_to_rgb(colors.qualitative.D3[1])  # orange

            },
            'plate': {
                'id': 7,
                'locations': [(1, 9), (1, 10)],
                # light gray
                'color': colors.hex_to_rgb(colors.qualitative.T10[9])

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

        })
        self.garden_locations = [(i, j) for i in range(
            1, self.garden_height) for j in range(1, self.garden_width)]

        self.empty_garden_locations = list(
            set(self.garden_locations) - set(self.objects['garden'].locations))

        self.ingredients = ['tomato', 'lettuce']

        for i, ingredient in enumerate(self.ingredients):
            self.objects[ingredient] = addict.Dict({
                'id': i+8,
                'location': None,
                'color': colors.hex_to_rgb(colors.qualitative.T10[i+2])

            })

        # finalize the coordinates of wall
        self.wall_positions = set(self.objects.wall['locations'])

        # init
        self._init()
        # init renderer
        color_map = {
            self.objects[obj].id: self.objects[obj].color for obj in self.objects}
        self.renderer = Grid_Renderer(grid_size=20, color_map=color_map)

    def get_observation_space(self):
        return spaces.Box(low=0, high=1,
                          shape=(144,),
                          dtype=self.obs_type)

    def render(self, mode='rgb_array'):
        return self.renderer.render_2d_grid(self.grid)

    def _init(self):
        # update flags and state info
        self.cut_ingredients = False
        self.has_ingredients = False
        self.mixed_ingredients = False
        self.plated_dish = False
        self.total_served = 2
        self.served_dishes = 0
        self.has_plate = False

        self.at_destination = False
        self.failed = False

        self.score = 0
        self.steps = 0
        self.max_allowed_steps = 200
        self.default_reward = 0
        self.inventory = []

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

        # place resources
        for garden_pos in self.objects.garden.locations:
            self.grid[garden_pos[0], garden_pos[1]] = self.objects.garden.id

        for cutting_station_pos in self.objects.cutting_station.locations:
            self.grid[cutting_station_pos[0], cutting_station_pos[1]
                      ] = self.objects.cutting_station.id

        for mixing_station_pos in self.objects.mixing_station.locations:
            self.grid[mixing_station_pos[0], mixing_station_pos[1]
                      ] = self.objects.mixing_station.id

        for plating_station_pos in self.objects.plating_station.locations:
            self.grid[plating_station_pos[0], plating_station_pos[1]
                      ] = self.objects.plating_station.id

        for serving_station_pos in self.objects.serving_station.locations:
            self.grid[serving_station_pos[0], serving_station_pos[1]
                      ] = self.objects.serving_station.id

        self.place_ingredients()

        self.place_plate()

    def valid_action_mask(self):
        row = self.agent_pos[0]
        col = self.agent_pos[1]
        possible_grid_ids = [self.grid[row-1, col],
                             self.grid[row+1, col], self.grid[row, col-1], self.grid[row, col+1]]
        return [int(grid_id != self.objects.wall.id) for grid_id in possible_grid_ids]

    def render(self, mode='rgb_array'):
        return self.renderer.render_2d_grid(self.grid)

    def make_observation(self):
        grid_obs = np.copy(self.grid).flatten()
        grid_obs[grid_obs != 0] = 1
        obs = np.append(grid_obs, [])

        return obs.astype(self.obs_type)

    def make_info(self, result_of_action=[], rewards_per_action=[]):
        info = {
            "result_of_action": result_of_action,
            "rewards_per_action": rewards_per_action,
            "score": self.score
        }
        return info

    def place_ingredients(self):
        self.ingredient_locations = random.sample(
            self.empty_garden_locations, len(self.ingredients))
        for i, ingredient in enumerate(self.ingredients):

            self.objects[ingredient].location = self.ingredient_locations[i]

            ingredient_pos = self.objects[ingredient].location
            grid_pos = self.grid[ingredient_pos[0], ingredient_pos[1]]
            if int(grid_pos) == 0:
                self.grid[ingredient_pos[0], ingredient_pos[1]
                          ] = self.objects[ingredient].id

    def place_plate(self):
        zeros = np.argwhere(self.grid == 0)  # Indices where board == 0
        indices = np.ravel_multi_index(
            [zeros[:, 0], zeros[:, 1]], self.grid.shape)  # Linear indices

        # Randomly select your index to replace
        ind = np.random.choice(indices)

        plate_pos = np.unravel_index(ind, self.grid.shape)

        self.objects.plate.location = plate_pos
        grid_pos = self.grid[plate_pos[0], plate_pos[1]]
        if int(grid_pos) == 0:
            self.grid[plate_pos[0], plate_pos[1]
                      ] = self.objects.plate.id

    def step(self, action):
        def _go_to(x, y):
            _old_pos = tuple(self.agent_pos)
            self.agent_pos = (x, y)
            self.grid[x, y] = self.objects.agent.id
            # you only move into spaces you are allowed to interact with
            # resources are picked up upon entry and do not have to be replaced after moving to another space
            # crafting benches have to be replaced upon exit
            if self.agent_pos != _old_pos:
                if _old_pos in self.objects.garden.locations:
                    self.grid[_old_pos[0], _old_pos[1]
                              ] = self.objects.garden.id
                elif _old_pos in self.objects.cutting_station.locations:
                    self.grid[_old_pos[0], _old_pos[1]
                              ] = self.objects.cutting_station.id
                elif _old_pos in self.objects.mixing_station.locations:
                    self.grid[_old_pos[0], _old_pos[1]
                              ] = self.objects.mixing_station.id
                elif _old_pos in self.objects.plating_station.locations:
                    self.grid[_old_pos[0], _old_pos[1]
                              ] = self.objects.plating_station.id
                elif _old_pos in self.objects.serving_station.locations:
                    self.grid[_old_pos[0], _old_pos[1]
                              ] = self.objects.serving_station.id
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
            rewards_per_action.append(self.default_reward)

            pass
        elif self.grid[next_x, next_y] == self.objects.garden.id:
            # in the garden
            results_of_action.append(self.action_desc[1])
            rewards_per_action.append(self.default_reward)
            _go_to(next_x, next_y)
        elif self.grid[next_x, next_y] == self.objects.plate.id:
            if not self.has_plate:
                self.has_plate = True
                self.inventory.append('plate')
                # print(f"picked up plate")

                # if self.score == 0:
                #     reward += 1
                # else:
                #     reward += 1

                results_of_action.append(self.action_desc[13])
                rewards_per_action.append(self.default_reward)

                self.place_plate()

                _go_to(next_x, next_y)

        elif self.grid[next_x, next_y] == self.objects.cutting_station.id:
            # "at cutting station"
            results_of_action.append(self.action_desc[3])
            rewards_per_action.append(self.default_reward)
            if self.has_ingredients:

                self.cut_ingredients = True

                self.has_ingredients = False

                if self.score == 0:
                    reward += 1
                else:
                    reward += 1
                results_of_action.append(self.action_desc[4])
                rewards_per_action.append(reward)

                _go_to(next_x, next_y)
        elif self.grid[next_x, next_y] == self.objects.mixing_station.id:
            # print("at mixing station")
            results_of_action.append(self.action_desc[5])
            rewards_per_action.append(self.default_reward)
            if self.cut_ingredients:
                self.cut_ingredients = False
                self.mixed_ingredients = True

                reward += 1
                results_of_action.append(self.action_desc[6])
                rewards_per_action.append(reward)
                _go_to(next_x, next_y)
        elif self.grid[next_x, next_y] == self.objects.plating_station.id:
            # print("at plating station")
            results_of_action.append(self.action_desc[7])
            rewards_per_action.append(self.default_reward)
            if self.mixed_ingredients and self.has_plate:
                self.plated_dish = True
                self.has_plate = False
                self.mixed_ingredients = False

                reward += 1
                results_of_action.append(self.action_desc[8])
                rewards_per_action.append(reward)

                _go_to(next_x, next_y)
        elif self.grid[next_x, next_y] == self.objects.serving_station.id:
            # print("at serving station")
            results_of_action.append(self.action_desc[9])
            rewards_per_action.append(self.default_reward)
            if self.plated_dish:
                self.plated_dish = False
                self.place_ingredients()

                reward += 1
                results_of_action.append(self.action_desc[10])
                rewards_per_action.append(reward)
                self.served_dishes += 1

                if self.served_dishes == self.total_served:
                    self.at_destination = True
                    reward += 50
                    results_of_action.append(self.action_desc[12])
                    rewards_per_action.append(reward)

            _go_to(next_x, next_y)

        elif self.grid[next_x, next_y] == 0:
            _go_to(next_x, next_y)

        # ressource pickup (if required tool obtained)
        for ingredient in self.ingredients:
            if self.grid[next_x, next_y] == self.objects[ingredient].id:
                if ingredient not in self.inventory:
                    self.inventory.append(ingredient)
                    # print(f"picked {ingredient}")

                # if self.score == 0:
                #     reward += 1
                # else:
                #     reward += 1

                results_of_action.append(self.action_desc[2])
                rewards_per_action.append(self.default_reward)

                if set(sorted(self.ingredients)).issubset(set(sorted(self.inventory))):
                    self.has_ingredients = True

                _go_to(next_x, next_y)

        self.steps += 1
        if self.steps >= self.max_allowed_steps:
            reward -= 1
            results_of_action.append(self.action_desc[11])
            rewards_per_action.append(self.default_reward)
            logging.debug("max steps reached")
            if self.steps >= self.max_allowed_steps:
                self.failed = True

        # get info
        self.score += reward

        # return info
        terminated = self.at_destination or self.failed

        info = self.make_info(results_of_action, rewards_per_action)

        # reward = self.success_reward if self.at_destination else 0
        return self.make_observation(), float(reward), terminated, False, info

    def reset(self, seed=None):
        super().reset(seed=seed)
        self._init()
        return self.make_observation(), self.make_info()


def self_play():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.debug('Start game')

    env = SaladChefEnv()

    check_env(env)

    obs = env.reset()

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

        obs, reward, terminated, info = env.step(int(a))

        if reward != prev_reward:
            logging.debug(f"Reward: {reward}")

        if len(info['result_of_action']) != 0:
            trajectory.append(info['result_of_action'])

        prev_reward = reward
        score += reward

        print(f"Reward: {reward}, Score: {score}")

        steps += 1
        if terminated:
            print(f"Episode score: {score}")
            print(trajectory)
            logging.debug(f'Total steps: {steps}')
            logging.debug("terminated")


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.debug('Start game')

    env = SaladChefEnv()
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
