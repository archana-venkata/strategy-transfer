from typing import List
from util.Move import Move
from util.Trajectory import Trajectory
import pickle
import os


def save_params(args, dir_name):
    f = os.path.join(dir_name, "params.txt")

    with open(f, "a+") as f_w:
        f_w.write("\n")
        for param, val in sorted(vars(args).items()):
            f_w.write("{0}:{1}\n".format(param, val))


def create_trajectory(memory):
    t = Trajectory()
    for state, action, reward, next_state, done, info in memory:
        for index, value in enumerate(info["result_of_action"]):
            if value == info["result_of_action"][-1]:
                t.add(Move(state, action, info["rewards_per_action"][index], next_state,
                           done, {"result_of_action": value}))
            else:
                t.add(Move(state, action, info["rewards_per_action"][index], next_state,
                           0, {"result_of_action": value}))

    return t

# function which will remove contradicting trajectories
# e.g. if a trajectory suggests a ghost was killed without a powerup being collected
# when in actual fact the powerup was collected in the same move as the ghost kill

# strategy 1


def filter_trajectories(data):
    for idx, traj in enumerate(data):
        result = any(checkIfMatch(move) for move in traj.get_moves())
        if not result:
            del data[idx]
    return data


def checkIfMatch(elem):
    return elem.info['result_of_action'] == 'collected powerup'

# strategy 2


def filter_trajectories(data):
    for idx, traj in enumerate(data):
        result = any(checkIfMatch(move) for move in traj.get_moves())
        if not result:
            del data[idx]
    return data


def checkIfMatch(elem):
    return elem.info['result_of_action'] == 'collected powerup'


def find_shortest_trajectory(trajectories: List[Trajectory]):
    num_moves = [len(t.get_moves()) for t in trajectories]
    shortest_trajectory_idx = num_moves.index(min(num_moves))
    shortest_trajectory = trajectories[shortest_trajectory_idx].get_moves()
    return shortest_trajectory, shortest_trajectory_idx


def is_subsequence(a, b):
    b_it = iter(b)
    count = 0
    try:
        for a_val in a:
            if a_val in b:
                count += 1
                while str(next(b_it)) != str(a_val):
                    pass

    except StopIteration:
        return False, 0

    if count == 0:
        return False, 0
    else:
        return True, count/len(a)


if __name__ == "__main__":
    b = ["collect dot", "kill ghost"]
    a = ["collect dot", "collect dot", "collect powerup"]
    c = [6, 0, 5]

    print(is_subsequence(b, a))
    # print(list(set(a) & set(b)))
