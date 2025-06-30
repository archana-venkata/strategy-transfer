# Strategy Transfer in Reinforcement Learning via Reward Shaping

### A. Components of Strategy Transfer

- **PPO Agent**
	- Proximal Policy Optimization is used to implement the baseline policy and as a basis for the reward shaping policies.
	- The agent is created in ``main.py`` using [Stable Baselines 3](https://stable-baselines3.readthedocs.io/en/master/)
	- Hyperparameters are set to their default as defined here [MaskablePPO](https://sb3-contrib.readthedocs.io/en/master/_modules/sb3_contrib/ppo_mask/ppo_mask.html#MaskablePPO)
 
- **Gym Environments**
    - All environments are implemented using [Gymnasium](https://gymnasium.farama.org/index.html)
    - To install run: `pip install -e .` from the `gym_envs` directory

- **Reward Wrappers**
    - Reward shaping is incorporated into training using the reward wrapper classes defined in ``gym_envs/wrappers``

- **Strategies**
	- Strategy instantiations used for each environment during experimentation are provided in ``strategies.json``

### B. Running the Code
- **Scripts**
	- To install all required packages `pip install -r requirements.txt`
	- To run experiments use the shell script ``run.sh``
	- The list of available command line arguments can be found in ``main.py``

- **Logging**
	- We implement the logging through Tensorboard: ``tensorboard --logdir=./data``
	- After executing run.py, you can monitor training through your browser at ``localhost:6006``
	- The monitoring callback is defined in ``main.py``
