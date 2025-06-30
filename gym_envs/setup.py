from setuptools import setup

setup(name='gym_envs',
      version='7.0.0',
      install_requires=['gym'],  # And any other dependencies required
      packages=["envs", "wrappers"]

      )
