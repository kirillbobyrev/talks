'''
This file contains implementation of 19-State Random Walk and Stochastic Windy
Gridworld environments used in Multi-step Reinforcement Learning: A Unifying
Algorithm (https://arxiv.org/abs/1703.01327) paper.
'''

import gym
import gym.spaces
import numpy


class RandomWalk(gym.Env):
    '''
    Replication of n-State Random Walk (originally n=19) environment.
    '''
    def __new__(self, n_states=19):
        action_space = gym.spaces.Discrete(2)
        observation_space = 0
        reward_range = (-1, 1)

    def _step(self, action):
        pass

    def _reset(self):
        pass

    def _render(self):
        pass

    def _seed(self, seed=None):
        pass


class StochasticWindyGridworld(gym.Env):
    '''
    Slightly modified tabular navigation task in standard gridworld which is
    described in (Sutton & Barto, Chapter , 2017).
    '''
    def __new__(self, n_states=19):
        width = 10
        height = 7
        action_space = gym.spaces.Discrete(4)
        observation_space = gym.spaces.Discrete(width * height)
        reward_range = (-numpy.inf, 1)

    def _step(self, action):
        pass

    def _reset(self):
        pass

    def _render(self):
        pass

    def _seed(self, seed=None):
        pass
