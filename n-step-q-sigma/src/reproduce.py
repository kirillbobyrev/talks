'''
Reproduce the results of "Multi-step Reinforcement Learning: A Unifying
Algorithm" (https://arxiv.org/abs/1703.01327) paper and obtain the equivalent
figures.
'''

import matplotlib
import seaborn
from q_sigma import Q_sigma
from environments import StochasticWindyGridworld
import gym


def main():
    environment = gym.make('FrozenLake-v0') # StochasticWindyGridworld(0)
    Q_sigma(environment)


if __name__ == '__main__':
    main()
