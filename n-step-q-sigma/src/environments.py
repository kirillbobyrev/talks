'''
This file contains implementation of Stochastic Windy Gridworld
environment used in Multi-step Reinforcement Learning: A Unifying Algorithm
(https://arxiv.org/abs/1703.01327) paper.

TODO: Add more documentation.
'''

import gym
import gym.spaces
import numpy


class StochasticWindyGridworld(gym.Env):
    '''
    Slightly modified tabular navigation task in standard gridworld which is
    described in (Sutton & Barto, Chapter 6.4, Example 6.5, 2017).

    Stochastic Windy Gridworld is a tabular navigation taks with Start and Goal
    states (donoted by S and G respectively) and four possible moves: right,
    left, up and down.

    When the agent moves into one of the middle columns of the grid-world, it
    is affected by an upward "wind" shifting the next state by the number of
    cells indicated below the corresponding column in the following scheme.
    Agent returns to the nearest state at the edge of the world upon being
    pushed out of the world or leaving it voluntarily.

    At the end of each time step the agent receives a constant reward of -1
    unless the Goal state is reached.

    0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0
    S 0 0 0 0 0 0 G 0 0
    0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0
    - - - - - - - - - -
    0 0 0 1 1 1 2 2 1 0
    '''
    width = 10
    height = 7
    action_space = gym.spaces.Discrete(4)
    observation_space = gym.spaces.MultiDiscrete([[0, width], [0, height]])
    reward_range = (-numpy.inf, 0)

    RIGHT, LEFT, UP, DOWN = (0, 1, 2, 3)

    metadata = {
        'render.modes': ['human'],
    }

    def __init__(self, random_step_probability=0.1):
        self.start_state = numpy.array((0, 3), dtype=numpy.int8)
        self.goal_state = numpy.array((7, 3), dtype=numpy.int8)
        self.state = self.start_state.copy()
        self.random_step_probability = random_step_probability

    def _step(self, action):
        assert(not numpy.array_equal(self.state, self.goal_state))
        mapping = [(1, 0), (-1, 0), (0, -1), (0, 1)]
        movement = mapping[action]
        if numpy.random.uniform() < self.random_step_probability:
            # Assign movement vector to a random vector resulting in adjacent
            # state.
            random_movement = [(1, 0), (-1,  0), (0, -1), (0,  1),
                               (1, 1), (-1, -1), (1, -1), (-1, 1)]
            random_vector_index = numpy.random.choice(len(random_movement))
            movement = random_movement[random_vector_index]
        self.state += movement
        self.state = numpy.amin((self.state, (self.width - 1, self.height - 1)),
                                axis=0)
        self.state = numpy.amax((self.state, (0, 0)), axis=0)
        done = numpy.array_equal(self.state, self.goal_state)
        reward = -1
        if done:
            reward = 0
        info = {}
        return self.state, reward, done, info

    def _reset(self):
        self.state = self.start_state.copy()

    def _render(self, mode='human', close=False):
        for j in range(self.height):
            for i in range(self.width):
                symbol = 'o'
                if numpy.array_equal((i, j), self.state):
                    symbol = 'X'
                elif numpy.array_equal((i, j), self.start_state):
                    symbol = 'S'
                if numpy.array_equal((i, j), self.goal_state):
                    symbol = 'G'
                print('{} '.format(symbol), end='')
            print()
        print()

    def _seed(self, seed=None):
        numpy.random.seed(seed)
