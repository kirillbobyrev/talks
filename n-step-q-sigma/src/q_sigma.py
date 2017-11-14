'''
Implements Q(\sigma) algorithm as described in "Multi-step Reinforcement
Learning: A Unifying Algorithm" (https://arxiv.org/abs/1703.01327) paper.
'''

import numpy
import matplotlib


class QPolicy(object):
    def __init__(self, states_count, actions_count):
        self.states_count = states_count
        self.actions_count = actions_count
        self.Q = numpy.random.rand(states_count, actions_count)

    def get_probability(self, action, state):
        probabilities = numpy.zeros(self.actions_count)
        probabilities[numpy.argmax(self.Q[state])] = 1
        return probabilities[action]

    def sample_action(self, state):
        return numpy.argmax(self.Q[state])

    def getQ(self, state, action):
        return self.Q[state][action]

    def setQ(self, state, action, value):
        self.Q[state][action] = value


def Q_sigma(environment,
            episodes_count=50000,
            steps_count=4,
            sigma=0.5,
            discount_factor=0.9,
            learning_rate=0.4):
    assert (0 < learning_rate and learning_rate <= 1)
    policy = QPolicy(environment.observation_space.n,
                     environment.action_space.n)
    total_rewards = []

    for episode in range(episodes_count):
        states = [environment.reset()]
        actions = [policy.sample_action(states[0])]
        deltas = []
        Q = [policy.getQ(states[0], actions[0])]
        # Storing 0 to the \pi_{t = 0} helps to make the backup more concise.
        pi = [0]
        total_reward = 0
        rewards = []

        t = 0
        T = numpy.infty
        while True:
            if t < T:
                '''
                If the episode is not finished, make an action, observe a state
                and sample next action.
                '''
                current_state, reward, done, _ = \
                        environment.step(actions[t])
                states.append(current_state)
                rewards.append(reward)
                total_reward += reward
                if done:
                    T = t + 1
                    delta = reward - Q[t]
                    deltas.append(delta)
                else:
                    actions.append(policy.sample_action(states[t + 1]))
                    Q.append(policy.getQ(states[t + 1], actions[t + 1]))
                    # Calculate \delta_t and store it.
                    pure_expectation = 0
                    for action in range(environment.action_space.n):
                        pure_expectation += policy.get_probability(
                            action, states[t + 1]) * policy.getQ(
                                states[t + 1], action)
                    pure_expectation *= discount_factor * (1 - sigma)
                    delta = reward + discount_factor * sigma * Q[t + 1] + \
                        pure_expectation - Q[t]
                    deltas.append(delta)
                    pi.append(
                        policy.get_probability(actions[t + 1], states[t + 1]))
            tau = t - steps_count + 1
            if tau >= 0:
                '''
                Perform backup.
                '''
                Z = 1
                G = Q[tau]
                for k in range(tau, min(tau + steps_count, T)):
                    G = G + Z * deltas[k]
                    Z = discount_factor * Z * ((1 - sigma) * pi[k] + sigma)
                updated_value = Q[tau] + learning_rate * (G - Q[tau])
                policy.setQ(states[tau], actions[tau], updated_value)
            '''
            Stopping criterion. Since I can't do `for t in range(T)` and
            dynamically change T in the loop, the episode learning would just
            stop as soon as all backup steps have been performed.
            '''
            if tau == T - 1:
                break
            t += 1
        total_rewards.append(total_reward)
        if episode % 500 == 0:
            print('Success rate: ' + str(numpy.sum(total_rewards) / 500))
            total_rewards = []
    '''
    Plot rewards.
    '''
    print('Success rate: {}'.format(numpy.sum(total_rewards) / episodes_count))
