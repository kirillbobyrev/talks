'''
Implements Q(\sigma) algorithm as described in "Multi-step Reinforcement
Learning: A Unifying Algorithm" (https://arxiv.org/abs/1703.01327) paper.
'''

import numpy


class EpsilonGreedyQPolicy(object):
    def __init__(self, states_count, actions_count, exploration_rate=0.1):
        assert(0 < exploration_rate and exploration_rate <= 1)
        self.states_count = states_count
        self.actions_count = actions_count
        self.Q = numpy.random.rand(states_count, actions_count)
        self.exploration_rate = exploration_rate
        print(self.Q)

    def get_probability(self, action, state):
        probabilities = self.Q[state] / numpy.sum(self.Q[state])
        return probabilities[action]

    def sample_action(self, state):
        return numpy.random.choice(range(self.actions_count),
                                   p=self.Q[state] / numpy.sum(self.Q[state]))

    def getQ(self, state, action):
        return self.Q[state][action]

    def setQ(self, state, action, value):
        self.Q[state][action] = value


def Q_sigma(environment, episodes_count=100, steps_number=4, sigma=0.5,
            discount_factor=1.0, exploration_rate=0.1, learning_rate=0.2):
    assert(0 < learning_rate and learning_rate <= 1)
    assert(0 < exploration_rate and exploration_rate <= 1)
    policy = EpsilonGreedyQPolicy(environment.observation_space.n,
                                  environment.action_space.n,
                                  exploration_rate)

    for episode in range(episodes_count):
        states = [environment.reset()]
        actions = [policy.sample_action(states[0])]
        deltas = []
        Q = [policy.getQ(states[0], actions[0])]
        # Storing 0 to the \pi_{t = 0} helps to make the backup more concise.
        pi = [0]

        t = 0
        T = numpy.infty
        while True:
            if t < T:
                current_state, reward, done, _ = \
                        environment.step(actions[t])
                states.append(current_state)
                if done:
                    T = t + 1
                    delta = reward - Q[t]
                else:
                    actions.append(policy.sample_action(states[t]))
                    Q.append(policy.getQ(states[t + 1], actions[t + 1]))
                    # Calculate \delta_t and store it.
                    pure_expectation = 0
                    for action in range(environment.action_space.n):
                        pure_expectation += policy.get_probability(action,
                            states[t + 1]) * policy.getQ(states[t + 1], action)
                    pure_expectation *= discount_factor * (1 - sigma)
                    delta = reward + discount_factor * sigma * Q[t + 1] + \
                            pure_expectation - Q[t]
                    pi.append(policy.get_probability(actions[t + 1],
                                                     states[t + 1]))
            tau = t - steps_number + 1
            if tau >= 0:
                '''
                Perform backup.
                '''
                pass
            '''
            Stopping criterion. Since I can't do `for t in range(T)` and
            dynamically change T in the loop, the episode learning would just
            stop as soon as all backup steps have been performed.
            '''
            Z = 1
            G = Q[tau]
            for k in range(min(tau + n - 1, T - 1)):
                G = G + Z * delta[k]
                Z = discount_factor * Z((1 - sigma) * pi[k + 1] + sigma)
            updated_value = policy.getQ(states[tau], actions[tau]) + \
                    learning_rate * (G - policy.getQ(states[tau], actions[tau]))
            policy.setQ(states[tau], actions[tau], updated_value)
            if tau == T - 1:
                break
            t += 1
