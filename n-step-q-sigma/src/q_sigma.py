'''
Implements Q(\sigma) algorithm as described in "Multi-step Reinforcement
Learning: A Unifying Algorithm" (https://arxiv.org/abs/1703.01327) paper.
'''

import numpy


def select_action(Q, state):
    return numpy.argmax(Q[state])


def policy(Q, action, state):
    '''
    Returns probability of taking action from state given Q function.
    '''
    policy_probabilities = Q[state] / numpy.sum(Q[state])
    return policy_probabilities[action]


def Q_sigma(environment, episodes_number=100, steps_number=1, sigma=0.5,
            discount_factor=1.0, epsilon=0.1, learning_rate=0.4):
    '''
    TODO: Document parameters and the function itself.
    '''
    Q = numpy.ones((environment.observation_space.high[0],
                         environment.observation_space.high[1],
                         environment.action_space.n))
    Q /= environment.action_space.n
    for episode in range(episodes_number):
        state = environment.reset()
        action = select_action(Q, state)
        states = [state]
        actions = [action]
        deltas = []
        T = 400000 # numpy.infty
        for t in range(T + steps_number):
            if t < T:
                state, reward, done, info = environment.step(actions[t])
                states.append(state)
                if done:
                    delta = reward - Q[states[t]][actions[t]]
                    deltas.append(delta)
                    T = t
                else:
                    action = select_action(Q, states[t + 1])
                    actions.append(action)
                    delta = reward + \
                            discount_factor * (sigma * \
                               Q[states[t + 1]][actions[t + 1]] + \
                               (1 - sigma) * get_value(Q, states[t])) - \
                            Q[states[t]][actions[t]]
                    deltas.append(delta)
            if t >= steps_number:
                sum = 0
                for k in range(t, numpy.min((t + steps_number, T))):
                    print(t)
                    print(t + steps_number)
                    print(T)
                    product = 1
                    for i in range(t + 1, k + 1):
                        product *= discount_factor * ((1 - sigma) * \
                                        policy(Q, action, state) + sigma)
                    sum += product
                n_step_return = Q[states[t]][actions[t]] + sum
                Q[states[t]][actions[t]] += learning_rate * (n_step_return - \
                                                Q[states[t]][actions[t]])
