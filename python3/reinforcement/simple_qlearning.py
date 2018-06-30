from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

NUM_STATE = 7
NUM_ACTION = 2
NUM_EPISODE = 100
NUM_STEP = 2
GOAL = 6
ETA = 0.1
GAMMA = 0.9
EPSILON = 0.5

def get_next_action(state, q):
    if np.random.rand() < EPSILON:
        return np.random.choice([0, 1])
    else:
        current = q[state]
        if current[0] < current[1]:
            return 1
        return 0

def get_next_state(state, next_action):
    return 2 * state + next_action + 1

def update_q(state, action, next_state, q):
    if next_state == GOAL:
        print('goal')
        return q[state, action] + ETA * (1 - q[state, action])
    else:
        return q[state, action] + ETA * (GAMMA * np.nanmax(q[next_state, :]) - q[state, action])

class Environment:
    def __init__(self):
        pass

    def run(self):
        # q = np.random.rand(NUM_STATE, NUM_ACTION)
        q = np.empty([NUM_STATE, NUM_ACTION], np.float32)
        q.fill(0.5)
        for episode in range(NUM_EPISODE):
            state = 0

            for step in range(NUM_STEP):
                action = get_next_action(state, q)
                next_state = get_next_state(state, action)
                print("state: {0}, action: {1}, next:{2}".format(state, action, next_state))
                
                q[state, action] = update_q(state, action, next_state, q)

                state = next_state

            print("episode: {0}".format(episode))
            print("{0}".format(q))

if __name__ == '__main__':
    env = Environment()
    env.run()
