from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable

NUM_STATE = 7
NUM_ACTION = 2
NUM_EPISODE = 100
NUM_STEP = 2
GOAL = 6
ETA = 0.1
GAMMA = 0.9
EPSILON = 0.5

class Net(nn.Module):
    def __init__(self, num_states, num_actions):
        super(Net, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions

        self.fc1 = nn.Linear(self.num_states, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, self.num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Environment:
    def __init__(self):
        self.model = Net(NUM_STATE, NUM_ACTION)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def get_next_state(self, state, next_action):
        return 2 * state + next_action + 1

    def create_input(self, state):
        e = torch.zeros([NUM_STATE])
        e[state] = 1
        return e

    def get_next_action(self, state):
        if np.random.rand() < EPSILON:
            return np.random.choice([0, 1])

        self.model.eval()

        return self.model(self.create_input(state)).argmax().item()

    def update_q(self, state, action, next_state):
        self.model.eval()

        variable = Variable(self.create_input(state))
        qvalue = self.model(variable)[action]

        next_variable = Variable(self.create_input(next_state))
        next_qvalue_max = self.model(next_variable).max()

        if next_state == GOAL:
            print('goal')
            target = qvalue + ETA * (1 - qvalue)
        else:
            target = qvalue + ETA * (GAMMA * next_qvalue_max - qvalue)

        self.model.train()

        loss = F.smooth_l1_loss(qvalue, Variable(target))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def run(self):
        for episode in range(NUM_EPISODE):
            state = 0

            for step in range(NUM_STEP):
                action = self.get_next_action(state)
                next_state = self.get_next_state(state, action)
                print("state: {0}, action: {1}, next:{2}".format(state, action, next_state))
                
                self.update_q(state, action, next_state)

                state = next_state

            print("episode: {0}".format(episode))

            self.model.eval()
            print(self.model(self.create_input(0)))
            print(self.model(self.create_input(1)))
            print(self.model(self.create_input(2)))

if __name__ == '__main__':
    env = Environment()
    env.run()
