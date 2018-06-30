from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

NUM_STATE = 8
NUM_ACTION = 4
NUM_STEPS = 4
NUM_EPISODE = 5000
NUM_HIDDEN_NODES = 32
LEARNING_RATE = 0.01
GAMMA = 0.99
GOAL = 8

class Net(nn.Module):
    def __init__(self, num_states, num_actions):
        super(Net, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions

        self.fc1 = nn.Linear(self.num_states, NUM_HIDDEN_NODES)
        self.fc2 = nn.Linear(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES)
        self.fc3 = nn.Linear(NUM_HIDDEN_NODES, self.num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x))

class Environment:
    def __init__(self):
        self.model = Net(NUM_STATE, NUM_ACTION)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    def create_input(self, state):
        array = np.zeros(NUM_STATE)
        array[state] = 1
        return torch.from_numpy(array).type(torch.FloatTensor)

    def get_next_state(self, state, action):
        if action == 0:
           s_next = state - 3
        elif action == 1:
           s_next = state + 1
        elif action == 2:
           s_next = state + 3
        elif action == 3:
           s_next = state - 1

        if s_next < 0 or NUM_STATE < s_next: 
            return state

        return s_next

    def get_next_action(self, state):
        self.model.eval()
        return self.model(self.create_input(state)).argmax().item()

    def discount_reward(self, rewards):
        discounted_rewards = np.zeros((rewards.size, NUM_ACTION))
        running_add = 0
        for i in range(rewards.size)[::-1]:
            running_add = running_add * GAMMA + rewards[i]
            for j in range(0, NUM_ACTION):
                discounted_rewards[i][j] = running_add

        return discounted_rewards

    def run_to_goal(self):
        state = 0
        history = []

        self.model.eval()

        for step in range(0, NUM_STEPS):
           output = self.model(self.create_input(state))
           props = output.data.numpy()

           action = np.random.choice(range(0, NUM_ACTION), p=props)

           next_state = self.get_next_state(state, action)

           ys = np.zeros(NUM_ACTION)
           ys[action] = 1

           history.append([state, action, 0.0, output, ys])

           if next_state == state:
               break;

           if next_state == GOAL:
               history[-1][2] = 1.0
               break

           state = next_state

        if history[-1][2] == 0.0:
           history[-1][2] = -1.0

        return history

    def update_policy(self, history, episode):
        self.model.train()

        rewards = np.zeros((len(history)))
        targets = np.zeros((len(history), NUM_ACTION))
        for i, entry in enumerate(history):
            rewards[i] = entry[2]
            targets[i] = entry[4]
            
        discounted_rewards = self.discount_reward(rewards)
        targets = targets * discounted_rewards

        targets.reshape(-1, NUM_ACTION)
        targets = torch.tensor(targets, dtype=torch.float32)

        self.optimizer.zero_grad()
        for i, entry in enumerate(history):
            # print(entry)
            loss = F.smooth_l1_loss(entry[3], targets[i])
            loss.backward()

        self.optimizer.step()
        

    def display_model(self, episode):
        print("episode: {0}".format(episode))
        for i in range(0, NUM_STATE):
            print(self.model(self.create_input(i)))

    def run(self):
        for episode in range(NUM_EPISODE):
            state = 0

            history = self.run_to_goal()
            self.update_policy(history, episode)

            self.model.eval()
            if episode % 10 == 0:
                self.display_model(episode)

if __name__ == '__main__':
    env = Environment()
    env.run()

