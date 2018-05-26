from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple 
import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable

BATCH_SIZE = 128
CAPACITY = 10000
GAMMA = 0.99

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.index = 0

    def push(self, state, action, state_next, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.index] = Transition(state, action, state_next, reward)
        self.index = (self.index + 1) % self.capacity

    def sample(self, size):
        return random.sample(self.memory, size)

    def __len__(self):
        return len(self.memory)

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

class Brain:
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions

        self.memory = ReplayMemory(CAPACITY)

        self.model = Net(num_states, num_actions)
        print(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
    def reply(self):
        if (len(self.memory) < BATCH_SIZE):
            return

        transitions = self.memory.sample(BATCH_SIZE)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))

        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))
        non_final_next_state = Variable(torch.cat([s for s in batch.next_state if s is not None]))
        
        self.model.eval()

        state_action_values = torch.squeeze(self.model(state_batch).gather(1, action_batch))

        next_state_values = Variable(torch.zeros(BATCH_SIZE).type(torch.FloatTensor))
        next_state_values[non_final_mask] = self.model(non_final_next_state).data.max(1)[0]

        expected_state_action_values = reward_batch + GAMMA * next_state_values
        
        self.model.train()

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decide_action(self, state, episode):
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon < random.uniform(0, 1):
            self.model.eval()
            action = self.model(Variable(state)).data.max(1)[1].view(1, 1)
        else:
            action = torch.LongTensor([[random.randrange(self.num_actions)]])

        return action

class Agent:
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions
        self.brain = Brain(num_states, num_actions)

    def update_q_function(self):
        self.brain.reply()

    def get_action(self, state, step):
        return self.brain.decide_action(state, step)

    def memory(self, state, action, state_next, reward):
        return self.brain.memory.push(state, action, state_next, reward)
        

