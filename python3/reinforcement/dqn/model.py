from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn
import torch.nn.functional as F

class NetFC(nn.Module):
    def __init__(self, num_states, num_actions):
        super(NetFC, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions

        self.fc1 = nn.Linear(self.num_states, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, self.num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DuelingNetFC(nn.Module):
    def __init__(self, num_states, num_actions):
        super(DuelingNetFC, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions

        self.fc1 = nn.Linear(self.num_states, 32)
        self.fc2 = nn.Linear(32, 32)

        self.fcV = nn.Linear(32, 1)
        self.fcA = nn.Linear(32, self.num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        V = self.fcV(x)
        A = self.fcA(x)

        averageA = A.mean(1).unsqueeze(1)
        return V.expand(-1, self.num_actions) + (A - averageA.expand(-1, self.num_actions))
