from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
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
    def __init__(self, num_states, num_actions, is_noisy=False):
        super(DuelingNetFC, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions

        self.fc1 = nn.Linear(self.num_states, 32)
        if is_noisy:
            self.fcV1 = FactorizedNoisy(32, 32)
            self.fcA1 = FactorizedNoisy(32, 32)
            self.fcV2 = FactorizedNoisy(32, 1)
            self.fcA2 = FactorizedNoisy(32, self.num_actions)
        else:
            self.fcV1 = nn.Linear(32, 32)
            self.fcA1 = nn.Linear(32, 32)
            self.fcV2 = nn.Linear(32, 1)
            self.fcA2 = nn.Linear(32, self.num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        V = self.fcV2(F.relu(self.fcV1(x)))
        A = self.fcA2(F.relu(self.fcA1(x)))

        averageA = A.mean(1).unsqueeze(1)
        return V.expand(-1, self.num_actions) + (A - averageA.expand(-1, self.num_actions))

class NetConv2d(nn.Module):
    def __init__(self, num_states, num_actions):
        super(NetConv2d, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(num_states, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(21 * 21 * 64, 256)
        self.fc2 = nn.Linear(256, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view([-1, 21 * 21 * 64])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.4, training=self.training)
        return self.fc2(x)

class DuelingNetConv2d(nn.Module):
    def __init__(self, num_states, num_actions, is_noisy=False):
        super(DuelingNetConv2d, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(num_states, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
        if is_noisy:
            # 7 * 7 * 64 = 3136
            self.fcV1 = FactorizedNoisy(3136, 256)
            self.fcA1 = FactorizedNoisy(3136, 256)
            self.fcV2 = FactorizedNoisy(256, 1)
            self.fcA2 = FactorizedNoisy(256, num_actions)
        else:
            self.fcV1 = nn.Linear(3136, 256)
            self.fcA1 = nn.Linear(3136, 256)
            self.fcV2 = nn.Linear(256, 1)
            self.fcA2 = nn.Linear(256, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view([-1, 3136])
        V = self.fcV2(F.relu(self.fcV1(x)))
        A = self.fcA2(F.relu(self.fcA1(x)))

        averageA = A.mean(1).unsqueeze(1)
        return V.expand(-1, self.num_actions) + (A - averageA.expand(-1, self.num_actions))

class FactorizedNoisy(nn.Module):
    def __init__(self, in_features, out_features):
        super(FactorizedNoisy, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.u_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigma_w  = nn.Parameter(torch.Tensor(out_features, in_features))
        self.u_b = nn.Parameter(torch.Tensor(out_features))
        self.sigma_b = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.u_w.size(1))
        self.u_w.data.uniform_(-stdv, stdv)
        self.u_b.data.uniform_(-stdv, stdv)

        initial_sigma = 0.5 * stdv
        self.sigma_w.data.fill_(initial_sigma)
        self.sigma_b.data.fill_(initial_sigma)

    def forward(self, x):
        rand_in = self._f(torch.randn(1, self.in_features, device=self.u_w.device))
        rand_out = self._f(torch.randn(self.out_features, 1, device=self.u_w.device))
        epsilon_w = torch.matmul(rand_out, rand_in)
        epsilon_b = rand_out.squeeze()

        w = self.u_w + self.sigma_w * epsilon_w
        b = self.u_b + self.sigma_b * epsilon_b
        return F.linear(x, w, b)

    def _f(self, x):
        return torch.sign(x) * torch.sqrt(torch.abs(x))

