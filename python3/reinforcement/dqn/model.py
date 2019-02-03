from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from enum import Enum
import math
import torch
from torch import nn
import torch.nn.functional as F

class ApplySoftmax(Enum):
    NONE = 0
    NORMAL = 1
    LOG = 2

class NetFC(nn.Module):
    def __init__(self, num_states, num_actions, num_atoms=1):
        super(NetFC, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions

        self.fc1 = nn.Linear(self.num_states, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, self.num_actions * num_atoms)

    def reset_noise(self):
        pass

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DuelingNetFC(nn.Module):
    def __init__(self, num_states, num_actions, num_atoms=1, is_noisy=False):
        super(DuelingNetFC, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.is_noisy = is_noisy
        self.num_atoms = num_atoms

        num_node_value = num_atoms
        num_node_advantage = num_actions * num_atoms

        self.fc1 = nn.Linear(self.num_states, 32)
        if is_noisy:
            self.fcV1 = FactorizedNoisy(32, 32)
            self.fcA1 = FactorizedNoisy(32, 32)
            self.fcV2 = FactorizedNoisy(32, num_node_value)
            self.fcA2 = FactorizedNoisy(32, num_node_advantage)
        else:
            self.fcV1 = nn.Linear(32, 32)
            self.fcA1 = nn.Linear(32, 32)
            self.fcV2 = nn.Linear(32, num_node_value)
            self.fcA2 = nn.Linear(32, num_node_advantage)

    def reset_noise(self):
        if not self.is_noisy:
            return

        self.fcV1.reset_noise()
        self.fcA1.reset_noise()
        self.fcV2.reset_noise()
        self.fcA2.reset_noise()

    def forward(self, x, apply_softmax=ApplySoftmax.NONE):
        x = F.relu(self.fc1(x))

        V = self.fcV2(F.relu(self.fcV1(x)))
        A = self.fcA2(F.relu(self.fcA1(x)))

        v = V.view(-1, 1, self.num_atoms)
        a = A.view(-1, self.num_actions, self.num_atoms)

        averageA = a.mean(1, keepdim=True)
        output = v.expand(-1, self.num_actions, self.num_atoms) + (a - averageA.expand(-1, self.num_actions, self.num_atoms))

        if apply_softmax == ApplySoftmax.NORMAL:
            return F.softmax(output, dim=2)

        if apply_softmax == ApplySoftmax.LOG:
            return F.log_softmax(output, dim=2)
        
        # num_atoms == 1 in this case
        return output.squeeze()

class NetConv2d(nn.Module):
    def __init__(self, num_states, num_actions, num_atoms=1):
        super(NetConv2d, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(num_states, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(21 * 21 * 64, 256)
        self.fc2 = nn.Linear(256, num_actions * num_atoms)

    def reset_noise(self):
        pass

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
    def __init__(self, num_states, num_actions, num_atoms=1, is_noisy=False, apply_softmax=ApplySoftmax.NONE):
        super(DuelingNetConv2d, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.is_noisy = is_noisy
        self.num_atoms = num_atoms
        self.apply_softmax = apply_softmax

        self.conv1 = nn.Conv2d(num_states, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        num_node_value = num_atoms
        num_node_advantage = num_actions * num_atoms

        hidden_size = 512

        if is_noisy:
            # 7 * 7 * 64 = 3136
            self.fcV1 = FactorizedNoisy(3136, hidden_size)
            self.fcA1 = FactorizedNoisy(3136, hidden_size)
            self.fcV2 = FactorizedNoisy(hidden_size, num_node_value)
            self.fcA2 = FactorizedNoisy(hidden_size, num_node_advantage)
        else:
            self.fcV1 = nn.Linear(3136, hidden_size)
            self.fcA1 = nn.Linear(3136, hidden_size)
            self.fcV2 = nn.Linear(hidden_size, num_node_value)
            self.fcA2 = nn.Linear(hidden_size, num_node_advantage)

    def reset_noise(self):
        if not self.is_noisy:
            return

        self.fcV1.reset_noise()
        self.fcA1.reset_noise()
        self.fcV2.reset_noise()
        self.fcA2.reset_noise()

    def forward(self, x, apply_softmax=ApplySoftmax.NONE):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view([-1, 3136])
        V = self.fcV2(F.relu(self.fcV1(x)))
        A = self.fcA2(F.relu(self.fcA1(x)))

        v = V.view(-1, 1, self.num_atoms)
        a = A.view(-1, self.num_actions, self.num_atoms)

        averageA = a.mean(1, keepdim=True)
        output = v.expand(-1, self.num_actions, self.num_atoms) + (a - averageA.expand(-1, self.num_actions, self.num_atoms))

        if apply_softmax == ApplySoftmax.NORMAL:
            return F.softmax(output, dim=2)

        if apply_softmax == ApplySoftmax.LOG:
            return F.log_softmax(output, dim=2)
        
        # num_atoms == 1 in this case
        return output.squeeze()

class FactorizedNoisy(nn.Module):
    def __init__(self, in_features, out_features):
        super(FactorizedNoisy, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.u_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigma_w  = nn.Parameter(torch.Tensor(out_features, in_features))
        self.u_b = nn.Parameter(torch.Tensor(out_features))
        self.sigma_b = nn.Parameter(torch.Tensor(out_features))
        self._reset_parameters()
        self.reset_noise()

    def _reset_parameters(self):
        stdv = 1. / math.sqrt(self.u_w.size(1))
        self.u_w.data.uniform_(-stdv, stdv)
        self.u_b.data.uniform_(-stdv, stdv)

        initial_sigma = 0.5 * stdv
        self.sigma_w.data.fill_(initial_sigma)
        self.sigma_b.data.fill_(initial_sigma)

    def reset_noise(self):
        rand_in = self._f(torch.randn(1, self.in_features, device=self.u_w.device))
        rand_out = self._f(torch.randn(self.out_features, 1, device=self.u_w.device))
        self.epsilon_w = torch.matmul(rand_out, rand_in)
        self.epsilon_b = rand_out.squeeze()

    def forward(self, x):
        w = self.u_w + self.sigma_w * self.epsilon_w
        b = self.u_b + self.sigma_b * self.epsilon_b
        return F.linear(x, w, b)

    def _f(self, x):
        return torch.sign(x) * torch.sqrt(torch.abs(x))

