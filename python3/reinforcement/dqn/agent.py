from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import numpy as np
from collections import namedtuple 
import copy
import random
import torch
from torch import optim
import torch.nn.functional as F

from model import DuelingNetFC, DuelingNetConv2d
from config import Config
from sum_tree import SumTree

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

GAMMA = 0.99
BATCH_SIZE = 32

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.index = 0

    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.index] = transition
        self.index = (self.index + 1) % self.capacity

    def sample(self, size):
        return (None, random.sample(self.memory, size), None)

    def update(self, idx, td_error):
        pass

    def __len__(self):
        return len(self.memory)

class PERMemory:
    epsilon = 0.0001
    alpha = 0.6
    beta = 0.4
    STEPS_TO_REACH_BETA_MAX = 1000000
    size = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = SumTree(capacity)

    def _getPriority(self, td_error):
        return (td_error + self.epsilon) ** self.alpha

    def push(self, transition):
        self.size += 1

        priority = self.tree.max()
        if priority <= 0:
            priority = 1

        self.tree.add(priority, transition)

    def sample(self, size):
        list = []
        indexes = []
        weights = np.empty(size, dtype='float32')
        total = self.tree.total()
        for i, rand in enumerate(np.random.uniform(0, total, size)):
            (idx, priority, data) = self.tree.get(rand)
            list.append(data)
            indexes.append(idx)
            weights[i] = (self.capacity * priority / total) ** (-self.beta)

        beta = self.beta + 1 / self.STEPS_TO_REACH_BETA_MAX
        self.beta = beta if beta < 1 else 1.0

        return (indexes, list, weights / weights.max())

    def update(self, idx, td_error):
        priority = self._getPriority(td_error)
        self.tree.update(idx, priority)

    def __len__(self):
        return self.size

class Brain:
    def __init__(self, config, num_states, num_actions):
        self.config = config
        self.num_states = num_states
        self.num_actions = num_actions

        capacity = config.reply_memory_capacity
        self.memory = PERMemory(capacity) if config.use_per else ReplayMemory(capacity)
        self.multi_step_transitions = []

        self.model = self._create_model(config, num_states, num_actions)
        self.target_model = copy.deepcopy(self.model)
        self.target_model.eval()
        print(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
    
    def _create_model(self, config, num_states, num_actions):
        use_noisy = self.config.use_noisy_network
        if config.model_type == Config.MODEL_TYPE_CONV2D:
            return DuelingNetConv2d(num_states, num_actions, use_noisy).to(device=config.device)

        return DuelingNetFC(num_states, num_actions, use_noisy).to(device=config.device)

    def _get_state_action_values(self, transitions):
        batch_size = len(transitions)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.uint8, device=self.config.device)

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        next_state_values = torch.zeros(batch_size).to(self.config.device, dtype=torch.float32)

        next_states = [s for s in batch.next_state if s is not None]
        if len(next_states) != 0:
            with torch.no_grad():
                non_final_next_state = torch.cat(next_states)
                best_actions = torch.argmax(self.model(non_final_next_state), dim=1, keepdim=True)
                next_state_values[non_final_mask] = self.target_model(non_final_next_state).gather(1, best_actions).squeeze()

        gamma = GAMMA ** self.config.num_multi_step_reward
        expected_values = reward_batch + gamma * next_state_values

        with torch.set_grad_enabled(self.model.training):
            values = torch.squeeze(self.model(state_batch).gather(1, action_batch))
            values.to(self.config.device, dtype=torch.float32)

        return (values, expected_values)

    def loss(self, input, target, weights):
        if self.config.use_IS:
            loss = torch.abs(target - input) * torch.from_numpy(weights).to(device=self.config.device)
            return loss.mean()

        return F.smooth_l1_loss(input, target)

    def reply(self):
        if len(self.memory) < self.config.steps_learning_start:
            return

        self.model.train()

        indexes, transitions, weights = self.memory.sample(BATCH_SIZE)
        values, expected_values = self._get_state_action_values(transitions)

        loss = self.loss(values, expected_values, weights)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if (indexes != None):
            for i, value in enumerate(values):
                td_error = abs(expected_values[i].item() - value.item())
                self.memory.update(indexes[i], td_error)

    def add_memory(self, transition):
        if 1 < self.config.num_multi_step_reward:
            transition = self._get_multi_step_transition(transition)

        if transition == None:
            return

        self.memory.push(transition)

        if len(self.memory) == self.config.steps_learning_start:
            print('start reply from next step')

    def _get_multi_step_transition(self, transition):
        self.multi_step_transitions.append(transition)
        if len(self.multi_step_transitions) < self.config.num_multi_step_reward:
            return None

        nstep_reward = 0
        for i in range(self.config.num_multi_step_reward):
            r = self.multi_step_transitions[i].reward
            nstep_reward += r * GAMMA ** i

        state, action, _, _ = self.multi_step_transitions.pop(0)

        return Transition(state, action, transition.next_state, nstep_reward)

    def decide_action(self, state, episode):
        epsilon = 0.5 * (1 / (episode + 1 + 0.001))

        if self.config.use_noisy_network or epsilon < random.uniform(0, 1):
            self.model.eval()
            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1, 1)
        else:
            rand = random.randrange(self.num_actions)
            action = torch.tensor([[rand]], dtype=torch.long, device=self.config.device)

        return action

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        if self.config.is_saved:
            self.save_model()

    def load_model(self):
        print('load data from {0}'.format(self.config.data_path))
        self.model.load_state_dict(torch.load(self.config.data_path, map_location=self.config.device_name))

    def save_model(self):
        path = self.config.data_path
        if os.path.isdir('data'):
            path = 'data/{0}'.format(self.config.data_path)

        print('save model parameters to {0}'.format(path))
        torch.save(self.model.state_dict(), path)

class Agent:
    def __init__(self, config, num_states, num_actions):
        self.config = config
        self.num_states = num_states
        self.num_actions = num_actions
        self.steps_accumulated = 0
        self.brain = Brain(config, num_states, num_actions)

    def learn(self):
        self.brain.reply()
        self._update_target_model();

    def get_action(self, state, episode):
        return self.brain.decide_action(state, episode)

    def observe(self, state, action, state_next, reward):
        self.brain.add_memory(Transition(state, action, state_next, reward))

    def load_model(self):
        self.brain.load_model()

    def save_model(self):
        self.brain.save_model()

    def _update_target_model(self):
        self.steps_accumulated += 1

        if self.config.num_steps_to_update_target <= self.steps_accumulated:
            self.steps_accumulated = 0
            self.brain.update_target_model()
            return
        
