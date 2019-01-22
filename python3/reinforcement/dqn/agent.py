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

from model import *
from config import Config
from sum_tree import SumTree

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

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

    def sample(self, size, _):
        return (None, random.sample(self.memory, size), None)

    def update(self, idx, td_error):
        pass

    def __len__(self):
        return len(self.memory)

class PERMemory:
    EPSILON = 0.0001
    ALPHA = 0.5
    BETA = 0.4
    size = 0

    def __init__(self, config, capacity):
        self.config = config
        self.capacity = capacity
        self.tree = SumTree(capacity)

    def _getPriority(self, td_error):
        return (td_error + self.EPSILON) ** self.ALPHA

    def push(self, transition):
        self.size += 1

        priority = self.tree.max()
        if priority <= 0:
            priority = 1

        self.tree.add(priority, transition)

    def sample(self, size, episode):
        list = []
        indexes = []
        weights = np.empty(size, dtype='float32')
        total = self.tree.total()
        beta = self.BETA + (1 - self.BETA) * episode / self.config.num_episodes
        beta = min(1.0, beta)

        for i, rand in enumerate(np.random.uniform(0, total, size)):
            (idx, priority, data) = self.tree.get(rand)
            list.append(data)
            indexes.append(idx)
            weights[i] = (self.capacity * priority / total) ** (-beta)

        return (indexes, list, weights / weights.max())

    def update(self, idx, td_error):
        priority = self._getPriority(td_error)
        self.tree.update(idx, priority)

    def __len__(self):
        return self.size

class Brain:
    def __init__(self, config, num_states, num_actions, num_atoms):
        self.config = config
        self.num_states = num_states
        self.num_actions = num_actions

        capacity = config.replay_memory_capacity
        self.memory = PERMemory(config, capacity) if config.use_per else ReplayMemory(capacity)
        self.multi_step_transitions = []

        self.model = self._create_model(config, num_states, num_actions)
        self.target_model = copy.deepcopy(self.model)
        self.target_model.eval()

        self.Vmax = self.config.categorical_v
        self.Vmin = self.Vmax * (-1)
        if num_atoms != 1:
            self.delta_z = (self.Vmax - self.Vmin) / (num_atoms - 1)
            self.support = torch.Tensor([self.Vmin + i * self.delta_z for i in range(num_atoms)]).to(device=config.device)

        print(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate, eps=self.config.adam_epsilon)
    
    def _create_model(self, config, num_states, num_actions):
        use_noisy = self.config.use_noisy_network
        num_atoms = self.config.num_atoms
        if config.model_type == Config.MODEL_TYPE_CONV2D:
            return DuelingNetConv2d(num_states, num_actions, num_atoms, use_noisy).to(device=config.device)

        return DuelingNetFC(num_states, num_actions, num_atoms, use_noisy).to(device=config.device)

    def _get_state_action_values(self, transitions):
        batch_size = len(transitions)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.uint8, device=self.config.device)

        state_batch = torch.cat(batch.state).to(torch.float32)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward).to(torch.float32)
        
        next_state_values = torch.zeros(batch_size).to(self.config.device, dtype=torch.float32)

        next_states = [s for s in batch.next_state if s is not None]
        if len(next_states) != 0:
            with torch.no_grad():
                non_final_next_state = torch.cat(next_states).to(torch.float32)

                Q = self._get_Q(self.model, non_final_next_state)
                best_actions = torch.argmax(Q, dim=1, keepdim=True)

                Q_target = self._get_Q(self.target_model, non_final_next_state)
                next_state_values[non_final_mask] = Q_target.gather(1, best_actions).squeeze()

        gamma = self.config.gamma ** self.config.num_multi_step_reward
        expected_values = reward_batch + gamma * next_state_values

        with torch.set_grad_enabled(self.model.training):
            Q = self._get_Q(self.model, state_batch)
            values = torch.squeeze(Q.gather(1, action_batch))
            values.to(self.config.device, dtype=torch.float32)

        return (values, expected_values)

    def _get_Q(self, model, model_input):
        model.reset_noise()

        if not self.config.use_categorical:
            return model(model_input)

        model_output = model(model_input, ApplySoftmax.NORMAL)

        return torch.sum(model_output * self.support, dim=2)

    def loss(self, input, target, weights):
        if self.config.use_IS:
            loss = torch.abs(target - input) * torch.from_numpy(weights).to(device=self.config.device)
            return loss.mean()

        return F.smooth_l1_loss(input, target)

    def loss_categorical(self, transitions):
        num_atoms = self.config.num_atoms
        batch_size = len(transitions)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.uint8, device=self.config.device)

        state_batch = torch.cat(batch.state).to(torch.float32)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward).to(torch.float32)
        
        next_states = [s for s in batch.next_state if s is not None]

        with torch.no_grad():
            non_final_next_state = torch.cat(next_states).to(torch.float32)

            best_actions = self._get_Q(self.model, non_final_next_state).argmax(dim=1)

            self.target_model.reset_noise()
            p_next = self.target_model(non_final_next_state, ApplySoftmax.NORMAL)

            p_next_best = torch.zeros(0).to(self.config.device, dtype=torch.float32).new_full((batch_size, num_atoms), 1.0 / num_atoms)
            p_next_best[non_final_mask] = p_next[range(len(non_final_next_state)), best_actions]

            gamma = torch.zeros(batch_size, num_atoms).to(self.config.device)
            gamma[non_final_mask] = self.config.gamma ** self.config.num_multi_step_reward

            Tz = (reward_batch.unsqueeze(1) + gamma * self.support.unsqueeze(0)).clamp(self.Vmin, self.Vmax)
            b = (Tz - self.Vmin) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            l[(l == u) * (0 < l)] -= 1
            # the values in l has already changed, so same index is not processed for u
            u[(l == u) * (u < num_atoms - 1)] += 1

            m = torch.zeros(batch_size, num_atoms).to(self.config.device, dtype=torch.float32)
            offset = torch.linspace(0, ((batch_size-1) * num_atoms), batch_size).unsqueeze(1).expand(batch_size, num_atoms).to(l)
            m.view(-1).index_add_(0, (l + offset).view(-1), (p_next_best * (u.float() - b)).view(-1))
            m.view(-1).index_add_(0, (u + offset).view(-1), (p_next_best * (b - l.float())).view(-1))

        self.model.reset_noise()
        log_p = self.model(state_batch, ApplySoftmax.LOG)
        log_p_a = log_p[range(batch_size), action_batch.squeeze()]

        return -torch.sum(m * log_p_a, dim=1)

    def replay(self, episode):
        if len(self.memory) < self.config.steps_learning_start:
            return

        self.model.train()

        indexes, transitions, weights = self.memory.sample(self.config.batch_size, episode)

        if self.config.use_categorical:
            losses = self.loss_categorical(transitions)
            self._update_memory(indexes, losses)
            loss = (losses * torch.from_numpy(weights).to(losses)).mean() if self.config.use_IS else losses.mean()
        else:
            values, expected = self._get_state_action_values(transitions)
            with torch.no_grad():
                self._update_memory(indexes, torch.abs(expected - values))
            loss = self.loss(values, expected, weights)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _update_memory(self, indexes, values):
        if (indexes == None):
            return

        for i, index in enumerate(indexes):
            self.memory.update(index, values[i].item())


    def add_memory(self, transition):
        if 1 < self.config.num_multi_step_reward:
            transition = self._get_multi_step_transition(transition)

        if transition == None:
            return

        self.memory.push(transition)

        if len(self.memory) == self.config.steps_learning_start:
            print('start replay from next step')

    def _get_multi_step_transition(self, transition):
        self.multi_step_transitions.append(transition)
        if len(self.multi_step_transitions) < self.config.num_multi_step_reward:
            return None

        next_state = transition.next_state
        nstep_reward = 0
        for i in range(self.config.num_multi_step_reward):
            r = self.multi_step_transitions[i].reward
            nstep_reward += r * self.config.gamma ** i

            if self.multi_step_transitions[i].next_state is None:
                next_state = None
                break

        state, action, _, _ = self.multi_step_transitions.pop(0)

        return Transition(state, action, next_state, nstep_reward)

    def decide_action(self, state, step):
        start = self.config.epsilon_start;
        end = self.config.epsilon_end;
        epsilon = start + (end - start) * step / self.config.epsilon_end_step
        epsilon = max(end, min(start, epsilon))

        if self.config.use_noisy_network or epsilon < random.uniform(0, 1):
            self.model.eval()
            with torch.no_grad():
                Q = self._get_Q(self.model, state.float())
                action = Q.argmax().view(1,1)
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
    def __init__(self, config, num_states, num_actions, num_atoms):
        self.config = config
        self.num_states = num_states
        self.num_actions = num_actions
        self.steps_accumulated = 0
        self.brain = Brain(config, num_states, num_actions, num_atoms)

    def learn(self, episode):
        self.brain.replay(episode)
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
        
