from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import gym
from gym import wrappers
import numpy as np
from collections import namedtuple 
import random
import torch
from torch import optim
import torch.nn.functional as F
from model import DuelingNetFC
from config import Config
from sum_tree import SumTree

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

ENV = 'CartPole-v0'
GAMMA = 0.99
MAX_STEPS = 200
NUM_EPISODE = 500
BATCH_SIZE = 32
CAPACITY = 10000
MEMORY_SIZE_TO_START_REPLY = 32

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.index = 0

    def push(self, _, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.index] = transition
        self.index = (self.index + 1) % self.capacity

    def sample(self, size):
        return (None, random.sample(self.memory, size))

    def update(self, idx, td_error):
        pass

    def __len__(self):
        return len(self.memory)

class PERMemory:
    epsilon = 0.0001
    alpha = 0.6
    size = 0

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, td_error):
        return (td_error + self.epsilon) ** self.alpha

    def push(self, td_error, transition):
        self.size += 1
        priority = self._getPriority(td_error)
        self.tree.add(priority, transition)

    def sample(self, size):
        list = []
        indexes = []
        for rand in np.random.uniform(0, self.tree.total(), size):
            (idx, _, data) = self.tree.get(rand)
            list.append(data)
            indexes.append(idx)

        return (indexes, list)

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

        # self.memory = ReplayMemory(CAPACITY)
        self.memory = PERMemory(CAPACITY)

        self.model = DuelingNetFC(num_states, num_actions).to(device=self.config.device)
        self.target_model = DuelingNetFC(num_states, num_actions).to(device=self.config.device)
        self.target_model.eval()
        print(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
    
    def _get_state_action_values(self, transitions):
        batch_size = len(transitions)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.uint8, device=self.config.device)

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        self.model.eval()

        next_state_values = torch.zeros(batch_size).to(self.config.device, dtype=torch.float32)

        next_states = [s for s in batch.next_state if s is not None]
        if len(next_states) != 0:
            non_final_next_state = torch.cat(next_states)
            best_actions = torch.argmax(self.model(non_final_next_state), dim=1, keepdim=True)
            next_state_values[non_final_mask] = self.target_model(non_final_next_state).gather(1, best_actions).squeeze()

        expected_values = reward_batch + GAMMA * next_state_values.detach()

        values = torch.squeeze(self.model(state_batch).gather(1, action_batch))
        values.to(self.config.device, dtype=torch.float32)

        return (values, expected_values)

    def reply(self):
        if (len(self.memory) < MEMORY_SIZE_TO_START_REPLY):
            return

        indexes, transitions = self.memory.sample(BATCH_SIZE)
        values, expected_values = self._get_state_action_values(transitions)

        self.model.train()

        loss = F.smooth_l1_loss(values, expected_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if (indexes != None):
            for i, value in enumerate(values):
                td_error = expected_values[i].item() - value.item()
                self.memory.update(indexes[i], abs(td_error))

    def add_memory(self, transition):
        values, expected_values = self._get_state_action_values([transition])
        td_error = abs(values.item() - expected_values.item())
        self.memory.push(td_error, transition)

        if (len(self.memory) == MEMORY_SIZE_TO_START_REPLY):
            print('start reply from next step')

    def decide_action(self, state, episode):
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon < random.uniform(0, 1):
            self.model.eval()
            action = self.model(state).data.max(1)[1].view(1, 1)
        else:
            rand = random.randrange(self.num_actions)
            action = torch.tensor([[rand]], dtype=torch.long, device=self.config.device)

        return action

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

class Agent:
    def __init__(self, config, num_states, num_actions):
        self.config = config
        self.num_states = num_states
        self.num_actions = num_actions
        self.steps_accumulated = 0
        self.brain = Brain(config, num_states, num_actions)

    def learn(self):
        self.brain.reply()

    def get_action(self, state, step):
        return self.brain.decide_action(state, step)

    def observe(self, state, action, state_next, reward):
        self.brain.add_memory(Transition(state, action, state_next, reward))

    def update_target_model(self):
        self.steps_accumulated += 1

        if self.config.num_steps_to_update_target <= self.steps_accumulated:
            self.steps_accumulated = 0
            self.brain.update_target_model()
            return
        
class Environment:
    def __init__(self, config):
        print(config.device)
        self.config = config
        self.env = gym.make(ENV)
        # self.env = wrappers.Monitor(self.env, '/tmp/gym/cartpole_dqn', force=True)
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        self.agent = Agent(config, self.num_states, self.num_actions)
        self.total_step = np.zeros(10)

    def is_success_episode(self, step):
        return 195 <= step

    def run_episode(self, episode):
        start_time = time.time()
        observation = self.env.reset()
        state = torch.from_numpy(observation).to(self.config.device, dtype=torch.float32).unsqueeze(0)

        for step in range(MAX_STEPS):
            action = self.agent.get_action(state, episode)

            observation_next, _, done, _ = self.env.step(action.item())

            if done:
                state_next = None
                self.total_step = np.hstack((self.total_step[1:], step + 1))
                if self.is_success_episode(step):
                    reward = torch.tensor([1.0], dtype=torch.float32, device=self.config.device)
                else:
                    reward = torch.tensor([-1.0], dtype=torch.float32, device=self.config.device)

            else:
                reward = torch.tensor([0.0], dtype=torch.float32, device=self.config.device)
                state_next = torch.from_numpy(observation_next).to(self.config.device, dtype=torch.float32).unsqueeze(0)

            self.agent.observe(state, action, state_next, reward)
            self.agent.learn()
            self.agent.update_target_model();

            state = state_next

            if done:
                elapsed_time = round(time.time() - start_time, 3)
                print('episode: {0}, steps: {1}, mean steps {2}, time: {3}'.format(episode, step, self.total_step.mean(), elapsed_time))
                return self.is_success_episode(step)

    def run(self):
        complete_episodes = 0

        for episode in range(NUM_EPISODE):
            if 10 <= complete_episodes:
                print('success 10 times in sequence, total episode: {0}'.format(episode))
                break

            is_success = self.run_episode(episode)
            if is_success:
                complete_episodes += 1
            else:
                complete_episodes = 0
                    
        self.env.close()
        
if __name__ == '__main__':
    argv = sys.argv[1:]
    config = Config(argv)

    for i in range(config.num_epochs):
        env = Environment(config)
        env.run()

