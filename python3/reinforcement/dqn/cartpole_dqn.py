from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
from gym import wrappers
import numpy as np
from collections import namedtuple 

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

ENV = 'CartPole-v0'
GAMMA = 0.99
MAX_STEPS = 200
NUM_EPISODE = 500

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

import random
import torch
from torch import optim
import torch.nn.functional as F
from model import NetFC

BATCH_SIZE = 32
CAPACITY = 10000

class Brain:
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions

        self.memory = ReplayMemory(CAPACITY)

        self.model = NetFC(num_states, num_actions)
        print(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
    
    def reply(self):
        if (len(self.memory) < BATCH_SIZE):
            return

        transitions = self.memory.sample(BATCH_SIZE)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_state = torch.cat([s for s in batch.next_state if s is not None])
        
        self.model.eval()

        state_action_values = torch.squeeze(self.model(state_batch).gather(1, action_batch))

        next_state_values = torch.zeros(BATCH_SIZE).type(torch.FloatTensor)
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
            action = self.model(state).data.max(1)[1].view(1, 1)
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
        
class Environment:
    def __init__(self):
        env = gym.make(ENV)
        self.env = wrappers.Monitor(env, '/tmp/gym/cartpole_dqn', force=True)
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        self.agent = Agent(self.num_states, self.num_actions)
        self.total_step = np.zeros(10)

    def run(self):
        complete_episodes = 0

        for episode in range(NUM_EPISODE):
            observation = self.env.reset()
            state = torch.from_numpy(observation).type(torch.FloatTensor)
            state = torch.unsqueeze(state, 0)

            for step in range(MAX_STEPS):
                action = self.agent.get_action(state, episode)

                observation_next, _, done, _ = self.env.step(action.item())

                if done:
                    state_next = None
                    self.total_step = np.hstack((self.total_step[1:], step + 1))
                    if step < 195:
                        reward = torch.FloatTensor([-1.0])
                        complete_episodes = 0
                    else:
                        reward = torch.FloatTensor([1.0])
                        complete_episodes = complete_episodes + 1

                else:
                    reward = torch.FloatTensor([0.0])
                    state_next = torch.from_numpy(observation_next).type(torch.FloatTensor)
                    state_next = torch.unsqueeze(state_next, 0)

                self.agent.memory(state, action, state_next, reward)

                self.agent.update_q_function()

                state = state_next

                if done:
                    print('episode: {0}, steps: {1}, mean steps {2}'.format(episode, step, self.total_step.mean()))
                    break

                if 10 <= complete_episodes:
                    print('success 10 times in sequence')
                    
        self.env.close()
        
if __name__ == '__main__':
    env = Environment()
    env.run()

