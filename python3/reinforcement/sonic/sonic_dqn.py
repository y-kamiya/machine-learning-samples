from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import argparse
import sys
import gym
from gym import wrappers
import retro
import numpy as np
from collections import namedtuple 
from baselines.common.atari_wrappers import WarpFrame, FrameStack
from utils import SonicDiscretizer, RewardScaler, AllowBacktracking

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

GAMMA = 0.99
NUM_STEPS_DEFAULT = 2500
NUM_EPISODES_DEFAULT = 400
NUM_STATES = 4
REWARD_RATE = 0.01
DATA_PATH_DEFAULT = 'model_state_sonic_dqn.dat'

def make_env(num_steps, stack=True, scale_rew=True):
    """
    Create an environment with some standard wrappers.
    """
    env = retro.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1')
    env = gym.wrappers.TimeLimit(env, max_episode_steps=num_steps)
    env = SonicDiscretizer(env)
    env = AllowBacktracking(env)
    if scale_rew:
        env = RewardScaler(env, REWARD_RATE)
    env = WarpFrame(env)
    if stack:
        env = FrameStack(env, NUM_STATES)
    return env

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
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable

BATCH_SIZE = 32
SIZE_REPLY_START = 1000
CAPACITY = 10000
LEARNING_RATE = 0.0001

class Net(nn.Module):
    def __init__(self, num_states, num_actions):
        super(Net, self).__init__()
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
        x = self.fc1(x)
        x = F.dropout(x, p=0.4, training=self.training)
        return F.softmax(self.fc2(x))

class Brain:
    def __init__(self, num_states, num_actions, device):
        self.num_states = num_states
        self.num_actions = num_actions
        self.device = device

        self.memory = ReplayMemory(CAPACITY)

        self.model = Net(num_states, num_actions).to(device=device)
        print(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
    
    def reply(self):
        if (len(self.memory) < SIZE_REPLY_START):
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

        next_state_values = torch.zeros(BATCH_SIZE).to(self.device, dtype=torch.float32)
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
            action = torch.tensor([[random.randrange(self.num_actions)]], dtype=torch.int64, device=self.device)

        return action

class Agent:
    def __init__(self, num_states, num_actions, device):
        self.num_states = num_states
        self.num_actions = num_actions
        self.brain = Brain(num_states, num_actions, device)

    def update_q_function(self):
        self.brain.reply()

    def get_action(self, state, step):
        return self.brain.decide_action(state, step)

    def memory(self, state, action, state_next, reward):
        return self.brain.memory.push(state, action, state_next, reward)
        
class Environment:
    def __init__(self):
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device_name)
        self.num_steps = args.steps if args.steps != None else NUM_STEPS_DEFAULT
        self.num_episodes = args.episodes if args.episodes != None else NUM_EPISODES_DEFAULT

        self.env = make_env(self.num_steps)
        # self.env = wrappers.Monitor(env, '/tmp/gym/sonic_dqn', force=True)
        self.num_states = NUM_STATES
        self.num_actions = self.env.action_space.n
        self.agent = Agent(self.num_states, self.num_actions, self.device)

        data_path = args.path
        if data_path:
            print('load data from {0}'.format(data_path))
            self.get_model().load_state_dict(torch.load(data_path, map_location=device_name))
        self.data_path = data_path if data_path != None else DATA_PATH_DEFAULT

        self.is_saved = not args.nosave
        self.is_render = args.render

    def get_model(self):
        return self.agent.brain.model

    def prepro(self, I):
        ret = np.zeros((4, 84, 84))
        # I = I[::4,::4, :] # downsample by factor of 2
        ret[0] = I[:, :, 0]
        ret[1] = I[:, :, 1]
        ret[2] = I[:, :, 2]
        ret[3] = I[:, :, 3]
        return ret

    def run(self):
        for episode in range(self.num_episodes):
            observation = self.env.reset()
            state = self.prepro(observation)
            state_diff = np.zeros(4*84*84).reshape(4,84,84)
            tensor_state = torch.from_numpy(state_diff).to(self.device, dtype=torch.float32).unsqueeze(0)

            start_time = time.time()
            for step in range(self.num_steps):
                tensor_action = self.agent.get_action(tensor_state, episode)
                action = tensor_action.cpu().item()

                observation_next, reward, done, info = self.env.step(action)
                if self.is_render:
                    self.env.render()

                if done:
                    state_next = None
                    tensor_reward = torch.tensor([-REWARD_RATE], dtype=torch.float32, device=self.device)
                else:
                    state_next = self.prepro(observation_next)
                    state_diff = state_next - state
                    tensor_state_next = torch.from_numpy(state_diff).to(self.device, dtype=torch.float32).unsqueeze(0)
                    tensor_reward = torch.tensor([reward], dtype=torch.float32, device=self.device)

                if not self.is_render:
                    print('episode {0}, step {1}, action {2}, reward {3}'.format(episode, step, action, tensor_reward.item()))
                    if step % BATCH_SIZE == 0 or step == self.num_steps - 1:
                        self.agent.memory(tensor_state, tensor_action, tensor_state_next, tensor_reward)
                        self.agent.update_q_function()

                state = state_next
                tensor_state = tensor_state_next

                if done:
                    print('done')
                    break

            if self.is_saved:
                print(self.data_path)
                torch.save(self.get_model().state_dict(), self.data_path)

            print('episode {0} info {1}'.format(episode, info))
            print('episode {0} elapsed time {1} sec'.format(episode, time.time() - start_time))

        self.env.close()
        
if __name__ == '__main__':
    argv = sys.argv[1:]

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('-p', '--path', help='path to model data file')
    parser.add_argument('--nosave', action='store_true', help='model parameters are saved')
    parser.add_argument('--steps', type=int, help='step count')
    parser.add_argument('--episodes', type=int, help='episode count')
    parser.add_argument('--render', action='store_true', help='render game')
    args = parser.parse_args(argv)

    env = Environment()
    env.run()


