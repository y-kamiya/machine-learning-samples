from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import sys
import gym
from gym import wrappers
import torch
import retro
import numpy as np
from baselines.common.atari_wrappers import WarpFrame, FrameStack
from utils import SonicDiscretizer, RewardScaler, AllowBacktracking

from config import Config
from agent import Agent

NUM_STATES = 4
REWARD_RATE = 0.01

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


class Environment:
    def __init__(self, config):
        config.model_type = Config.MODEL_TYPE_CONV2D

        self.device = config.device
        self.num_steps = config.num_steps
        self.num_episodes = config.num_episodes

        self.env = make_env(self.num_steps)
        # self.env = wrappers.Monitor(env, '/tmp/gym/sonic_dqn', force=True)
        self.num_states = NUM_STATES
        self.num_actions = self.env.action_space.n

        self.is_saved = config.is_saved
        self.is_render = config.is_render

        self.agent = Agent(config, self.num_states, self.num_actions)

        self.data_path = config.data_path
        if self.data_path != Config.DATA_PATH_DEFAULT:
            print('load data from {0}'.format(self.data_path))
            self.get_model().load_state_dict(torch.load(self.data_path, map_location=config.device_name))

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
                    self.agent.observe(tensor_state, tensor_action, tensor_state_next, tensor_reward)
                    self.agent.learn()

                state = state_next
                tensor_state = tensor_state_next

                if done:
                    elapsed_time = round(time.time() - start_time, 3)
                    print('done episode {0}, time: {1}, info {2}'.format(episode, elapsed_time, info))
                    break

            if self.is_saved:
                print(self.data_path)
                torch.save(self.get_model().state_dict(), self.data_path)

        self.env.close()
        
if __name__ == '__main__':
    argv = sys.argv[1:]
    config = Config(argv)
    env = Environment(config)
    env.run()


