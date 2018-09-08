from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import gym
from gym import wrappers
import numpy as np
import torch

from config import Config
from agent import Agent

ENV = 'CartPole-v0'
MAX_STEPS = 200
NUM_EPISODE = 500
MEMORY_SIZE_TO_START_REPLY = 1000

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
                return step + 1

        return MAX_STEPS

    def run(self):
        complete_episodes = 0

        steps = 0
        while True:
            steps += self.run_episode(0)
            if MEMORY_SIZE_TO_START_REPLY < steps:
                break

        for episode in range(NUM_EPISODE):
            if 10 <= complete_episodes:
                print('success 10 times in sequence, total episode: {0}'.format(episode))
                break

            steps = self.run_episode(episode)
            if self.is_success_episode(steps):
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

