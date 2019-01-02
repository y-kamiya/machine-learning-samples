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
NUM_STEPS_TO_SUCCEED = 195
MEAN_STEPS_TO_SUCCEED = 150

class Environment:
    def __init__(self, config):
        print(config.device)
        self.config = config
        self.env = gym.make(ENV)
        # self.env = wrappers.Monitor(self.env, '/tmp/gym/cartpole_dqn', force=True)
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        self.agent = Agent(config, self.num_states, self.num_actions, config.num_atoms)
        self.total_step = np.zeros(100)

    def is_success_episode(self, step):
        return NUM_STEPS_TO_SUCCEED <= step

    def run_episode(self, episode, steps_accumulated=0):
        start_time = time.time()
        observation = self.env.reset()
        state = torch.from_numpy(observation).to(self.config.device, dtype=torch.float32).unsqueeze(0)

        for step in range(MAX_STEPS):
            action = self.agent.get_action(state, step + steps_accumulated)

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
            if step % self.config.replay_interval == 0:
                self.agent.learn(episode)

            state = state_next

            if done:
                elapsed_time = round(time.time() - start_time, 3)
                print('episode: {0}, steps: {1}, mean steps {2}, time: {3}'.format(episode, step, self.total_step.mean(), elapsed_time))
                return step + 1

        return MAX_STEPS

    def run(self):
        steps = 0
        while True:
            steps += self.run_episode(-1)
            if self.config.steps_learning_start < steps:
                break

        steps = 0
        for episode in range(self.config.num_episodes):
            if MEAN_STEPS_TO_SUCCEED <= self.total_step.mean():
                print('over {0} steps of average last 100 episodes, last episode: {1}, steps: {2}'.format(MEAN_STEPS_TO_SUCCEED, episode, steps))
                break

            steps += self.run_episode(episode, steps)

        self.env.close()
        
if __name__ == '__main__':
    argv = sys.argv[1:]
    config = Config(argv)
    print(" ".join(sys.argv))

    for i in range(config.num_epochs):
        env = Environment(config)
        env.run()

