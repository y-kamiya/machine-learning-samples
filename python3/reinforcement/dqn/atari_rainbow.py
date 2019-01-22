from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import gym
import numpy as np
import torch
from utils import make_atari, wrap_deepmind
# from baselines.common.atari_wrappers import make_atari, wrap_deepmind

from config import Config
from agent import Agent

class Environment:
    def __init__(self, config):
        config.model_type = Config.MODEL_TYPE_CONV2D

        print(config.device)
        self.config = config
        self.env = wrap_deepmind(make_atari(config.env), frame_stack=True)
        self.num_states = self.env.observation_space.shape[-1]
        self.num_actions = self.env.action_space.n
        self.agent = Agent(config, self.num_states, self.num_actions, self.config.num_atoms)
        self.total_step = np.zeros(100)

        self.data_path = config.data_path
        if self.data_path != Config.DATA_PATH_DEFAULT:
            self.agent.load_model()

    def prepro(self, observation):
        ret = np.zeros((4, 84, 84))
        ret[0] = observation[:, :, 0]
        ret[1] = observation[:, :, 1]
        ret[2] = observation[:, :, 2]
        ret[3] = observation[:, :, 3]
        return ret

    def run_episode(self, episode, steps_accumulated=0):
        start_time = time.time()
        total_reward = 0
        observation = self.prepro(self.env.reset())
        state = torch.from_numpy(observation).to(self.config.device, dtype=torch.uint8).unsqueeze(0)

        for step in range(self.config.num_steps):
            if self.config.is_render:
                time.sleep(0.064)
                self.env.render()

            action = self.agent.get_action(state, step + steps_accumulated)

            observation_next, reward, done, _ = self.env.step(action.item())

            if done:
                state_next = None
                self.total_step = np.hstack((self.total_step[1:], step + 1))
            else:
                state_next = self.prepro(observation_next)
                state_next = torch.from_numpy(state_next).to(self.config.device, dtype=torch.uint8).unsqueeze(0)

            total_reward += reward
            reward = torch.tensor([reward], dtype=torch.uint8, device=self.config.device)

            if not self.config.is_render:
                self.agent.observe(state, action, state_next, reward)
                if step % self.config.replay_interval == 0:
                    self.agent.learn(episode)

            state = state_next

            if done:
                elapsed_time = round(time.time() - start_time, 3)
                print('episode: {0}, steps: {1}, mean steps {2}, time: {3}, reward: {4}'.format(episode, step, self.total_step.mean(), elapsed_time, total_reward))
                return step + 1

        return self.config.num_steps

    def run(self):
        if not self.config.is_render:
            steps = 0
            while True:
                steps += self.run_episode(-1)
                if self.config.steps_learning_start <= steps:
                    break

        steps = 0
        for episode in range(self.config.num_episodes):
            steps += self.run_episode(episode, steps)

        self.env.close()

        if self.config.is_saved:
            self.agent.save_model()
        
if __name__ == '__main__':
    argv = sys.argv[1:]
    config = Config(argv)
    print(" ".join(sys.argv))

    for i in range(config.num_epochs):
        env = Environment(config)
        env.run()


