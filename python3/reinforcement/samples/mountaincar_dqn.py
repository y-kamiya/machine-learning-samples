from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
from gym import wrappers
import torch

from agent import Agent

ENV = 'MountainCar-v0'
NUM_EPISODE = 5000
MAX_STEPS = 300

class Environment():
    def __init__(self):
        env = gym.make(ENV)
        self.env = wrappers.Monitor(env, '/tmp/gym/mountaincar_dqn', force=True)
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        self.agent = Agent(self.num_states, self.num_actions)

    def run(self):
        complete_episodes = 0
        episode_final = False
        output = open('result.log', 'w')

        print(self.num_states, self.num_actions)
        for episode in range(NUM_EPISODE):
            observation = self.env.reset()
            state = torch.from_numpy(observation).type(torch.FloatTensor)
            state = torch.unsqueeze(state, 0)

            for step in range(MAX_STEPS):
                if episode_final:
                    self.env.render(mode='rgb_array')

                action = self.agent.get_action(state, episode)

                observation_next, _, done, _ = self.env.step(action.item())

                state_next = torch.from_numpy(observation_next).type(torch.FloatTensor)
                state_next = torch.unsqueeze(state_next, 0)

                reward = torch.FloatTensor([0.0])
                if done:
                    state_next = None
                    if 199 <= step:
                        reward = torch.FloatTensor([-1.0])
                        complete_episodes = 0
                    else:
                        reward = torch.FloatTensor([1.0])
                        complete_episodes = complete_episodes + 1

                self.agent.memory(state, action, state_next, reward)
                self.agent.update_q_function()

                state = state_next

                if done:
                    message = 'episode: {0}, step: {1}'.format(episode, step)
                    print(message)
                    output.write(message + '\n')
                    break

                if episode_final:
                    break

                if 10 <= complete_episodes:
                    print('success 10 times in sequence')
                    # episode_final = True

        self.env.close()
        output.close()

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import animation
# from gym import wrappers
#
# def display_frames_as_gif(frames):
#     """
#     Displays a list of frames as a gif, with controls
#     """
#     plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0),
#                dpi=72)
#     patch = plt.imshow(frames[0])
#     plt.axis('off')
#  
#     def animate(i):
#         patch.set_data(frames[i])
#  
#     anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames),
#                                    interval=50)
#  
#     anim.save('movie_cartpole_DQN.mp4')  # 動画のファイル名と保存です

if __name__ == '__main__':
    env = Environment()
    env.run()
    # frames = []
    # env = gym.make(ENV)
    # env = wrappers.Monitor(env, '/tmp/gym')
    # env.reset()
    # for _ in range(0, 200):
    #     frames.append(env.render(mode='rgb_array'))
    #     action = np.random.choice(2)
    #     observation, reward, done, info = env.step(action)
    #     if done:
    #         break
    #
    # env.close()
    # display_frames_as_gif(frames)

