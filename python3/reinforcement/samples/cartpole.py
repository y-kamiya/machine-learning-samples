from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

ENV = 'CartPole-v0'
NUM_DIGITIZE = 6
GAMMA = 0.99
ETA = 0.5
MAX_STEPS = 200
NUM_EPISODE = 1000

def display_frames_as_gif(frames):
    plt.figure(figsize=(frames[0].shape[1]/72.0, frames[0].shape[0]/72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anime = animation.FuncAnimaiton(plt.gcf(), animate, frames=len(frames), interval=50)

    anime.save('movie_cartpole.mp4')

class Agent:
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions
        self.brain = Brain(num_states, num_actions)

    def update_q_function(self, observation, action, reward, observation_next):
        self.brain.update_qtable(observation, action, reward, observation_next)

    def get_action(self, observation, step):
        return self.brain.decide_action(observation, step)
        
class Brain:
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions
        self.q_table = np.random.uniform(low=0, high=1, size=(num_states**NUM_DIGITIZE, num_actions))
    def bins(self, clip_min, clip_max, num):
        return np.linspace(clip_min, clip_max, num)[1:-1]

    def digitize_state(self, observation):
        cart_pos, cart_v, pole_angle, pole_v = observation
        digitized = [
            np.digitize(cart_pos, bins=self.bins(-2.4, 2.4, NUM_DIGITIZE)),
            np.digitize(cart_v, bins=self.bins(-3.0, 3.0, NUM_DIGITIZE)),
            np.digitize(pole_angle, bins=self.bins(-0.5, 0.5, NUM_DIGITIZE)),
            np.digitize(pole_v, bins=self.bins(-2.0, 2.0, NUM_DIGITIZE)),
        ]
        return sum([x * (NUM_DIGITIZE**i) for i, x in enumerate(digitized)])

    def update_qtable(self, observation, action, reward, observation_next):
        state = self.digitize_state(observation)
        state_next = self.digitize_state(observation_next)
        max_q_next = max(self.q_table[state_next][:])
        self.q_table[state, action] = self.q_table[state, action] + ETA * (reward + GAMMA * max_q_next - self.q_table[state, action])

    def decide_action(self, observation, episode):
        state = self.digitize_state(observation)
        epsilon = 0.5 / (episode + 1)

        if epsilon < np.random.uniform(0, 1):
            return np.argmax(self.q_table[state][:])
        else:
            return np.random.choice(self.num_actions)

class Environment():
    def __init__(self):
        self.env = gym.make(ENV)
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        self.agent = Agent(self.num_states, self.num_actions)

    def run(self):
        complete_episodes = 0
        episode_final = False

        for episode in range(NUM_EPISODE):
            observation = self.env.reset()
            episode_reward = 0

            for step in range(MAX_STEPS):
                if episode_final:
                    frames.append(self.env.render(mode='rgb_array'))

                action = self.agent.get_action(observation, episode)
                observation_next, _, done, _ = self.env.step(action)

                reward = 0
                if done:
                    if step < 195:
                        reward = -1
                        complete_episodes = 0
                    else:
                        reward = 1
                        complete_episodes = complete_episodes + 1

                episode_reward += reward

                self.agent.update_q_function(observation, action, reward, observation_next)
                observation = observation_next

                if done:
                    print('episode: {0}, step: {1}'.format(episode, step))
                    break

                if episode_final:
                    # display_frames_as_gif(frames)
                    break

                if 10 <= complete_episodes:
                    print('success 10 times in sequence')
                    frames = []
                    episode_final = True
                    
        self.env.close()
        
if __name__ == '__main__':
    # frames = []
    # env = gym.make(ENV)
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
    env = Environment()
    env.run()

