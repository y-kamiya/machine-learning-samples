from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from baselines.common.atari_wrappers import WarpFrame, FrameStack
import gym
import retro
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

NUM_EPISODE = 10
NUM_STEPS = 1000
LEARNING_RATE = 0.001
GAMMA = 0.999

def make_env(stack=True, scale_rew=True):
    """
    Create an environment with some standard wrappers.
    """
    env = retro.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1')
    env = SonicDiscretizer(env)
    if scale_rew:
        env = RewardScaler(env)
    env = WarpFrame(env)
    if stack:
        env = FrameStack(env, 4)
    return env

class Net(nn.Module):
    def __init__(self, num_actions):
        super(Net, self).__init__()
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(4, 32, kernel_size=5, padding=2)
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

class Environment:
    def __init__(self):
        self.env = make_env()
        self.model = Net(self.env.action_space.n)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    def prepro(self, I):
        ret = np.zeros((4, 84, 84))
        # I = I[::4,::4, :] # downsample by factor of 2
        ret[0] = I[:, :, 0]
        ret[1] = I[:, :, 1]
        ret[2] = I[:, :, 2]
        ret[3] = I[:, :, 3]
        return ret

    def run_to_goal(self):
        observation = self.env.reset()
        state_prev = None
        history = []

        self.model.eval()

        for step in range(0, NUM_STEPS):
           print("step: {0}".format(step))
           state = self.prepro(observation)
           state_diff = state - state_prev if state_prev is not None else np.zeros(4*84*84).reshape(4,84,84)
           state_prev = state

           tensor = torch.tensor(state_diff, dtype=torch.float32).unsqueeze(0)
           output = self.model(tensor).squeeze(0)
           props = output.data.numpy()

           action_index = np.random.choice(range(0, self.env.action_space.n), p=props)

           action = np.zeros(self.env.action_space.n)
           action[action_index] = 1
           next_observation, reward, done, info = self.env.step(action_index)

           history.append([step, action_index, reward, output, action])

           if done:
               break

           observation = next_observation

        return history

    def discount_reward(self, rewards):
        discounted_rewards = np.zeros((rewards.size, self.env.action_space.n))
        running_add = 0
        for i in range(rewards.size)[::-1]:
            running_add = running_add * GAMMA + rewards[i]
            for j in range(0, self.env.action_space.n):
                discounted_rewards[i][j] = running_add

        return discounted_rewards

    def update_policy(self, history, episode):
        self.model.train()

        rewards = np.zeros((len(history)))
        targets = np.zeros((len(history), self.env.action_space.n))
        for i, entry in enumerate(history):
            rewards[i] = entry[2]
            targets[i] = entry[4]
            
        discounted_rewards = self.discount_reward(rewards)
        targets = targets * discounted_rewards

        targets.reshape(-1, self.env.action_space.n)
        targets = torch.tensor(targets, dtype=torch.float32)

        self.optimizer.zero_grad()
        for i, entry in enumerate(history):
            loss = F.smooth_l1_loss(entry[3], targets[i])
            loss.backward()

        self.optimizer.step()
        
    def run(self):
        for episode in range(NUM_EPISODE):
            obs = self.env.reset()
            history = self.run_to_goal()
            self.update_policy(history, episode)

class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()

class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.
    This is incredibly important and effects performance
    drastically.
    """
    def reward(self, reward):
        return reward * 0.01

class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """
    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    def reset(self, **kwargs): # pylint: disable=E0202
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action): # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info


if __name__ == '__main__':
    env = Environment()
    env.run()


