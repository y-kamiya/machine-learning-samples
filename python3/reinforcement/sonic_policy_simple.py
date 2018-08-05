from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
from baselines.common.atari_wrappers import WarpFrame, FrameStack
import gym
import retro
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import namedtuple 

NUM_EPISODES_DEFAULT = 100
NUM_STEPS_DEFAULT = 3000
LEARNING_RATE = 0.0001
GAMMA = 0.99
DATA_PATH_DEFAULT = 'model_state_sonic.dat'

def make_env(num_steps, stack=True, scale_rew=True):
    """
    Create an environment with some standard wrappers.
    """
    env = retro.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1')
    # env.auto_record('./data')
    env = gym.wrappers.TimeLimit(env, max_episode_steps=num_steps)
    env = SonicDiscretizer(env)
    env = AllowBacktracking(env)
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
    def __init__(self, args):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.env = make_env(args.steps)
        self.model = Net(self.env.action_space.n).to(device=self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        data_path = args.path
        if data_path:
            self.model.load_state_dict(torch.load(data_path))
        self.data_path = data_path if data_path != None else DATA_PATH_DEFAULT

        self.is_saved = not args.nosave
        self.is_render = args.render
        self.num_steps = args.steps if args.steps != None else NUM_STEPS_DEFAULT
        self.num_episodes = args.episodes if args.episodes != None else NUM_EPISODES_DEFAULT

    def prepro(self, I):
        ret = np.zeros((4, 84, 84))
        # I = I[::4,::4, :] # downsample by factor of 2
        ret[0] = I[:, :, 0]
        ret[1] = I[:, :, 1]
        ret[2] = I[:, :, 2]
        ret[3] = I[:, :, 3]
        return ret

    def run_to_goal(self, episode):
        print('start run_to_goal')
        observation = self.env.reset()
        state_prev = None
        history = [[] for _ in range(0,3)]

        self.model.eval()

        for step in range(0, self.num_steps):
           state = self.prepro(observation)
           state_diff = state - state_prev if state_prev is not None else np.zeros(4*84*84).reshape(4,84,84)
           state_prev = state

           tensor = torch.tensor(state_diff, dtype=torch.float32, device=self.device).unsqueeze(0)
           output = self.model(tensor)
           props = output.cpu().squeeze(0).data.numpy()

           action_index = np.random.choice(range(0, self.env.action_space.n), p=props)

           next_observation, reward, done, info = self.env.step(action_index)
           # print(info)
           if self.is_render:
               self.env.render()
           # print("step: {0}, action: {1}, reward: {2}".format(step, action_index, reward))

           history[0].append(action_index)
           history[1].append(reward)
           history[2] = torch.cat([history[2], output]) if step != 0 else output

           if done:
               print('done')
               history[1][-1] += 0.01 if 0 < info['score'] else -0.01
               break

           observation = next_observation

        return history

    def discount_reward(self, rewards):
        print('start discount_reward')
        size = len(rewards)
        discounted_rewards = np.zeros((size, self.env.action_space.n))
        running_add = 0
        for i in range(size)[::-1]:
            running_add = running_add * GAMMA + rewards[i]
            for j in range(0, self.env.action_space.n):
                discounted_rewards[i][j] = running_add

        # discounted_rewards -= np.mean(discounted_rewards)
        # discounted_rewards /= np.std(discounted_rewards)

        return discounted_rewards

    def update_policy(self, history, episode):
        print('start update_policy')
        self.model.train()

        actions = history[0]
        rewards = history[1]
        outputs = history[2]
        print(actions, rewards, outputs)

        targets = np.zeros((len(actions), self.env.action_space.n))
        for i in range(0, len(actions)):
            targets[i][actions[i]] = 1
            
        discounted_rewards = self.discount_reward(rewards)
        targets = targets * discounted_rewards

        targets.reshape(-1, self.env.action_space.n)
        targets = torch.tensor(targets, dtype=torch.float32, device=self.device)

        loss = F.smooth_l1_loss(outputs, targets)
        loss.backward()

        self.optimizer.step()
        
    def run(self):
        for episode in range(self.num_episodes):
            history = self.run_to_goal(episode)
            self.update_policy(history, episode)
            if self.is_saved:
                torch.save(self.model.state_dict(), self.data_path)

            print('finish episode {0}'.format(episode))

        self.env.close()

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
    argv = sys.argv[1:]

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('-p', '--path', help='path to model data file')
    parser.add_argument('--nosave', action='store_true', help='model parameters are saved')
    parser.add_argument('--steps', type=int, help='step count')
    parser.add_argument('--episodes', type=int, help='episode count')
    parser.add_argument('--render', action='store_true', help='render game')
    args = parser.parse_args(argv)

    env = Environment(args)
    env.run()


