from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from retro_contest.local import make

def main():
    env = make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1', bk2dir='./data')
    obs = env.reset()
    while True:
        action = env.action_space.sample()
        action[7] = 1
        obs, rew, done, info = env.step(action)
        env.render()
        if done:
            break
            # obs = env.reset()

    env.close()

if __name__ == '__main__':
    main()
