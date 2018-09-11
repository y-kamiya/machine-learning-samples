import gym
from baselines.common.atari_wrappers import make_atari, wrap_deepmind

ENV = 'BreakoutNoFrameskip-v4'
# env = gym.make(ENV)
env = wrap_deepmind(make_atari(ENV), frame_stack=True)
env.reset()
# for _ in range(1000):
index = 0
while True:
    env.render()
    _, reward, done, info = env.step(env.action_space.sample())
    print(index, reward, done, info)
    if done:
        break

    index += 1

env.close()
