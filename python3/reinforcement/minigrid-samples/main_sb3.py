from __future__ import annotations

import os
import time

from gymnasium.wrappers import FlattenObservation, RecordEpisodeStatistics, TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from minigrid.wrappers import ImgObsWrapper

from maze_from_text import MazeFromTextEnv


MAP_DEF = [
    # "########",
    # "#s.....#",
    # "#.###..#",
    # "#...#..#",
    # "#####.##",
    # "#......#",
    # "#.#g#.##",
    # "########"
    "######",
    "#s...#",
    "#.##.#",
    "#.#..#",
    "#..#g#",
    "######",
]


def make_env(
    map_def=MAP_DEF, seed=0, render_mode="rgb_array", max_steps=200, is_train=True
):
    def _init():
        env = MazeFromTextEnv(
            map_def=map_def, render_mode=render_mode, max_steps=max_steps
        )
        env.reset(seed=seed)
        if is_train:
            env = RecordEpisodeStatistics(env)
            env = TimeLimit(env, max_episode_steps=max_steps)
        env = ImgObsWrapper(env)
        env = FlattenObservation(env)
        return env

    return _init


MODEL_DIR = "./models"
MODEL_PATH = os.path.join(MODEL_DIR, "ppo_maze")
os.makedirs(MODEL_DIR, exist_ok=True)

ROLLOUT_STEPS = 2048
BATCH_SIZE = 16


def train(total_timesteps=100_000, n_envs=4):
    env_fns = [make_env(seed=i) for i in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(venv=vec_env)

    policy_kwargs = dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])])

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        batch_size=BATCH_SIZE,
        n_steps=ROLLOUT_STEPS // n_envs,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./tb_logs/ppo_maze/",
        device="auto",
    )

    model.learn(total_timesteps=total_timesteps)

    model.save(MODEL_PATH)
    vec_env.close()
    print(f"Model saved to {MODEL_PATH}")


def evaluate(model_path=MODEL_PATH, episodes=5, is_dump=True):
    env = make_env(map_def=MAP_DEF, render_mode="human", is_train=False)()
    obs, info = env.reset(seed=0)

    model = PPO.load(model_path)

    for ep in range(episodes):
        obs, info = env.reset()
        step = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            print(f"step {step}: action {action}")
            obs, reward, terminated, truncated, info = env.step(int(action))
            step += 1

            if is_dump:
                base = getattr(env, "unwrapped", env)
                while hasattr(base, "env"):
                    base = base.env
                print(f"\n--- Episode {ep + 1} Step {step} ---")
                print(base.dump())

            time.sleep(0.1)

            if terminated or truncated:
                print(
                    f"Episode {ep + 1} finished after {step} steps, reward={info.get('episode', {}).get('r')}"
                )
                break
    env.close()


if __name__ == "__main__":
    train(total_timesteps=100000, n_envs=2)
    evaluate(episodes=3, is_dump=False)
