from __future__ import annotations

import gymnasium as gym
from gymnasium.envs.registration import register
from minigrid.manual_control import ManualControl

from maze_from_text import MazeFromTextEnv


if __name__ == "__main__":
    map_def = [
        "########",
        "#s.....#",
        "#.###..#",
        "#...#..#",
        "#####.##",
        "#......#",
        "#.#g#.##",
        "########"
    ]

    register(
        id="MazeFromText-v0",
        entry_point="maze_from_text:MazeFromTextEnv",
        kwargs={"map_def": map_def},
    )

    env = gym.make("MazeFromText-v0", render_mode="human")

    # manual_control = ManualControl(env, seed=42)
    # manual_control.start()

    env.reset()
    for t in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\nStep {t+1} reward={reward}, done={terminated or truncated}")
        print(env.unwrapped.dump())
        if terminated:
            print("\nReached goal!")
            break


