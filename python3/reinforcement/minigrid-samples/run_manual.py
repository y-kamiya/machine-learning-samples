import sys
import argparse
import gymnasium as gym
from gymnasium.envs.registration import register
from minigrid.manual_control import ManualControl

# Custom maze definition
MAP_DEF = [
    "########",
    "#s.....#",
    "#.###..#",
    "#...#..#",
    "#####.##",
    "#......#",
    "#.#g#.##",
    "########",
]

class CustomManualControl(ManualControl):
    def key_handler(self, event):
        key = event.key  # Returns a string like "p", "d", "left", etc.
        
        # Ensure we have access to the MiniGrid actions
        actions = self.env.unwrapped.actions

        # Custom key bindings using string comparison
        if key == "p":
            print(f"[DEBUG] Key '{key}' -> Action: Pickup")
            self.step(actions.pickup)
        elif key == "d":
            print(f"[DEBUG] Key '{key}' -> Action: Drop")
            self.step(actions.drop)
        elif key == "q":
            print(f"[DEBUG] Key '{key}' -> Action: Done")
            self.step(actions.done)
        else:
            # Let the parent handle arrows, space, backspace, etc.
            super().key_handler(event)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "env_id",
        nargs="?",
        default="MazeFromText-Manual-v1",
        help="Gymnasium environment ID",
    )
    args = parser.parse_args()

    env_id = args.env_id
    is_predefined = env_id in gym.envs.registry.keys()

    if not is_predefined:
        try:
            register(
                id=env_id,
                entry_point="maze_from_text:MazeFromTextEnv",
                kwargs={"map_def": MAP_DEF, "max_steps": 200},
            )
        except Exception:
            pass

    # Use render_mode="human" for ManualControl
    env = gym.make(env_id, render_mode="human")

    print(f"\nManual Control: {env_id}")
    print("Controls:")
    print("- Arrows: Move / Turn")
    print("- Space: Toggle")
    print("- p: Pickup")
    print("- d: Drop")
    print("- q: Done")
    print("- Backspace: Reset")
    print("- Esc: Quit")

    manual_control = CustomManualControl(env, seed=42)
    manual_control.start()

if __name__ == "__main__":
    main()
