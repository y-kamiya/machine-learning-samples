from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv


class MazeFromTextEnv(MiniGridEnv):
    def __init__(self, map_def, render_mode=None, max_steps=10):
        self.map_def = map_def
        self.height = len(map_def)
        self.width = len(map_def[0])
        self.max_steps = max_steps

        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space,
            grid_size=self.width,
            max_steps=max_steps,
            render_mode=render_mode,
        )

    @staticmethod
    def _gen_mission():
        return "Reach the goal"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        start_pos = None

        for y, row in enumerate(self.map_def):
            for x, ch in enumerate(row):
                if ch == "#":
                    self.grid.set(x, y, Wall())
                elif ch == "g":
                    self.put_obj(Goal(), x, y)
                elif ch == "s":
                    start_pos = (x, y)

        if start_pos is None:
            raise ValueError("Map does not contain start 's'")

        self.agent_pos = start_pos
        self.agent_dir = 0

        return self.grid

    def _cell_char(self, x, y):
        ax, ay = self.agent_pos
        if (x, y) == (ax, ay):
            return "A"

        obj = self.grid.get(x, y)
        if obj is None:
            return "."

        char_map = {"Wall": "#", "Goal": "G"}
        return char_map.get(obj.__class__.__name__, "?")

    def dump(self):
        rows = []
        for y in range(self.height):
            row = [self._cell_char(x, y) for x in range(self.width)]
            rows.append("".join(row))

        return "\n".join(rows)

