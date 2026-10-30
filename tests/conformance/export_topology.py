"""
Export the Python topology semantics that Flatland.jl's tests compare against (julia/Flatland/test/data/topology.json):
- `_check_action_new` for every valid cell code, heading and action;
- every valid entry point of two generated 30x30 sparse rails (one with level-free crossings);
- every entry point, valid or not, of small random non-square grids filled with random valid cell codes. They are not
  closed rails, so moves off the grid, into empty cells, into dead ends and facing symmetric switches all occur, and
  level-free crossings are anywhere, borders included.

Usage: python tests/conformance/export_topology.py julia/Flatland/test/data/topology.json
"""
import json
import sys

import numpy as np

from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.core.grid.grid_resource_map import GridResourceMap
from flatland.envs.grid.rail_env_grid import RailEnvTransitionsEnum
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap


def check_action_table():
    """`_check_action_new` for every valid cell code, heading and action."""
    rows = []
    for code in sorted({int(t.value) for t in RailEnvTransitionsEnum}):
        rail = RailGridTransitionMap(width=1, height=1, grid=np.array([[code]], dtype=np.uint16))
        for orientation in range(4):
            for action in RailEnvActions:
                new_direction, transition_valid, preprocessed, action_valid = rail._check_action_new(action, (0, 0), orientation)
                rows.append([code, orientation, action.value, int(new_direction), bool(transition_valid), preprocessed.value, bool(action_valid)])
    return rows


# entry point, validity, target per action (None if invalid), successors, resource (cell or cell and axis)
ENTRY_POINT_FIELDS = ["r", "c", "d", "valid", "next", "successors", "resource"]


def entry_point_record(rail, resource_map, ep):
    """What Python computes for entry point `ep`, as a list of `ENTRY_POINT_FIELDS`."""
    nexts = []
    for action in RailEnvActions:
        target = rail.apply_action_independent(action, ep)
        nexts.append(None if target is None else [target[0][0], target[0][1], target[1]])
    successors = [[s[0][0], s[0][1], s[1]] for s in rail.get_successor_entry_points(ep)]
    resource = resource_map.get_resource(ep, None)
    resource = [resource[0], resource[1]] if isinstance(resource[0], int) else [resource[0][0], resource[0][1], resource[1]]
    (r, c), d = ep
    return [r, c, d, bool(rail.is_valid_entry_point(ep)), nexts, successors, resource]


def random_rail_records(n, seed):
    """`n` random non-square grids of 2 to 6 rows and columns, filled with random valid cell codes."""
    rng = np.random.default_rng(seed)
    codes = sorted({int(t.value) for t in RailEnvTransitionsEnum})
    records = []
    for _ in range(n):
        height, width = int(rng.integers(2, 7)), int(rng.integers(2, 7))
        if height == width:
            width += 1
        grid = rng.choice(codes, size=(height, width)).astype(np.uint16)
        level_free = {(int(r), int(c)) for r, c in zip(rng.integers(0, height, 3), rng.integers(0, width, 3))}
        rail = RailGridTransitionMap(width=width, height=height, grid=grid)
        resource_map = GridResourceMap(level_free_positions=level_free)
        entry_points = [entry_point_record(rail, resource_map, ((r, c), d)) for r in range(height) for c in range(width) for d in range(4)]
        records.append({"grid": grid.astype(int).tolist(), "level_free": sorted([r, c] for r, c in level_free), "entry_points": entry_points})
    return records


def rail_record(seed, p_level_free):
    env = RailEnv(
        width=30, height=30, number_of_agents=2, random_seed=seed,
        rail_generator=sparse_rail_generator(max_num_cities=4, max_rails_between_cities=2, max_rail_pairs_in_city=2, p_level_free=p_level_free),
        line_generator=sparse_line_generator(), obs_builder_object=DummyObservationBuilder(),
    )
    env.reset(random_seed=seed)
    rail = env.rail
    entry_points = [entry_point_record(rail, env.resource_map, ((r, c), d))
                    for r in range(rail.height) for c in range(rail.width) for d in range(4)
                    if rail.is_valid_entry_point(((r, c), d))]
    return {
        "seed": seed,
        "p_level_free": p_level_free,
        "grid": rail.grid.astype(int).tolist(),
        "level_free": sorted([int(p[0]), int(p[1])] for p in env.resource_map.level_free_positions),
        "entry_points": entry_points,
    }


if __name__ == "__main__":
    data = {
        "entry_point_fields": ENTRY_POINT_FIELDS,
        "check_action": check_action_table(),
        "rails": [rail_record(1, 0.0), rail_record(2, 1.0)],
        "random_rails": random_rail_records(20, seed=0),
    }
    with open(sys.argv[1], "w") as f:
        json.dump(data, f, separators=(",", ":"))
