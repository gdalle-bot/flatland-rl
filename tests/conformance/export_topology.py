"""
Export the Python topology semantics that Flatland.jl's tests compare against (julia/Flatland/test/data/topology.json).

Usage: python tests/conformance/export_topology.py julia/Flatland/test/data/topology.json
"""
import json
import sys

import numpy as np

from flatland.core.env_observation_builder import DummyObservationBuilder
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


def rail_record(seed, p_level_free):
    env = RailEnv(
        width=30, height=30, number_of_agents=2, random_seed=seed,
        rail_generator=sparse_rail_generator(max_num_cities=4, max_rails_between_cities=2, max_rail_pairs_in_city=2, p_level_free=p_level_free),
        line_generator=sparse_line_generator(), obs_builder_object=DummyObservationBuilder(),
    )
    env.reset(random_seed=seed)
    rail = env.rail
    entry_points = []
    for r in range(rail.height):
        for c in range(rail.width):
            for d in range(4):
                ep = ((r, c), d)
                if not rail.is_valid_entry_point(ep):
                    continue
                nexts = []
                for action in RailEnvActions:
                    target = rail.apply_action_independent(action, ep)
                    nexts.append(None if target is None else [target[0][0], target[0][1], target[1]])
                successors = [[s[0][0], s[0][1], s[1]] for s in rail.get_successor_entry_points(ep)]
                resource = env.resource_map.get_resource(ep, None)
                resource = [resource[0], resource[1]] if isinstance(resource[0], int) else [resource[0][0], resource[0][1], resource[1]]
                entry_points.append({"ep": [r, c, d], "next": nexts, "successors": successors, "resource": resource})
    return {
        "seed": seed,
        "p_level_free": p_level_free,
        "grid": rail.grid.astype(int).tolist(),
        "level_free": sorted([int(p[0]), int(p[1])] for p in env.resource_map.level_free_positions),
        "entry_points": entry_points,
    }


if __name__ == "__main__":
    data = {
        "check_action": check_action_table(),
        "rails": [rail_record(1, 0.0), rail_record(2, 1.0)],
    }
    with open(sys.argv[1], "w") as f:
        json.dump(data, f, separators=(",", ":"))
