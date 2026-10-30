import hashlib
import json

import numpy as np
import pytest

from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.envs.graph_rail_env import GraphRailEnv
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator


def _env(seed, p_level_free, line_length):
    env = RailEnv(
        width=40, height=40, number_of_agents=12, random_seed=seed,
        rail_generator=sparse_rail_generator(max_num_cities=4, max_rails_between_cities=2, max_rail_pairs_in_city=2, p_level_free=p_level_free),
        line_generator=sparse_line_generator({1.: .5, 1. / 3.: .5}, line_length=line_length),
        obs_builder_object=DummyObservationBuilder(),
    )
    env.reset(random_seed=seed)
    return env


def _grid_digest(seed, p_level_free, line_length):
    env = _env(seed, p_level_free, line_length)
    h = hashlib.sha256()
    # the timetable depends on the timetable generator's own distance map (intermediate stops included)
    h.update(json.dumps([env._max_episode_steps] + [[a.earliest_departure, a.latest_arrival] for a in env.agents]).encode())
    h.update(np.ascontiguousarray(env.distance_map.get()).tobytes())
    # cloning (a load from the env's full state) recomputes the distance map for the loaded agents and rail
    clone = RailEnv(width=env.width, height=env.height, obs_builder_object=DummyObservationBuilder())
    clone.clone_from(env)
    h.update(np.ascontiguousarray(clone.distance_map.get()).tobytes())
    return h.hexdigest()


def _graph_digest(seed):
    env = GraphRailEnv.from_rail_env(_env(seed, 0.0, 2), observation_builder=DummyObservationBuilder(), seed=seed)
    env.reset(random_seed=seed)
    dm = env.distance_map.get()
    return hashlib.sha256(json.dumps({str(k): sorted((str(ep), float(d)) for ep, d in v.items()) for k, v in sorted(dm.items())}).encode()).hexdigest()


@pytest.mark.parametrize("seed, p_level_free, line_length, expected", [
    (1, 0.0, 2, "3a1d2e1492cda04b47f64be39a9867734653ad0940382f30170eff0e30cc374d"),
    (2, 1.0, 3, "4bc160c847121f12f7f3070ddf0a09d4e307c8408e2ada0bd12d541e1ba595d9"),
])
def test_grid_distance_map_unchanged(seed, p_level_free, line_length, expected):
    """
    Twelve agents on a 40x40 sparse rail with four cities, once without level-free crossings and lines of two stops,
    once with level-free crossings and lines of three stops.
    - The timetable, the distance map after reset, and the distance map of a clone of the env
      hash to the values recorded with flatland-rl 4.3.
    """
    assert _grid_digest(seed, p_level_free, line_length) == expected


def test_graph_distance_map_unchanged():
    """The distance map of a `GraphRailEnv` converted from a 40x40 grid env hashes to the value recorded with flatland-rl 4.3."""
    assert _graph_digest(1) == "6a439bd0488a657b3ed2d7d41fcc063aaac2894b6cd00575d8f35730e0ab18d0"
