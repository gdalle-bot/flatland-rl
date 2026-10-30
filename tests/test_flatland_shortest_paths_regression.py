import copy
import hashlib
import json
import pickle

import numpy as np
import pytest

from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.envs.graph_rail_env import GraphRailEnv
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.malfunction_generators import MalfunctionParameters, ParamMalfunctionGen
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator


def _make_env(p_level_free, line_length):
    env = RailEnv(
        width=30, height=30, number_of_agents=10, random_seed=1,
        rail_generator=sparse_rail_generator(max_num_cities=4, max_rails_between_cities=2, max_rail_pairs_in_city=2, p_level_free=p_level_free),
        line_generator=sparse_line_generator({1.: .25, 1. / 2.: .25, 1. / 3.: .25, 1. / 4.: .25}, line_length=line_length),
        malfunction_generator=ParamMalfunctionGen(MalfunctionParameters(malfunction_rate=1 / 20, min_duration=2, max_duration=5)),
        obs_builder_object=DummyObservationBuilder(),
    )
    env.reset(random_seed=1)
    return env


def _paths_and_predictions_digest(p_level_free, line_length, steps):
    env = _make_env(p_level_free=p_level_free, line_length=line_length)
    predictor = ShortestPathPredictorForRailEnv(max_depth=20)
    predictor.reset(env)
    rng = np.random.default_rng(0)
    h = hashlib.sha256()
    h.update(json.dumps([env._max_episode_steps] + [[a.earliest_departure, a.latest_arrival] for a in env.agents]).encode())

    def record():
        full = env.distance_map.get_shortest_paths()
        cut = env.distance_map.get_shortest_paths(max_depth=7)
        for paths in (full, cut):
            h.update(json.dumps({str(k): None if v is None else [[list(w.position), w.direction] for w in v]
                                 for k, v in sorted(paths.items())}).encode())
        predictions = predictor.get()
        h.update(json.dumps({str(k): np.nan_to_num(v, nan=-7.0).tolist() for k, v in sorted(predictions.items())}).encode())
        h.update(json.dumps({str(k): sorted(map(str, v)) for k, v in sorted(env.dev_pred_dict.items())}).encode())

    for _ in range(steps):
        record()
        _, _, dones, _ = env.step({a: int(rng.integers(0, 5)) for a in env.get_agent_handles()})
        if dones["__all__"]:
            break
    record()
    return h.hexdigest()


@pytest.mark.parametrize("p_level_free, line_length, expected", [
    (0.0, 2, "c5534adb6ea53f9ef5e9fc05f5ab8aad1e0bf5b9496e1aba63c962d7dc7c0ed1"),
    (1.0, 3, "553a0c02ece6f38cbc4522e3d53445e53bb88eaaf81894cf7b81e73e22514ba7"),
])
def test_shortest_paths_and_predictions_unchanged_along_episode(p_level_free, line_length, expected):
    """
    Ten agents with mixed speeds and frequent malfunctions on a 30x30 sparse rail with four cities follow seeded random
    actions for 60 steps, once without level-free crossings and lines of two stops, once with level-free crossings
    and lines of three stops.
    - The timetable (max episode steps, every agent's earliest departure and latest arrival), every agent's shortest
      path (full and cut at 7 cells), the shortest-path predictor's predictions (depth 20) and its visited cells hash,
      at every step, to a fixed digest.
    """
    assert _paths_and_predictions_digest(p_level_free, line_length, steps=60) == expected


def _paths(distance_map):
    return {h: None if p is None else list(p) for h, p in distance_map.get_shortest_paths().items()}


def _paths_without_memo(distance_map):
    """Shortest paths from a copy of the distance map, which starts without memoized paths."""
    return _paths(copy.copy(distance_map))


def _env_after_steps(steps):
    env = _make_env(p_level_free=0.0, line_length=2)
    rng = np.random.default_rng(0)
    for _ in range(steps):
        env.step({a: int(rng.integers(0, 5)) for a in env.get_agent_handles()})
    return env


@pytest.mark.parametrize("how", ["set", "assign"])
def test_shortest_paths_follow_distance_changes(how):
    """
    Ten agents on a 30x30 sparse rail, after 10 steps of seeded random actions.
    - Setup: every agent's shortest path is queried once; then the distance of the third entry point on agent 0's
      path becomes infinite, through `set()` or by assigning the distance map attribute.
    - Agent 0's shortest path changes, and every agent's path is the same as for a copy of the distance map, which
      starts without memoized paths.
    """
    env = _env_after_steps(10)
    distance_map = env.distance_map
    before = _paths(distance_map)
    waypoint = before[0][2]
    distances = distance_map.get().copy()
    distances[(0, *waypoint.position, waypoint.direction)] = np.inf
    if how == "set":
        distance_map.set(distances)
    else:
        distance_map.distance_map = distances
    after = _paths(distance_map)
    assert after[0] != before[0]
    assert after == _paths_without_memo(distance_map)


def test_shortest_paths_follow_rail_changes():
    """
    Ten agents on a 30x30 sparse rail, after 10 steps of seeded random actions.
    - Setup: every agent's shortest path is queried once; then the rail cell of the third entry point on agent 0's path
      loses all its transitions.
    - Agent 0's shortest path changes, and every agent's path is the same as for a copy of the distance map, which
      starts without memoized paths.
    """
    env = _env_after_steps(10)
    before = _paths(env.distance_map)
    env.rail.set_transitions(before[0][2].position, 0)
    after = _paths(env.distance_map)
    assert after[0] != before[0]
    assert after == _paths_without_memo(env.distance_map)


def test_shortest_paths_after_unpickling_a_state_without_memo():
    """
    Ten agents on a 30x30 sparse rail, after 10 steps of seeded random actions.
    - A distance map restored from a pickled state that has no memoized paths and no record of which agents share
      distances, as pickled by earlier versions, gives the same shortest paths as the original.
    """
    env = _env_after_steps(10)
    distance_map = env.distance_map
    before = _paths(distance_map)
    state = distance_map.__getstate__()
    state.pop("_row_key_of")
    restored = type(distance_map).__new__(type(distance_map))
    restored.__setstate__(pickle.loads(pickle.dumps(state)))
    assert _paths(restored) == before


def test_graph_shortest_paths_unchanged_by_memo():
    """
    Ten agents on a 30x30 sparse rail, converted to a graph env, after 10 steps of seeded random actions.
    - Querying every agent's shortest path twice gives the same paths both times, and the same as for a copy of
      the distance map, which starts without memoized paths.
    """
    grid_env = _make_env(p_level_free=0.0, line_length=2)
    env = GraphRailEnv.from_rail_env(grid_env, DummyObservationBuilder(), seed=1, rewards=type(grid_env.rewards)())
    rng = np.random.default_rng(0)
    for _ in range(10):
        env.step({a: int(rng.integers(0, 5)) for a in env.get_agent_handles()})
    first = _paths(env.distance_map)
    assert any(path is not None for path in first.values())
    assert _paths(env.distance_map) == first
    assert first == _paths_without_memo(env.distance_map)
