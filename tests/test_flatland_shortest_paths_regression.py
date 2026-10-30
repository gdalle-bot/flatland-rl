import hashlib
import json

import numpy as np
import pytest

from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.malfunction_generators import MalfunctionParameters, ParamMalfunctionGen
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.core.env_observation_builder import DummyObservationBuilder


def _paths_and_predictions_digest(p_level_free, line_length, steps):
    env = RailEnv(
        width=30, height=30, number_of_agents=10, random_seed=1,
        rail_generator=sparse_rail_generator(max_num_cities=4, max_rails_between_cities=2, max_rail_pairs_in_city=2, p_level_free=p_level_free),
        line_generator=sparse_line_generator({1.: .25, 1. / 2.: .25, 1. / 3.: .25, 1. / 4.: .25}, line_length=line_length),
        malfunction_generator=ParamMalfunctionGen(MalfunctionParameters(malfunction_rate=1 / 20, min_duration=2, max_duration=5)),
        obs_builder_object=DummyObservationBuilder(),
    )
    env.reset(random_seed=1)
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
      at every step, to the values recorded with flatland-rl 4.3.
    """
    assert _paths_and_predictions_digest(p_level_free, line_length, steps=60) == expected
