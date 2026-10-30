import hashlib
import json

import numpy as np
import pytest

from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.malfunction_generators import MalfunctionParameters, ParamMalfunctionGen
from flatland.envs.observations import Node, TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator


def _canonical(node):
    """A JSON-serializable form of a tree observation, independent of numpy's scalar repr."""
    if isinstance(node, Node):
        return {"f": [float(x) for x in node[:-1]],
                "c": {k: _canonical(v) for k, v in sorted(node.childs.items())}}
    if node is None:
        return None
    return float(node)


def _tree_obs_digest(p_level_free, steps):
    env = RailEnv(
        width=30, height=30, number_of_agents=10, random_seed=1,
        rail_generator=sparse_rail_generator(max_num_cities=3, max_rails_between_cities=2, max_rail_pairs_in_city=2, p_level_free=p_level_free),
        line_generator=sparse_line_generator({1.: .25, 1. / 2.: .25, 1. / 3.: .25, 1. / 4.: .25}),
        malfunction_generator=ParamMalfunctionGen(MalfunctionParameters(malfunction_rate=1 / 20, min_duration=2, max_duration=5)),
        obs_builder_object=TreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv(max_depth=20)),
    )
    observations, _ = env.reset(random_seed=1)
    rng = np.random.default_rng(0)
    h = hashlib.sha256()
    for _ in range(steps):
        h.update(json.dumps({str(k): _canonical(v) for k, v in sorted(observations.items())}).encode())
        observations, _, dones, _ = env.step({a: int(rng.integers(0, 5)) for a in env.get_agent_handles()})
        if dones["__all__"]:
            break
    h.update(json.dumps({str(k): _canonical(v) for k, v in sorted(observations.items())}).encode())
    return h.hexdigest()


@pytest.mark.parametrize("p_level_free, expected", [
    (0.0, "94bae7e1ed920f55410345f3caf831b15ab824c50ef3b1cba9114fb6cdd83ba2"),
    (1.0, "2b05b9ca7a9d85193e1812c0b5c16209ae2adf1312817e7ab51d7150bf5966fe"),
])
def test_tree_obs_unchanged_along_episode(p_level_free, expected):
    """
    Ten agents with mixed speeds and frequent malfunctions on a 30x30 sparse rail, with and without level-free
    crossings, follow seeded random actions for 60 steps, observed with `TreeObsForRailEnv(max_depth=3)` and a
    shortest-path predictor of depth 20.
    - Every observation along the episode, including right after reset, hashes to the value recorded with
      flatland-rl 4.3's tree observation, so speed-ups of the tree observation must keep all features identical.
    """
    assert _tree_obs_digest(p_level_free, steps=60) == expected
