import hashlib
import json
import random

import numpy as np
import pytest

from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.malfunction_generators import MalfunctionParameters, ParamMalfunctionGen
from flatland.envs.observations import Node, TreeObsForRailEnv
from flatland.envs.predictions import DummyPredictorForRailEnv, ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_generators import sparse_rail_generator


def _make_env(p_level_free, predictor, remove_agents_at_target=True):
    return RailEnv(
        width=30, height=30, number_of_agents=10, random_seed=1,
        rail_generator=sparse_rail_generator(max_num_cities=3, max_rails_between_cities=2, max_rail_pairs_in_city=2, p_level_free=p_level_free),
        line_generator=sparse_line_generator({1.: .25, 1. / 2.: .25, 1. / 3.: .25, 1. / 4.: .25}),
        malfunction_generator=ParamMalfunctionGen(MalfunctionParameters(malfunction_rate=1 / 20, min_duration=2, max_duration=5)),
        obs_builder_object=TreeObsForRailEnv(max_depth=3, predictor=predictor),
        remove_agents_at_target=remove_agents_at_target,
    )


def _canonical(node):
    """A JSON-serializable form of a tree observation, independent of numpy's scalar repr."""
    if isinstance(node, Node):
        return {"f": [float(x) for x in node[:-1]],
                "c": {k: _canonical(v) for k, v in sorted(node.childs.items())}}
    if node is None:
        return None
    return float(node)


def _observations_json(observations):
    return json.dumps({str(k): _canonical(v) for k, v in sorted(observations.items())})


def _actions(env, rng):
    """Seeded actions, mostly greedy on the distance map, so that some agents reach their targets."""
    distances = env.distance_map.get()
    actions = {}
    for agent in env.agents:
        if agent.current_entry_point is None or rng.random() < 0.2:
            actions[agent.handle] = rng.randrange(5)
            continue
        best, best_distance = RailEnvActions.MOVE_FORWARD, np.inf
        for action in (RailEnvActions.MOVE_LEFT, RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_RIGHT):
            result = env.rail.apply_action_independent(action, agent.current_entry_point)
            if result is not None:
                (row, column), direction = result
                if distances[agent.handle, row, column, direction] < best_distance:
                    best, best_distance = action, distances[agent.handle, row, column, direction]
        actions[agent.handle] = best.value
    return actions


def _tree_obs_digest(env, steps):
    observations, _ = env.reset(random_seed=1)
    rng = random.Random(0)
    h = hashlib.sha256()
    for _ in range(steps):
        h.update(_observations_json(observations).encode())
        observations, _, dones, _ = env.step(_actions(env, rng))
        if dones["__all__"]:
            break
    h.update(_observations_json(observations).encode())
    return h.hexdigest()


@pytest.mark.parametrize("p_level_free, predictor, remove_agents_at_target, expected", [
    (0.0, "shortest_path", True, "01fde77b23e282caf1eac5519c1ee0129e3a07c10d0463d5db60509c0ec338c2"),
    (1.0, "shortest_path", True, "c378287459c1f156900f24856e81387205d3c73fd0a98793c54a64b4cd80e4f9"),
    (0.0, None, True, "01fde77b23e282caf1eac5519c1ee0129e3a07c10d0463d5db60509c0ec338c2"),
    (0.0, "shortest_path", False, "c7f3b7f9aa698a70134b4b8bb192cd14846c2c698e4cc1ce23c1d193223c8266"),
])
def test_tree_obs_unchanged_along_episode(p_level_free, predictor, remove_agents_at_target, expected):
    """
    Ten agents with mixed speeds and frequent malfunctions on a 30x30 sparse rail follow seeded, mostly greedy
    actions on the distance map for 150 steps, observed with `TreeObsForRailEnv(max_depth=3)`. Parametrized over
    level-free crossings, a shortest-path predictor of depth 20 or none, and whether agents are removed at their target.
    - One or two agents reach their targets during the episode.
    - Without a predictor, the observations are the same as with one.
    - Every observation along the episode, including right after reset, hashes to a fixed digest, so speed-ups of the
      tree observation must keep all features identical.
    """
    env = _make_env(
        p_level_free=p_level_free,
        predictor=ShortestPathPredictorForRailEnv(max_depth=20) if predictor == "shortest_path" else None,
        remove_agents_at_target=remove_agents_at_target,
    )
    assert _tree_obs_digest(env, steps=150) == expected


def test_tree_obs_with_dummy_predictor_while_some_agents_are_off_map():
    """
    Ten agents on a 30x30 sparse rail, observed with `TreeObsForRailEnv(max_depth=3)` and a `DummyPredictorForRailEnv`,
    which only predicts for agents on the map.
    - Setup: all agents start off the map, then follow seeded, mostly greedy actions for 30 steps, so some
      depart while others still wait.
    - Every step, every agent gets a tree observation, identical to the one built without a predictor.
    """
    env = _make_env(p_level_free=0.0, predictor=DummyPredictorForRailEnv(max_depth=5))
    reference = _make_env(p_level_free=0.0, predictor=None)
    observations, _ = env.reset(random_seed=1)
    reference_observations, _ = reference.reset(random_seed=1)
    rng = random.Random(0)
    some_off_map = some_on_map = False
    for _ in range(30):
        assert _observations_json(observations) == _observations_json(reference_observations)
        some_off_map |= any(agent.current_entry_point is None for agent in env.agents)
        some_on_map |= any(agent.current_entry_point is not None for agent in env.agents)
        actions = _actions(env, rng)
        observations, _, _, _ = env.step(actions)
        reference_observations, _, _, _ = reference.step(actions)
    assert some_off_map and some_on_map


def test_tree_obs_get_many_on_a_subset_of_handles():
    """
    Ten agents on a 30x30 sparse rail with a shortest-path predictor, after 20 steps of seeded, mostly greedy
    actions.
    - `get_many([7, 2, 5])` returns exactly the observations of agents 7, 2 and 5, identical to those from
      `get_many` over all agents.
    """
    env = _make_env(p_level_free=0.0, predictor=ShortestPathPredictorForRailEnv(max_depth=20))
    env.reset(random_seed=1)
    rng = random.Random(0)
    for _ in range(20):
        env.step(_actions(env, rng))
    all_observations = env.obs_builder.get_many(env.get_agent_handles())
    some_observations = env.obs_builder.get_many([7, 2, 5])
    assert list(some_observations) == [7, 2, 5]
    assert _observations_json(some_observations) == _observations_json({h: all_observations[h] for h in [7, 2, 5]})
