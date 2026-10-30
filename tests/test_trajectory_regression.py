import hashlib
import json
from fractions import Fraction

import numpy as np
import pytest

from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.envs.graph_rail_env import GraphRailEnv
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.malfunction_generators import MalfunctionParameters, ParamMalfunctionGen
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator

SPEEDS = {1.: .25, 1. / 2.: .25, 1. / 3.: .25, 1. / 4.: .25}


def _make_env(config):
    kwargs = dict(
        width=30, height=30, number_of_agents=12, random_seed=1,
        rail_generator=sparse_rail_generator(max_num_cities=4, max_rails_between_cities=2, max_rail_pairs_in_city=2,
                                             p_level_free=1.0 if config == "level_free" else 0.0),
        line_generator=sparse_line_generator(SPEEDS, line_length=3 if config == "level_free" else 2),
        malfunction_generator=ParamMalfunctionGen(MalfunctionParameters(malfunction_rate=1 / 30, min_duration=2, max_duration=6)),
        obs_builder_object=DummyObservationBuilder(),
    )
    if config == "stay_at_target_fractional_deltas":
        kwargs.update(remove_agents_at_target=False, acceleration_delta=Fraction(1, 4), braking_delta=-Fraction(1, 3))
    env = RailEnv(**kwargs)
    if config == "graph":
        env.reset(random_seed=1)
        env = GraphRailEnv.from_rail_env(env, observation_builder=DummyObservationBuilder(), seed=1)
    return env


def _value(x):
    """A JSON-serializable, platform-independent form of step outputs."""
    if isinstance(x, dict):
        return {str(k): _value(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_value(v) for v in x]
    if isinstance(x, Fraction):
        return str(x)
    if hasattr(x, "name") and hasattr(x, "value"):  # enums
        return x.name
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    return x


def _trajectory_digest(config, steps=150):
    env = _make_env(config)
    _, info = env.reset(random_seed=1)
    rng = np.random.default_rng(0)
    h = hashlib.sha256()

    def record(rewards, dones, info):
        agents = [[a.current_entry_point, a.target_entry_point, a.speed_counter.speed, a.speed_counter.distance,
                   a.malfunction_handler.malfunction_down_counter, a.derived_state(env._elapsed_steps)] for a in env.agents]
        h.update(json.dumps(_value([env._elapsed_steps, agents, rewards, dones, info])).encode())

    record({}, {}, info)
    for _ in range(steps):
        _, rewards, dones, info = env.step({a: int(rng.integers(0, 5)) for a in env.get_agent_handles()})
        record(rewards, dones, info)
        if dones["__all__"]:
            break
    return h.hexdigest()


@pytest.mark.parametrize("config, expected", [
    ("default", "f828a86ba5628790fa586ca9941c5a2c73236ad3b2f52a074c1defcc3658a53c"),
    ("level_free", "4ee3a78933089307e96f583dd422d9976dba1ab0897af0fb936327671229280d"),
    ("stay_at_target_fractional_deltas", "732c703fe07f04bff6342144e19c109c58ca548b146f753a796ccfec6718f24f"),
    ("graph", "a88168456e25d16686eb6915aea5cbb57f681d6974d7a2405a3b7c18fb7e93e3"),
])
def test_trajectory_unchanged(config, expected):
    """
    Twelve agents with speeds 1, 1/2, 1/3 and 1/4 and frequent malfunctions on a 30x30 sparse rail with four cities
    follow seeded random actions for up to 150 steps, in four configurations: the defaults; level-free crossings with
    lines of three stops; agents staying at their target with acceleration 1/4 and braking 1/3 per step; the same env
    converted to a GraphRailEnv.
    - At every step, including right after reset, every agent's entry point, target entry point, speed, distance,
      malfunction counter and state, as well as the rewards, dones and infos, hash to the values recorded with
      flatland-rl 4.3. Speed-ups of step() must keep trajectories identical.
    """
    assert _trajectory_digest(config) == expected
