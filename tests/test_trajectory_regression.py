import hashlib
import json
import math
import random
from fractions import Fraction
from typing import NamedTuple

import numpy as np
import pytest

from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.envs.graph_rail_env import GraphRailEnv
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.malfunction_generators import MalfunctionParameters, ParamMalfunctionGen
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_env_state_machine_wrapper import RailEnvStateMachineWrapper
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.step_utils.states import TrainState

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
    if config == "state_machine":
        env = RailEnvStateMachineWrapper(env)
    if config == "graph":
        env.reset(random_seed=1)
        env = GraphRailEnv.from_rail_env(env, observation_builder=DummyObservationBuilder(), seed=1)
    return env


def _value(x):
    """A JSON-serializable form of rewards, dones and infos."""
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


def _actions(env, rng):
    """Seeded actions: mostly towards the target, sometimes random (invalid ones included) or stopping."""
    actions = {}
    for agent in env.agents:
        u = rng.random()
        if agent.current_entry_point is None:
            actions[agent.handle] = RailEnvActions.MOVE_FORWARD.value if u < 0.8 else int(rng.random() * 5)
        elif u < 0.1:
            actions[agent.handle] = int(rng.random() * 5)
        elif u < 0.15:
            actions[agent.handle] = RailEnvActions.STOP_MOVING.value
        else:
            # actions apply when the agent enters its next entry point
            best, best_distance = RailEnvActions.MOVE_FORWARD, math.inf
            for action in (RailEnvActions.MOVE_LEFT, RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_RIGHT):
                result = env.rail.apply_action_independent(action, agent.next_entry_point)
                if result is not None and env.distance_map.geometric_distance(result, agent.handle) < best_distance:
                    best, best_distance = action, env.distance_map.geometric_distance(result, agent.handle)
            actions[agent.handle] = best.value
    return actions


class _Coverage(NamedTuple):
    """What an episode went through, so that the test fails if it stops exercising these behaviours."""
    agents_done: int
    malfunctions: int
    level_free_visits: int


def _trajectory_digest(config):
    env = _make_env(config)
    _, info = env.reset(random_seed=1)
    rng = random.Random(0)
    h = hashlib.sha256()
    level_free = getattr(getattr(env, "resource_map", None), "level_free_positions", set())
    level_free_visits = 0

    def record(rewards, dones, info):
        nonlocal level_free_visits
        # entry points and speeds by repr, which shows their types (e.g. numpy integers in entry points)
        agents = [[repr(a.current_entry_point), repr(a.next_entry_point), repr(a.old_entry_point), repr(a.target_entry_point),
                   repr(a.speed_counter.speed), repr(a.speed_counter.distance), a.malfunction_handler.malfunction_down_counter,
                   a.derived_state(env._elapsed_steps).name, a.state.name, a.arrival_time] for a in env.agents]
        positions = env.agent_positions.tolist() if getattr(env, "agent_positions", None) is not None else None
        h.update(json.dumps(_value([env._elapsed_steps, agents, positions, rewards, dones, info])).encode())
        level_free_visits += sum(a.current_entry_point is not None and a.current_entry_point[0] in level_free for a in env.agents)

    record({}, {}, info)
    dones = {"__all__": False}
    while not dones["__all__"]:
        _, rewards, dones, info = env.step(_actions(env, rng))
        record(rewards, dones, info)
    coverage = _Coverage(
        agents_done=sum(a.derived_state(env._elapsed_steps) == TrainState.DONE for a in env.agents),
        malfunctions=sum(a.malfunction_handler.num_malfunctions for a in env.agents),
        level_free_visits=level_free_visits,
    )
    return h.hexdigest(), coverage


@pytest.mark.cython_ext
@pytest.mark.parametrize("config, expected", [
    pytest.param("default", "d3e26ce1a8328aee6689456dce722fac5444c686e2dbb2ab0d2e752c52d9e950", id="default"),
    pytest.param("level_free", "7b0f79ebceb28243f199e25821e4dc64b8d0151adb2ae0b877f68b91fcb69288", id="level_free"),
    pytest.param("stay_at_target_fractional_deltas", "371defde1847043c7c6d22c01775504a9c0091e8e126964876500ed9e587f35a", id="stay_at_target_fractional_deltas"),
    pytest.param("graph", "4296b9b94feaf0262b9243de0a6f430b3958189a1943e31cd8c2ae154cbf21aa", id="graph"),
    pytest.param("state_machine", "0540562f860eb7b3e319104ce93ffe5ea7726b23f767d95b411292a453ab891d", id="state_machine"),
])
def test_trajectory_unchanged(config, expected):
    """
    Twelve agents with speeds 1, 1/2, 1/3 and 1/4 and frequent malfunctions on a 30x30 sparse rail with up to four
    cities, until the episode ends. Agents mostly head for their target, and sometimes take a random action (invalid
    ones included) or stop. Five configurations: the defaults; level-free crossings with lines of three stops; agents
    staying at their target with acceleration 1/4 and braking 1/3 per step; the same env converted to a
    `GraphRailEnv`; the default env wrapped with `RailEnvStateMachineWrapper`.
    - Some agents reach their target, some malfunction, and with level-free crossings some agents stand on one.
    - At every step, including right after reset, every agent's entry points (current, next, previous, target), speed,
      distance, malfunction counter, state and arrival time, the grid of agent positions, and the rewards, dones and
      infos hash to a fixed digest. Speed-ups of step() must keep trajectories identical.
    - The digest also depends on the rail, line and timetable generators, so a change there requires recording it
      again.
    """
    digest, coverage = _trajectory_digest(config)
    assert coverage.agents_done >= 3
    assert coverage.malfunctions >= 1
    if config == "level_free":
        assert coverage.level_free_visits >= 1
    assert digest == expected
