from fractions import Fraction

import numpy as np
import pytest

from flatland.envs.agent_utils import virtual_entry_point
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.malfunction_generators import MalfunctionParameters, ParamMalfunctionGen
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.step_utils.states import TrainState


def _reference_global_obs(env, handle):
    """`GlobalObsForRailEnv.get` as of flatland-rl 4.3 (one pass over all agents per observing agent)."""
    rail_obs = np.zeros((env.height, env.width, 16))
    for i in range(rail_obs.shape[0]):
        for j in range(rail_obs.shape[1]):
            bitlist = [int(digit) for digit in bin(env.rail.get_full_transitions(i, j))[2:]]
            bitlist = [0] * (16 - len(bitlist)) + bitlist
            rail_obs[i, j] = np.array(bitlist)
    agent = env.agents[handle]
    agent_target = next(iter(agent.targets))[0]
    entry_point = virtual_entry_point(agent)
    if entry_point is None:
        return None
    agent_virtual_position, agent_virtual_direction = entry_point
    obs_targets = np.zeros((env.height, env.width, 2))
    obs_agents_state = np.zeros((env.height, env.width, 5)) - 1
    obs_agents_state[:, :, 4] = 0
    obs_agents_state[agent_virtual_position][0] = agent_virtual_direction if agent_virtual_direction is not None else -1
    obs_targets[agent_target][0] = 1
    for i in range(len(env.agents)):
        other_agent = env.agents[i]
        if other_agent.derived_state() == TrainState.DONE:
            continue
        obs_targets[next(iter(other_agent.targets))[0]][1] = 1
        if other_agent.current_entry_point is not None:
            other_agent_position = other_agent.current_entry_point[0]
            if i != handle:
                obs_agents_state[other_agent_position][1] = other_agent.current_entry_point[1]
            obs_agents_state[other_agent_position][2] = other_agent.malfunction_handler.malfunction_down_counter
            obs_agents_state[other_agent_position][3] = other_agent.speed_counter.speed
        if other_agent.derived_state(env._elapsed_steps).is_off_map_state():
            obs_agents_state[other_agent.initial_entry_point[0]][4] += 1
    return rail_obs, obs_agents_state, obs_targets


def _assert_matches_reference(env, observations):
    for handle in env.get_agent_handles():
        expected = _reference_global_obs(env, handle)
        actual = observations[handle]
        if expected is None:
            assert actual is None
            continue
        for e, a in zip(expected, actual):
            assert e.shape == a.shape
            assert np.array_equal(e, a), handle


def _make_env(p_level_free):
    return RailEnv(
        width=30, height=30, number_of_agents=10, random_seed=1,
        rail_generator=sparse_rail_generator(max_num_cities=3, max_rails_between_cities=2, max_rail_pairs_in_city=2, p_level_free=p_level_free),
        line_generator=sparse_line_generator({1.: .25, 1. / 2.: .25, 1. / 3.: .25, 1. / 4.: .25}),
        malfunction_generator=ParamMalfunctionGen(MalfunctionParameters(malfunction_rate=1 / 20, min_duration=2, max_duration=5)),
        obs_builder_object=GlobalObsForRailEnv(),
    )


@pytest.mark.parametrize("p_level_free", [0.0, 1.0])
def test_global_obs_matches_reference_along_episode(p_level_free):
    """
    Ten agents with mixed speeds and frequent malfunctions on a 30x30 sparse rail, with and without level-free
    crossings, follow seeded random actions for up to 150 steps.
    - Setup: reset with seed 1.
    - Every step, including right after reset, each agent's three observation arrays equal those of the previous
      one-pass-per-agent implementation, element by element; agents without an observation get `None` from both.
    """
    env = _make_env(p_level_free)
    observations, _ = env.reset(random_seed=1)
    rng = np.random.default_rng(0)
    for _ in range(150):
        _assert_matches_reference(env, observations)
        observations, _, dones, _ = env.step({h: int(rng.integers(0, 5)) for h in env.get_agent_handles()})
        if dones["__all__"]:
            break
    _assert_matches_reference(env, observations)


def test_global_obs_shared_cell_excludes_observing_agent():
    """
    Right after reset (all agents off the map), the first three agents are placed on the same cell (as on a
    level-free crossing) at speed 1/2, facing directions 0, 1 and 2 in handle order.
    - Each of the three sees in the second channel of that cell the direction of the last other agent there: agent 0
      and agent 1 see 2 (agent 2's direction), agent 2 sees 1 (agent 1's direction).
    - The results equal the previous implementation's for all agents, and `get()` for one handle equals `get_many()`.
    """
    env = _make_env(0.0)
    env.reset(random_seed=1)
    placed = env.agents[:3]
    cell = placed[0].initial_entry_point[0]
    for direction, agent in enumerate(placed):
        agent.current_entry_point = (cell, direction)
        agent.speed_counter.set(Fraction(1, 2), Fraction(0))
    observations = env.obs_builder.get_many(env.get_agent_handles())
    _assert_matches_reference(env, observations)
    seen = [observations[a.handle][1][cell][1] for a in placed]
    assert seen == [2, 2, 1]
    for a in placed:
        for x, y in zip(env.obs_builder.get(a.handle), observations[a.handle]):
            assert np.array_equal(x, y)


def test_default_observation_builder_is_not_shared_between_envs():
    """Two envs built without an explicit observation builder each get their own `GlobalObsForRailEnv`."""
    env1 = RailEnv(width=30, height=30)
    env2 = RailEnv(width=30, height=30)
    assert isinstance(env1.obs_builder, GlobalObsForRailEnv)
    assert isinstance(env2.obs_builder, GlobalObsForRailEnv)
    assert env1.obs_builder is not env2.obs_builder
