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


def _reference_rail_obs(env):
    rail_obs = np.zeros((env.height, env.width, 16))
    for i in range(rail_obs.shape[0]):
        for j in range(rail_obs.shape[1]):
            bitlist = [int(digit) for digit in bin(env.rail.get_full_transitions(i, j))[2:]]
            bitlist = [0] * (16 - len(bitlist)) + bitlist
            rail_obs[i, j] = np.array(bitlist)
    return rail_obs


def _reference_global_obs(env, rail_obs, handle):
    """The global observation of one agent, built with one pass over all agents for this observer."""
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


def _assert_matches_reference(env, rail_obs, observations):
    for handle in env.get_agent_handles():
        expected = _reference_global_obs(env, rail_obs, handle)
        actual = observations[handle]
        if expected is None:
            assert actual is None
            continue
        for e, a in zip(expected, actual):
            assert e.shape == a.shape
            assert np.array_equal(e, a), handle


def _make_env(p_level_free, remove_agents_at_target=True):
    return RailEnv(
        width=30, height=30, number_of_agents=10, random_seed=1,
        rail_generator=sparse_rail_generator(max_num_cities=3, max_rails_between_cities=2, max_rail_pairs_in_city=2, p_level_free=p_level_free),
        line_generator=sparse_line_generator({1.: .25, 1. / 2.: .25, 1. / 3.: .25, 1. / 4.: .25}),
        malfunction_generator=ParamMalfunctionGen(MalfunctionParameters(malfunction_rate=1 / 20, min_duration=2, max_duration=5)),
        obs_builder_object=GlobalObsForRailEnv(),
        remove_agents_at_target=remove_agents_at_target,
    )


@pytest.mark.parametrize("p_level_free", [0.0, 1.0])
@pytest.mark.parametrize("remove_agents_at_target", [True, False])
def test_global_obs_matches_reference_along_episode(p_level_free, remove_agents_at_target):
    """
    Ten agents with speeds 1, 1/2, 1/3 and 1/4 and frequent malfunctions on a 30x30 sparse rail with three cities, with
    and without level-free crossings, with agents removed at or kept on their target, follow seeded random actions for
    up to 150 steps.
    - Setup: reset with seed 1.
    - At every step, including right after reset, each agent's three observation arrays equal, element by element,
      those built with one pass over all agents for that observer.
    """
    env = _make_env(p_level_free, remove_agents_at_target)
    observations, _ = env.reset(random_seed=1)
    rail_obs = _reference_rail_obs(env)
    rng = np.random.default_rng(0)
    for _ in range(150):
        _assert_matches_reference(env, rail_obs, observations)
        observations, _, dones, _ = env.step({h: int(rng.integers(0, 5)) for h in env.get_agent_handles()})
        if dones["__all__"]:
            break
    _assert_matches_reference(env, rail_obs, observations)


def test_global_obs_shared_cell_excludes_observing_agent():
    """
    Right after reset (all agents off the map), agents are placed by hand so that several share a cell, which the env
    itself never does on an ordinary cell.
    - Setup: agents 0, 1 and 2 at speed 1/2 on one cell, facing directions 0, 1 and 2; on another cell, an observer on
      map and a DONE agent kept on that cell.
    - In the second channel of the shared cell, each of agents 0, 1, 2 sees the direction of the last other agent there:
      agents 0 and 1 see 2, agent 2 sees 1. The observer sees -1 on its own cell, since a DONE agent is not shown.
    - All observations equal those built with one pass over all agents for each observer, and `get()` for one agent
      equals `get_many()`.
    """
    env = _make_env(0.0, remove_agents_at_target=False)
    env.reset(random_seed=1)
    shared_cell = env.agents[0].initial_entry_point[0]
    for direction, agent in enumerate(env.agents[:3]):
        agent.current_entry_point = (shared_cell, direction)
        agent.speed_counter.set(Fraction(1, 2), Fraction(0))
    observer, done_agent = [a for a in env.agents[3:] if a.initial_entry_point[0] != shared_cell][:2]
    observer_cell = observer.initial_entry_point
    observer.current_entry_point = observer_cell
    observer.speed_counter.set(Fraction(1, 2), Fraction(0))
    done_agent.current_entry_point = (observer_cell[0], (observer_cell[1] + 2) % 4)
    done_agent.target_entry_point = done_agent.current_entry_point
    done_agent.speed_counter.set(Fraction(0), Fraction(0))
    assert done_agent.derived_state(env._elapsed_steps) == TrainState.DONE

    observations = env.obs_builder.get_many(env.get_agent_handles())
    _assert_matches_reference(env, _reference_rail_obs(env), observations)
    assert [observations[a.handle][1][shared_cell][1] for a in env.agents[:3]] == [2, 2, 1]
    assert observations[observer.handle][1][observer_cell[0]][1] == -1
    for a in env.agents[:3] + [observer, done_agent]:
        for x, y in zip(env.obs_builder.get(a.handle), observations[a.handle]):
            assert np.array_equal(x, y)


class _CountingGlobalObs(GlobalObsForRailEnv):
    def get(self, handle=0):
        rail_obs, agents_state, targets = super().get(handle)
        return rail_obs, agents_state + 1000, targets


def test_global_obs_subclass_get_is_used_by_get_many():
    """
    A subclass of `GlobalObsForRailEnv` that overrides `get()` (here adding 1000 to the agent-state layer) gets its
    override applied to every observation of `reset()` and `step()`.
    """
    env = _make_env(0.0)
    env.obs_builder = _CountingGlobalObs()
    observations, _ = env.reset(random_seed=1)
    rail_obs = _reference_rail_obs(env)
    for handle, obs in observations.items():
        assert np.array_equal(obs[1], _reference_global_obs(env, rail_obs, handle)[1] + 1000)
    observations, _, _, _ = env.step({})
    for handle, obs in observations.items():
        assert np.array_equal(obs[1], _reference_global_obs(env, rail_obs, handle)[1] + 1000)
