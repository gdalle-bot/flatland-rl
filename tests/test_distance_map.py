import math
from collections import deque

import numpy as np
import pytest

from flatland.core.distance_map_walker import DistanceMapWalker
from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.grid.distance_map import DistanceMap
from flatland.envs.grid.rail_env_grid import RailEnvTransitions
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap


def _make_simple_env():
    # _ _ _

    transitions = RailEnvTransitions()
    cells = transitions.transition_list
    dead_end_from_south = cells[7]
    dead_end_from_west = transitions.rotate_transition(dead_end_from_south, 90)
    dead_end_from_east = transitions.rotate_transition(dead_end_from_south, 270)
    vertical_straight = cells[1]
    horizontal_straight = transitions.rotate_transition(vertical_straight, 90)

    rail_map = np.array(
        [[dead_end_from_east] + [horizontal_straight] + [dead_end_from_west]], dtype=np.uint16)
    rail = RailGridTransitionMap(width=rail_map.shape[1], height=rail_map.shape[0], transitions=transitions)
    rail.grid = rail_map

    city_positions = [(0, 2), (0, 1)]
    train_stations = [
        [((0, 1), 0)],
        [((0, 2), 0)],
    ]
    city_orientations = [1, 0]
    agents_hints = {
        'num_agents': 1,
        'city_positions': city_positions,
        'train_stations': train_stations,
        'city_orientations': city_orientations
    }
    optionals = {'agents_hints': agents_hints}

    return RailEnv(width=rail_map.shape[1],
                   height=rail_map.shape[0],
                   rail_generator=rail_from_grid_transition_map(rail, optionals),
                   line_generator=sparse_line_generator(),
                   number_of_agents=1,
                   obs_builder_object=TreeObsForRailEnv(max_depth=2,
                                                        predictor=ShortestPathPredictorForRailEnv(max_depth=10)),
                   )


def test_walker():
    env = _make_simple_env()
    env.reset()

    # set initial position and direction for testing...
    env.agents[0].current_entry_point = ((0, 1), 1)
    env.agents[0].targets = {((0, 0), d) for d in Grid4TransitionsEnum}
    # reset to set agents from agents_static
    # env.reset(False, False)
    env.distance_map._compute(env.agents, env.rail)

    print(env.distance_map.get()[(0, *[0, 1], 1)])
    assert env.distance_map.get()[(0, *[0, 1], 1)] == 3
    print(env.distance_map.get()[(0, *[0, 2], 3)])
    assert env.distance_map.get()[(0, *[0, 2], 1)] == 2


def test_distances_follow_the_rail_after_reset():
    """
    Three agents on a 30x30 sparse rail, whose distance map was computed once.
    - Setup: the distance map is reset onto a copy of the rail where one cell of agent 0's shortest path (not a target
      cell) has no transitions, with the same agents and targets.
    - Agent 0's distances change, and all distances equal those of a new `DistanceMap` built on the modified rail.
    """
    env = RailEnv(width=30, height=30, number_of_agents=3, rail_generator=sparse_rail_generator(max_num_cities=3),
                  line_generator=sparse_line_generator(), obs_builder_object=DummyObservationBuilder())
    env.reset(random_seed=1)
    before = np.copy(env.distance_map.get())
    target_cells = {target[0] for agent in env.agents for target in agent.targets}
    grid = env.rail.grid.copy()
    cut = next(w.position for w in env.distance_map.get_shortest_paths()[0][1:] if w.position not in target_cells)
    grid[cut] = 0
    modified_rail = RailGridTransitionMap(width=env.width, height=env.height, transitions=env.rail.transitions)
    modified_rail.grid = grid

    env.distance_map.reset(env.agents, modified_rail)
    after = env.distance_map.get()
    fresh = DistanceMap(env.agents, env.height, env.width)
    fresh.reset(env.agents, modified_rail)
    assert not np.array_equal(before[0], after[0])
    assert np.array_equal(after, fresh.get())


class _DigraphRail:
    """A rail given by the predecessors of each entry point."""

    def __init__(self, edges):
        self.predecessors = {}
        for source, target in edges:
            self.predecessors.setdefault(target, []).append(source)

    def get_predecessor_entry_points(self, entry_point):
        return self.predecessors.get(entry_point, [])


def _distances_one_search_per_target(rail, targets):
    """Reference: one backward breadth-first search per target, then the minimum over targets."""
    best = {}
    for target in targets:
        distances = {target: 0}
        queue = deque([target])
        while queue:
            entry_point = queue.popleft()
            for predecessor in rail.get_predecessor_entry_points(entry_point):
                if predecessor not in distances:
                    distances[predecessor] = distances[entry_point] + 1
                    queue.append(predecessor)
        for entry_point, distance in distances.items():
            best[entry_point] = min(best.get(entry_point, math.inf), distance)
    return best


@pytest.mark.parametrize("targets", [["a"], ["a", "c"], ["c", "a", "a"], ["e"], ["a", "f"], []])
def test_walker_on_a_looped_rail_matches_one_search_per_target(targets):
    """
    A loop a -> b -> c -> d -> a, a branch e -> b into it, a branch d -> f out of it, and a self-loop on c. Walked
    backwards from one target, two targets on the loop (each reachable from the other), repeated targets, a target
    with no predecessors, a target on each side of the exit, and none.
    - The distance of every entry point to its nearest target equals the minimum over targets of one backward search
      per target.
    """
    rail = _DigraphRail([("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"), ("e", "b"), ("d", "f"), ("c", "c")])
    assert DistanceMapWalker(None)._distance_map_walker(rail, targets) == _distances_one_search_per_target(rail, targets)


def test_loaded_distance_map_is_used_for_shortest_paths():
    """
    Regression test: a distance map populated via `.set(...)` (as when loading a precomputed distance map from a
    saved rail/env file, without computing it) is used by `get_shortest_paths()`/`geometric_distance()`, which give
    the same paths as for the computed distance map.
    """
    env = _make_simple_env()
    env.reset(random_seed=1)  # seed known to place agent 0 with a valid (non-None) shortest path

    precomputed = env.distance_map.get()
    expected_paths = env.distance_map.get_shortest_paths()
    assert expected_paths[0] is not None

    fresh_dm = DistanceMap(agents=env.agents, env_height=env.height, env_width=env.width)
    fresh_dm.set(precomputed)
    fresh_dm.reset(env.agents, env.rail)

    actual_paths = fresh_dm.get_shortest_paths()
    assert actual_paths[0] is not None
    assert actual_paths[0] == expected_paths[0]


def test_agent_with_no_valid_targets_does_not_crash():
    """
    Regression test: an agent whose `targets` set has become empty (e.g. via `RailEnvPersister.set_full_state`'s
    post-load filtering to only rail-valid entry points, which can filter out every target) does not make the
    distance map computation raise, and `get_shortest_paths()` returns `None` (no path found) for that agent.
    """
    env = RailEnv(width=25, height=25,
                  rail_generator=sparse_rail_generator(max_num_cities=3, seed=1),
                  line_generator=sparse_line_generator(),
                  number_of_agents=2)
    env.reset(random_seed=1)
    env.agents[0].targets = set()

    paths = env.distance_map.get_shortest_paths()
    assert paths[0] is None
