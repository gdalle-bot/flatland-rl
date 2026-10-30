import ast

import networkx as nx
import numpy as np
import pytest

from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.env_generation.env_generator import env_generator
from flatland.envs.graph.rail_graph_transition_map import GraphTransitionMap
from flatland.envs.graph_rail_env import GraphRailEnv
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.rail_env import RailEnv
from flatland.envs.grid.rail_env_grid import RailEnvTransitions, RailEnvTransitionsEnum
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap

# rows are top to bottom!
_EAST_OWN = (1, 1, Grid4TransitionsEnum.EAST)  # at "W" side, heading "E"
_EAST_NEIGHBOR = (1, 2, Grid4TransitionsEnum.EAST)  # to the "E", heading "E"
_WEST_OWN = (1, 1, Grid4TransitionsEnum.WEST)  # at "E" side, heading "W"
_WEST_NEIGHBOR = (1, 0, Grid4TransitionsEnum.WEST)  # to the "W", heading "W"
_NORTH_OWN = (1, 1, Grid4TransitionsEnum.NORTH)  # at "S" side, heading "N"
_NORTH_NEIGHBOR = (0, 1, Grid4TransitionsEnum.NORTH)  # at "N" side, heading "N"
_SOUTH_OWN = (1, 1, Grid4TransitionsEnum.SOUTH)  # at "N" side, heading "S"
_SOUTH_NEIGHBOR = (2, 1, Grid4TransitionsEnum.SOUTH)  # to the "S", heading "S"


@pytest.mark.parametrize("transition,expected_edges", [
    # Case 0 - empty cell (1)
    (RailEnvTransitionsEnum.empty, []),

    # Case 1 - straight (2)
    (RailEnvTransitionsEnum.horizontal_straight, [
        (_EAST_OWN, _EAST_NEIGHBOR), (_WEST_OWN, _WEST_NEIGHBOR),
    ]),
    (RailEnvTransitionsEnum.vertical_straight, [
        (_NORTH_OWN, _NORTH_NEIGHBOR), (_SOUTH_OWN, _SOUTH_NEIGHBOR),
    ]),

    # Case 2 - simple switch left (4)
    (RailEnvTransitionsEnum.simple_switch_west_left, [
        (_WEST_OWN, _WEST_NEIGHBOR), (_EAST_OWN, _EAST_NEIGHBOR),
        (_WEST_OWN, _SOUTH_NEIGHBOR), (_NORTH_OWN, _EAST_NEIGHBOR),
    ]),
    (RailEnvTransitionsEnum.simple_switch_north_left, [
        (_SOUTH_OWN, _SOUTH_NEIGHBOR), (_NORTH_OWN, _NORTH_NEIGHBOR),
        (_NORTH_OWN, _WEST_NEIGHBOR), (_EAST_OWN, _SOUTH_NEIGHBOR),
    ]),
    (RailEnvTransitionsEnum.simple_switch_east_left, [
        (_WEST_OWN, _WEST_NEIGHBOR), (_EAST_OWN, _EAST_NEIGHBOR),
        (_EAST_OWN, _NORTH_NEIGHBOR), (_SOUTH_OWN, _WEST_NEIGHBOR),
    ]),
    (RailEnvTransitionsEnum.simple_switch_south_left, [
        (_SOUTH_OWN, _SOUTH_NEIGHBOR), (_NORTH_OWN, _NORTH_NEIGHBOR),
        (_SOUTH_OWN, _EAST_NEIGHBOR), (_WEST_OWN, _NORTH_NEIGHBOR),
    ]),

    # Case 3 - diamond crossing (1)
    (RailEnvTransitionsEnum.diamond_crossing, [
        (_SOUTH_OWN, _SOUTH_NEIGHBOR), (_NORTH_OWN, _NORTH_NEIGHBOR),
        (_WEST_OWN, _WEST_NEIGHBOR), (_EAST_OWN, _EAST_NEIGHBOR),
    ]),

    # Case 4 - single slip (4)
    (RailEnvTransitionsEnum.single_slip_SW, [
        (_SOUTH_OWN, _SOUTH_NEIGHBOR), (_NORTH_OWN, _NORTH_NEIGHBOR),
        (_WEST_OWN, _WEST_NEIGHBOR), (_EAST_OWN, _EAST_NEIGHBOR),
        (_EAST_OWN, _SOUTH_NEIGHBOR), (_NORTH_OWN, _WEST_NEIGHBOR),
    ]),
    (RailEnvTransitionsEnum.single_slip_NW, [
        (_SOUTH_OWN, _SOUTH_NEIGHBOR), (_NORTH_OWN, _NORTH_NEIGHBOR),
        (_WEST_OWN, _WEST_NEIGHBOR), (_EAST_OWN, _EAST_NEIGHBOR),
        (_SOUTH_OWN, _WEST_NEIGHBOR), (_EAST_OWN, _NORTH_NEIGHBOR),
    ]),
    (RailEnvTransitionsEnum.single_slip_NE, [
        (_SOUTH_OWN, _SOUTH_NEIGHBOR), (_NORTH_OWN, _NORTH_NEIGHBOR),
        (_WEST_OWN, _WEST_NEIGHBOR), (_EAST_OWN, _EAST_NEIGHBOR),
        (_WEST_OWN, _NORTH_NEIGHBOR), (_SOUTH_OWN, _EAST_NEIGHBOR),
    ]),
    (RailEnvTransitionsEnum.single_slip_SE, [
        (_SOUTH_OWN, _SOUTH_NEIGHBOR), (_NORTH_OWN, _NORTH_NEIGHBOR),
        (_WEST_OWN, _WEST_NEIGHBOR), (_EAST_OWN, _EAST_NEIGHBOR),
        (_NORTH_OWN, _EAST_NEIGHBOR), (_WEST_OWN, _SOUTH_NEIGHBOR),
    ]),

    # Case 5 - double slip (2)
    (RailEnvTransitionsEnum.double_slip_NW_SE, [
        (_SOUTH_OWN, _SOUTH_NEIGHBOR), (_NORTH_OWN, _NORTH_NEIGHBOR),
        (_WEST_OWN, _WEST_NEIGHBOR), (_EAST_OWN, _EAST_NEIGHBOR),
        (_SOUTH_OWN, _WEST_NEIGHBOR), (_EAST_OWN, _NORTH_NEIGHBOR),
        (_NORTH_OWN, _EAST_NEIGHBOR), (_WEST_OWN, _SOUTH_NEIGHBOR),
    ]),
    (RailEnvTransitionsEnum.double_slip_NE_SW, [
        (_SOUTH_OWN, _SOUTH_NEIGHBOR), (_NORTH_OWN, _NORTH_NEIGHBOR),
        (_WEST_OWN, _WEST_NEIGHBOR), (_EAST_OWN, _EAST_NEIGHBOR),
        (_WEST_OWN, _NORTH_NEIGHBOR), (_SOUTH_OWN, _EAST_NEIGHBOR),
        (_EAST_OWN, _SOUTH_NEIGHBOR), (_NORTH_OWN, _WEST_NEIGHBOR),
    ]),

    # Case 6 - symmetrical switch (4)
    (RailEnvTransitionsEnum.symmetric_switch_from_north, [
        (_SOUTH_OWN, _EAST_NEIGHBOR), (_WEST_OWN, _NORTH_NEIGHBOR),
        (_SOUTH_OWN, _WEST_NEIGHBOR), (_EAST_OWN, _NORTH_NEIGHBOR),
    ]),
    (RailEnvTransitionsEnum.symmetric_switch_from_east, [
        (_WEST_OWN, _NORTH_NEIGHBOR), (_NORTH_OWN, _EAST_NEIGHBOR),
        (_WEST_OWN, _SOUTH_NEIGHBOR), (_SOUTH_OWN, _EAST_NEIGHBOR),
    ]),
    (RailEnvTransitionsEnum.symmetric_switch_from_south, [
        (_WEST_OWN, _SOUTH_NEIGHBOR), (_NORTH_OWN, _EAST_NEIGHBOR),
        (_EAST_OWN, _SOUTH_NEIGHBOR), (_NORTH_OWN, _WEST_NEIGHBOR),
    ]),
    (RailEnvTransitionsEnum.symmetric_switch_from_west, [
        (_EAST_OWN, _NORTH_NEIGHBOR), (_SOUTH_OWN, _WEST_NEIGHBOR),
        (_EAST_OWN, _SOUTH_NEIGHBOR), (_NORTH_OWN, _WEST_NEIGHBOR),
    ]),

    # Case 7 - dead end (4)
    (RailEnvTransitionsEnum.dead_end_from_north, [(_SOUTH_OWN, _NORTH_NEIGHBOR)]),
    (RailEnvTransitionsEnum.dead_end_from_south, [(_NORTH_OWN, _SOUTH_NEIGHBOR)]),
    (RailEnvTransitionsEnum.dead_end_from_west, [(_EAST_OWN, _WEST_NEIGHBOR)]),
    (RailEnvTransitionsEnum.dead_end_from_east, [(_WEST_OWN, _EAST_NEIGHBOR)]),

    # Case 1b (8)  - simple turn right (4)
    # Case 1c (9)  - simple turn left (- same as Case 1b)
    (RailEnvTransitionsEnum.right_turn_from_north, [(_SOUTH_OWN, _WEST_NEIGHBOR), (_EAST_OWN, _NORTH_NEIGHBOR)]),
    (RailEnvTransitionsEnum.right_turn_from_east, [(_WEST_OWN, _NORTH_NEIGHBOR), (_SOUTH_OWN, _EAST_NEIGHBOR)]),
    (RailEnvTransitionsEnum.right_turn_from_south, [(_NORTH_OWN, _EAST_NEIGHBOR), (_WEST_OWN, _SOUTH_NEIGHBOR)]),
    (RailEnvTransitionsEnum.right_turn_from_west, [(_EAST_OWN, _SOUTH_NEIGHBOR), (_NORTH_OWN, _WEST_NEIGHBOR)]),

    # Case 2b (10) - simple switch right (4)
    (RailEnvTransitionsEnum.simple_switch_west_right, [
        (_WEST_OWN, _WEST_NEIGHBOR), (_EAST_OWN, _EAST_NEIGHBOR),
        (_WEST_OWN, _NORTH_NEIGHBOR), (_SOUTH_OWN, _EAST_NEIGHBOR),
    ]),
    (RailEnvTransitionsEnum.simple_switch_north_right, [
        (_SOUTH_OWN, _SOUTH_NEIGHBOR), (_NORTH_OWN, _NORTH_NEIGHBOR),
        (_NORTH_OWN, _EAST_NEIGHBOR), (_WEST_OWN, _SOUTH_NEIGHBOR),
    ]),
    (RailEnvTransitionsEnum.simple_switch_east_right, [
        (_WEST_OWN, _WEST_NEIGHBOR), (_EAST_OWN, _EAST_NEIGHBOR),
        (_EAST_OWN, _SOUTH_NEIGHBOR), (_NORTH_OWN, _WEST_NEIGHBOR),
    ]),
    (RailEnvTransitionsEnum.simple_switch_south_right, [
        (_SOUTH_OWN, _SOUTH_NEIGHBOR), (_NORTH_OWN, _NORTH_NEIGHBOR),
        (_SOUTH_OWN, _WEST_NEIGHBOR), (_EAST_OWN, _NORTH_NEIGHBOR),
    ]),
])
def test_grid_to_digraph(transition, expected_edges):
    RailEnvTransitions().print(transition)
    # use 3 x 3 not to go -1
    rail_map = np.array(
        [[RailEnvTransitionsEnum.empty] * 3] +
        [[RailEnvTransitionsEnum.empty, transition, RailEnvTransitionsEnum.empty]] +
        [[RailEnvTransitionsEnum.empty] * 3], dtype=np.uint16)

    gtm = RailGridTransitionMap(width=rail_map.shape[1], height=rail_map.shape[0], transitions=RailEnvTransitions())
    gtm.grid = rail_map

    g = GraphTransitionMap.grid_to_digraph(gtm)
    actual_edges = g.edges
    actual_edges = {(ast.literal_eval(u), ast.literal_eval(v)) for u, v in actual_edges}
    print(list(actual_edges))
    print(expected_edges)
    print(f"surplus edges: {set(actual_edges) - set(expected_edges)}")
    print(f"missing edges: {set(expected_edges) - set(actual_edges)}")
    assert len(actual_edges) == len(expected_edges)
    assert set(actual_edges) == set(expected_edges)


def test_successors_ordered_as_on_grid_rails():
    """
    A node "u" with four successors, added to the graph in this order: "other", reached with no action, then "right"
    (MOVE_RIGHT), "forward" (MOVE_FORWARD, DO_NOTHING, STOP_MOVING) and "left" (MOVE_LEFT).
    - The successors come in the order of grid rails, whatever the graph's insertion order or the hashing of the node
      names: "left", "forward", "right", then "other".
    """
    g = nx.DiGraph()
    g.add_edge("u", "other")
    g.add_edge("u", "right", actions={RailEnvActions.MOVE_RIGHT})
    g.add_edge("u", "forward", actions={RailEnvActions.MOVE_FORWARD, RailEnvActions.DO_NOTHING, RailEnvActions.STOP_MOVING})
    g.add_edge("u", "left", actions={RailEnvActions.MOVE_LEFT})
    assert list(GraphTransitionMap(g).get_successor_entry_points("u")) == ["left", "forward", "right", "other"]


def _grid_envs():
    """Generated grid envs, the last one with level-free crossings."""
    envs = [env_generator(seed=seed)[0] for seed in range(42, 50)]
    level_free = RailEnv(width=30, height=30, number_of_agents=2, random_seed=1,
                         rail_generator=sparse_rail_generator(max_num_cities=3, p_level_free=1.0),
                         line_generator=sparse_line_generator(), obs_builder_object=DummyObservationBuilder())
    level_free.reset(random_seed=1)
    return envs + [level_free]


@pytest.mark.parametrize("env_index", range(9))
def test_graph_successors_equal_grid_successors(env_index):
    """
    A generated grid env (the ninth one with level-free crossings), converted to a graph.
    - At every valid entry point, the graph lists the same successors as the grid, in the same order.
    """
    grid = _grid_envs()[env_index].rail
    graph = GraphTransitionMap(GraphTransitionMap.grid_to_digraph(grid))
    to_graph = GraphTransitionMap.grid_entry_point_to_graph_entry_point
    for r in range(grid.height):
        for c in range(grid.width):
            for d in range(4):
                if not grid.is_valid_entry_point(((r, c), d)):
                    continue
                expected = [to_graph(r2, c2, d2) for (r2, c2), d2 in grid.get_successor_entry_points(((r, c), d))]
                assert list(graph.get_successor_entry_points(to_graph(r, c, d))) == expected, ((r, c), d)


@pytest.mark.parametrize("seed", range(42, 50))
def test_graph_shortest_paths_equal_grid_shortest_paths(seed):
    """
    Seven agents on a generated grid env, converted to a GraphRailEnv.
    - Every agent's shortest path in the graph env is its shortest path in the grid env: ties between equally short
      successors are broken in the same order.
    """
    grid_env, _, _ = env_generator(seed=seed)
    graph_env = GraphRailEnv.from_rail_env(grid_env, DummyObservationBuilder(), seed=seed, rewards=type(grid_env.rewards)())
    grid_paths = grid_env.distance_map.get_shortest_paths()
    graph_paths = graph_env.distance_map.get_shortest_paths()
    for handle, grid_path in grid_paths.items():
        graph_path = graph_paths[handle]
        expected = None if grid_path is None else [(tuple(int(x) for x in w.position), int(w.direction)) for w in grid_path]
        actual = None if graph_path is None else [GraphTransitionMap.graph_entry_point_to_grid_entry_point(str(w)) for w in graph_path]
        assert actual == expected, handle
