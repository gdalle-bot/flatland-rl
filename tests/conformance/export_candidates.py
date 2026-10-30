"""
Export the collect phase's per-agent candidates that Flatland.jl's tests compare against
(julia/Flatland/test/data/candidates.json).

Each record holds the inputs of one call of `AbstractRailEnv.collect` (the agent's pre-step state, the action and the
entry point that action leads to), its outputs (candidate entry points, speed and distance), and the row of
`design_by_contract.md`'s Table 1 that it falls into. The records come from:
- every `collect` call of a few seeded episodes, thinned to at most `PER_KEY` per row, action and agent situation;
- a synthetic grid of agent states, fed to the real `collect` on stub env and agent objects, which reaches the rare
  combinations of Table 1 that episodes seldom produce (e.g. a malfunction while reaching the target, or an invalid
  departure). Its entry points are opaque indices: initial 1, current 2, next 3, the action's target 4, an off-map
  agent's target 5. It is thinned to one record per row, action (except for done agents) and combination of flags.

Usage: python tests/conformance/export_candidates.py julia/Flatland/test/data/candidates.json
"""
import itertools
import json
import math
import random
import sys
from fractions import Fraction
from types import SimpleNamespace
from typing import NamedTuple, Optional

import numpy as np

from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.env_generation.env_generator import env_generator_legacy
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.malfunction_generators import MalfunctionParameters, ParamMalfunctionGen
from flatland.envs.rail_env import AbstractRailEnv, RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.step_utils.speed_counter import SpeedCounter, cached_cell_exit

PER_KEY = 12

TABLE_1_ROWS = [
    "Done", "Target reached", "Malfunction", "Map entry", "Stay off map", "On-map cell transition",
    "Invalid action at cell exit", "Stopped", "Acceleration or start moving", "Braking", "Default",
]

# (max speed, acceleration delta, braking delta) of the synthetic grid; the third has a maximum speed below the
# acceleration
SYNTHETIC_PROFILES = [
    (Fraction(1), Fraction(1), -Fraction(1)),
    (Fraction(1, 2), Fraction(1, 4), -Fraction(1, 3)),
    (Fraction(1, 3), Fraction(1, 2), -Fraction(1, 2)),
    (Fraction(3, 4), Fraction(1, 2), -Fraction(1, 4)),
]
SYNTHETIC_ELAPSED_STEPS = 3


class CollectInput(NamedTuple):
    """The inputs of one `collect` call. Entry points are Flatland.jl indices (0 for None), rationals `[num, den]`."""
    current_entry_point: int
    next_entry_point: int
    initial_entry_point: int
    next_is_target: bool
    done: bool
    in_malfunction: bool
    speed: Optional[list]
    distance: Optional[list]
    max_speed: list
    earliest_departure: int
    elapsed_steps: int
    action: int
    independent: int


class CollectOutput(NamedTuple):
    """The candidates computed by one `collect` call, in the same encoding as `CollectInput`."""
    candidate_entry_point: int
    candidate_next_entry_point: int
    candidate_speed: list
    candidate_distance: Optional[list]


FIELDS = ["row", *CollectInput._fields, *CollectOutput._fields]


def table_1_row(agent, action, independent, candidate_entry_point):
    """The first row of Table 1 whose full formula holds, in the table's order of precedence."""
    off_map = agent.current_entry_point is None
    speed, distance = agent.speed_counter.speed, agent.speed_counter.distance
    cell_exit = off_map or cached_cell_exit(agent.speed_counter.max_speed, speed, distance)
    if agent.target_entry_point is not None:
        return "Done"
    if not off_map and cell_exit and agent.next_entry_point in agent.targets:
        return "Target reached"
    if agent.malfunction_handler.in_malfunction:
        return "Malfunction"
    if off_map:
        return "Stay off map" if candidate_entry_point is None else "Map entry"
    if cell_exit and independent is None:
        return "Invalid action at cell exit"
    if cell_exit:
        return "On-map cell transition"
    if speed == 0:
        return "Stopped"
    if action == RailEnvActions.MOVE_FORWARD:
        return "Acceleration or start moving"
    if action == RailEnvActions.STOP_MOVING:
        return "Braking"
    return "Default"


def opaque_index(entry_point):
    """The synthetic grid's entry points are already indices; 0 for None."""
    return 0 if entry_point is None else entry_point


def fraction(x):
    return None if x is None else [Fraction(x).numerator, Fraction(x).denominator]


def recorded_collect(env, agent, action, independent, index):
    """Call the original `collect`, and return its result and the `(row, CollectInput, CollectOutput)` record."""
    inputs = CollectInput(
        current_entry_point=index(agent.current_entry_point),
        next_entry_point=index(agent.next_entry_point),
        initial_entry_point=index(agent.initial_entry_point),
        next_is_target=agent.next_entry_point is not None and agent.next_entry_point in agent.targets,
        done=agent.target_entry_point is not None,
        in_malfunction=bool(agent.malfunction_handler.in_malfunction),
        speed=fraction(agent.speed_counter.speed),
        distance=fraction(agent.speed_counter.distance),
        max_speed=fraction(agent.speed_counter.max_speed),
        earliest_departure=int(agent.earliest_departure),
        elapsed_steps=int(env._elapsed_steps),
        action=int(action.value),
        independent=index(independent),
    )
    result = ORIGINAL_COLLECT(env, agent=agent, action=action, candidate_entry_point_independent=independent)
    data = env.temp_transition_data[agent.handle]
    row = table_1_row(agent=agent, action=action, independent=independent, candidate_entry_point=data.candidate_entry_point)
    outputs = CollectOutput(
        candidate_entry_point=index(data.candidate_entry_point),
        candidate_next_entry_point=index(data.candidate_next_entry_point),
        candidate_speed=fraction(data.candidate_speed),
        candidate_distance=fraction(data.candidate_distance),
    )
    return result, (row, inputs, outputs)


class Recorder:
    """Stands in for `AbstractRailEnv.collect` during an episode and records every call."""

    def __init__(self, width):
        self.width = width
        self.records = []

    def index(self, entry_point):
        """Flatland.jl's index of a grid entry point, 0 for None."""
        if entry_point is None:
            return 0
        (r, c), d = entry_point
        return 4 * (int(r) * self.width + int(c)) + int(d) + 1

    def collect(self, env, agent, action, candidate_entry_point_independent):
        result, record = recorded_collect(env, agent, action, candidate_entry_point_independent, self.index)
        self.records.append(record)
        return result


ORIGINAL_COLLECT = AbstractRailEnv.collect


def greedy_actions(env, rng, p_random):
    """Seeded actions, greedy on the distance map except with probability `p_random`."""
    distances = env.distance_map.get()
    actions = {}
    for agent in env.agents:
        if agent.current_entry_point is None or rng.random() < p_random:
            actions[agent.handle] = rng.randrange(5)
            continue
        best, best_distance = RailEnvActions.MOVE_FORWARD, np.inf
        for action in (RailEnvActions.MOVE_LEFT, RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_RIGHT):
            # actions apply when the agent enters its next entry point
            result = env.rail.apply_action_independent(action, agent.next_entry_point)
            if result is not None:
                (row, column), direction = result
                if distances[agent.handle, row, column, direction] < best_distance:
                    best, best_distance = action, distances[agent.handle, row, column, direction]
        actions[agent.handle] = best.value
    return actions


def config_record(seed, acceleration_delta, braking_delta, remove_agents_at_target, p_level_free, max_rail_pairs_in_city, steps=500):
    env = RailEnv(
        width=30, height=30, number_of_agents=10, random_seed=seed,
        rail_generator=sparse_rail_generator(max_num_cities=3, max_rails_between_cities=2, max_rail_pairs_in_city=max_rail_pairs_in_city,
                                             p_level_free=p_level_free),
        line_generator=sparse_line_generator({1.: .25, 1. / 2.: .25, 1. / 3.: .25, 1. / 4.: .25}),
        malfunction_generator=ParamMalfunctionGen(MalfunctionParameters(malfunction_rate=1 / 30, min_duration=2, max_duration=6)),
        obs_builder_object=DummyObservationBuilder(),
        remove_agents_at_target=remove_agents_at_target,
        acceleration_delta=acceleration_delta,
        braking_delta=braking_delta,
        check_step_pre_post_conditions=True,
    )
    env.reset(random_seed=seed)
    rng = random.Random(seed)
    return recorded_episode(env=env, action_dicts=(greedy_actions(env, rng, p_random=0.3) for _ in range(steps)))


def symmetric_switch_record():
    """
    One agent drives into the symmetric switch at (15, 15) from the west, where only turning is valid, as in
    `test_symmetric_switch_stop_action`: it is denied with straight-through actions at the cell exit, then turns.
    """
    env, _, _ = env_generator_legacy(seed=43, n_agents=1)
    env.braking_delta = -Fraction(1, 10)
    agent = env.agents[0]
    agent.initial_entry_point = ((15, 14), 1)
    script = [RailEnvActions.DO_NOTHING] * agent.earliest_departure + [
        RailEnvActions.MOVE_FORWARD, RailEnvActions.DO_NOTHING, RailEnvActions.STOP_MOVING, RailEnvActions.MOVE_FORWARD,
        RailEnvActions.MOVE_FORWARD, RailEnvActions.DO_NOTHING, RailEnvActions.MOVE_FORWARD, RailEnvActions.STOP_MOVING,
        RailEnvActions.MOVE_RIGHT, RailEnvActions.MOVE_RIGHT, RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_FORWARD,
    ]
    return recorded_episode(env=env, action_dicts=({agent.handle: action.value} for action in script))


def recorded_episode(env, action_dicts):
    """Step `env` with each of `action_dicts` until done, recording and thinning every `collect` call."""
    recorder = Recorder(width=env.width)
    AbstractRailEnv.collect = lambda self, agent, action, independent: recorder.collect(self, agent, action, independent)
    try:
        for actions in action_dicts:
            _, _, dones, _ = env.step(actions)
            if dones["__all__"]:
                break
    finally:
        AbstractRailEnv.collect = ORIGINAL_COLLECT
    kept, counts = [], {}
    for row, inputs, outputs in recorder.records:
        # thin by label, action, done/off-map situation and stopped
        key = (row, inputs.action, inputs.done, inputs.current_entry_point == 0, inputs.speed == [0, 1])
        if counts.get(key, 0) < PER_KEY:
            counts[key] = counts.get(key, 0) + 1
            kept.append([row, *inputs, *outputs])
    return {
        "acceleration_delta": fraction(env.acceleration_delta),
        "braking_delta": fraction(env.braking_delta),
        "remove_agents_at_target": env.remove_agents_at_target,
        "records": kept,
    }


def synthetic_agent(max_speed, off_map, done, in_malfunction, speed, distance, next_is_target, earliest_departure):
    speed_counter = SpeedCounter(max_speed)
    speed_counter._speed, speed_counter._distance = speed, distance
    return SimpleNamespace(
        handle=0, initial_entry_point=1, current_entry_point=None if off_map else 2, next_entry_point=None if off_map else 3,
        old_entry_point=None, malfunction_handler=SimpleNamespace(in_malfunction=in_malfunction), state=None,
        speed_counter=speed_counter, target_entry_point=(5 if off_map else 2) if done else None,
        targets=[3] if next_is_target else [9], earliest_departure=earliest_departure,
    )


def synthetic_config_record(max_speed, acceleration_delta, braking_delta, remove_agents_at_target):
    """
    Every speed and distance (in ticks) on the map, off-map agents before, at and after their departure step, done or
    not, in malfunction or not, with and without a target for the action, for every action, through the real
    `collect` on a stub env.
    """
    ticks = math.lcm(max_speed.denominator, acceleration_delta.denominator, braking_delta.denominator)
    env = SimpleNamespace(
        temp_transition_data={0: SimpleNamespace()}, _elapsed_steps=SYNTHETIC_ELAPSED_STEPS,
        acceleration_delta=acceleration_delta, braking_delta=braking_delta, remove_agents_at_target=remove_agents_at_target,
        check_step_pre_post_conditions=False,
        rail=SimpleNamespace(get_transitions=lambda entry_point: (1,)),
        resource_map=SimpleNamespace(get_resource=lambda entry_point, next_entry_point: (entry_point, next_entry_point)),
        _candidate_entry_points=AbstractRailEnv._candidate_entry_points,
        _candidate_speed=AbstractRailEnv._candidate_speed,
        _candidate_distance=AbstractRailEnv._candidate_distance,
    )
    # (off map, speed, distance, next is target, earliest departure)
    on_map = [(False, Fraction(s, ticks), Fraction(d, ticks), next_is_target, 0)
              for s in range(int(max_speed * ticks) + 1) for d in range(ticks + 1) for next_is_target in (False, True)]
    off_map = [(True, None, None, False, SYNTHETIC_ELAPSED_STEPS + k) for k in (-1, 0, 1)]
    kept, seen = [], set()
    for (off, speed, distance, next_is_target, earliest_departure), done, in_malfunction, independent, action in itertools.product(
            on_map + off_map, (False, True), (False, True), (None, 4), RailEnvActions):
        agent = synthetic_agent(max_speed=max_speed, off_map=off, done=done, in_malfunction=in_malfunction, speed=speed,
                                distance=distance, next_is_target=next_is_target, earliest_departure=earliest_departure)
        _, (row, inputs, outputs) = recorded_collect(env, agent, action, independent, opaque_index)
        cell_exit = not off and cached_cell_exit(max_speed, speed, distance)
        overshoot = not off and distance + speed > 1
        # a done agent's candidates do not depend on the action
        key = (row, None if done else action, done, off, speed == 0, in_malfunction, next_is_target, independent is None,
               cell_exit, overshoot, earliest_departure - SYNTHETIC_ELAPSED_STEPS)
        if key not in seen:
            seen.add(key)
            kept.append([row, *inputs, *outputs])
    return {
        "acceleration_delta": fraction(acceleration_delta),
        "braking_delta": fraction(braking_delta),
        "remove_agents_at_target": remove_agents_at_target,
        "records": kept,
    }


if __name__ == "__main__":
    episodes = [
        config_record(seed=1, acceleration_delta=Fraction(1), braking_delta=-Fraction(1), remove_agents_at_target=True, p_level_free=0.0,
                      max_rail_pairs_in_city=2),
        config_record(seed=2, acceleration_delta=Fraction(1, 2), braking_delta=-Fraction(1, 3), remove_agents_at_target=False, p_level_free=0.0,
                      max_rail_pairs_in_city=4),
        config_record(seed=3, acceleration_delta=Fraction(1, 4), braking_delta=-Fraction(1, 2), remove_agents_at_target=True, p_level_free=1.0,
                      max_rail_pairs_in_city=4),
        symmetric_switch_record(),
    ]
    synthetic = [synthetic_config_record(max_speed=max_speed, acceleration_delta=acceleration, braking_delta=braking, remove_agents_at_target=remove)
                 for (max_speed, acceleration, braking), remove in itertools.product(SYNTHETIC_PROFILES, (True, False))]
    for name, configs in (("episodes", episodes), ("synthetic", synthetic)):
        rows = [record[0] for config in configs for record in config["records"]]
        print(name, {row: rows.count(row) for row in TABLE_1_ROWS})
        missing = [row for row in TABLE_1_ROWS if row not in rows]
        assert not missing, f"Table 1 rows without {name} records: {missing}"
    with open(sys.argv[1], "w") as f:
        json.dump({"fields": FIELDS, "episodes": episodes, "synthetic": synthetic}, f, separators=(",", ":"))
