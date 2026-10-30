"""
Export the collect phase's per-agent candidates that Flatland.jl's tests compare against
(julia/Flatland/test/data/candidates.json).

Every call of `AbstractRailEnv.collect` in a few seeded episodes is recorded with its inputs (the agent's pre-step
state, the action and the entry point that action leads to) and its outputs (candidate entry points, speed and
distance). Each record is labelled with the row of `design_by_contract.md`'s Table 1 that it falls into, and the
records are thinned to at most `PER_KEY` per label, action and agent situation.

Usage: python tests/conformance/export_candidates.py julia/Flatland/test/data/candidates.json
"""
import json
import random
import sys
from fractions import Fraction

import numpy as np

from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.env_generation.env_generator import env_generator_legacy
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.malfunction_generators import MalfunctionParameters, ParamMalfunctionGen
from flatland.envs.rail_env import AbstractRailEnv, RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.step_utils.speed_counter import cached_cell_exit

PER_KEY = 12

TABLE_1_ROWS = [
    "Done", "Target reached", "Malfunction", "Map entry", "Stay off map", "On-map cell transition",
    "Invalid action at cell exit", "Stopped", "Acceleration or start moving", "Braking", "Default",
]


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


def index(env, entry_point):
    """Flatland.jl's index of a grid entry point, 0 for None."""
    if entry_point is None:
        return 0
    (r, c), d = entry_point
    return 4 * (int(r) * env.width + int(c)) + int(d) + 1


def fraction(x):
    return None if x is None else [Fraction(x).numerator, Fraction(x).denominator]


class Recorder:
    def __init__(self):
        self.records = []

    def collect(self, env, agent, action, independent):
        state = (
            index(env, agent.current_entry_point), index(env, agent.next_entry_point), index(env, agent.initial_entry_point),
            agent.next_entry_point is not None and agent.next_entry_point in agent.targets,
            agent.target_entry_point is not None, bool(agent.malfunction_handler.in_malfunction),
            fraction(agent.speed_counter.speed), fraction(agent.speed_counter.distance), fraction(agent.speed_counter.max_speed),
            int(agent.earliest_departure), int(env._elapsed_steps), int(action.value), index(env, independent),
        )
        result = ORIGINAL_COLLECT(env, agent, action, independent)
        data = env.temp_transition_data[agent.handle]
        row = table_1_row(agent, action, independent, data.candidate_entry_point)
        outputs = (
            index(env, data.candidate_entry_point), index(env, data.candidate_next_entry_point),
            fraction(data.candidate_speed), fraction(data.candidate_distance),
        )
        self.records.append((row, state, outputs))
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
    return recorded_episode(env, (greedy_actions(env, rng, p_random=0.3) for _ in range(steps)))


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
    return recorded_episode(env, ({agent.handle: action.value} for action in script))


def recorded_episode(env, action_dicts):
    """Step `env` with each of `action_dicts` until done, recording and thinning every `collect` call."""
    recorder = Recorder()
    AbstractRailEnv.collect = lambda self, agent, action, independent: recorder.collect(self, agent, action, independent)
    try:
        for actions in action_dicts:
            _, _, dones, _ = env.step(actions)
            if dones["__all__"]:
                break
    finally:
        AbstractRailEnv.collect = ORIGINAL_COLLECT
    kept, counts = [], {}
    for row, state, outputs in recorder.records:
        # thin by label, action, done/off-map situation and stopped
        key = (row, state[11], state[4], state[0] == 0, state[6] == [0, 1])
        if counts.get(key, 0) < PER_KEY:
            counts[key] = counts.get(key, 0) + 1
            kept.append([row, *state, *outputs])
    return {
        "width": env.width,
        "acceleration_delta": fraction(env.acceleration_delta),
        "braking_delta": fraction(env.braking_delta),
        "remove_agents_at_target": env.remove_agents_at_target,
        "records": kept,
    }


if __name__ == "__main__":
    configs = [
        config_record(seed=1, acceleration_delta=Fraction(1), braking_delta=-Fraction(1), remove_agents_at_target=True, p_level_free=0.0,
                      max_rail_pairs_in_city=2),
        config_record(seed=2, acceleration_delta=Fraction(1, 2), braking_delta=-Fraction(1, 3), remove_agents_at_target=False, p_level_free=0.0,
                      max_rail_pairs_in_city=4),
        config_record(seed=3, acceleration_delta=Fraction(1, 4), braking_delta=-Fraction(1, 2), remove_agents_at_target=True, p_level_free=1.0,
                      max_rail_pairs_in_city=4),
        symmetric_switch_record(),
    ]
    rows = {record[0] for config in configs for record in config["records"]}
    print({row: sum(record[0] == row for config in configs for record in config["records"]) for row in TABLE_1_ROWS})
    missing = [row for row in TABLE_1_ROWS if row not in rows]
    assert not missing, f"Table 1 rows without records: {missing}"
    with open(sys.argv[1], "w") as f:
        json.dump({"fields": ["row", "current_entry_point", "next_entry_point", "initial_entry_point", "next_is_target", "done",
                              "in_malfunction", "speed", "distance", "max_speed", "earliest_departure", "elapsed_steps", "action",
                              "independent", "candidate_entry_point", "candidate_next_entry_point", "candidate_speed",
                              "candidate_distance"],
                   "configs": configs}, f, separators=(",", ":"))
