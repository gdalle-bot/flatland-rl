"""
Draws of numpy's legacy `RandomState` and of `ParamMalfunctionGen`, with the random states before and after, that
Flatland.jl's tests compare against (see `random_state_data` and `malfunction_data`).
"""
import numpy as np
from numpy.random import RandomState

from flatland.envs.malfunction_generators import MalfunctionParameters, NBR_CHACHED_RAND, ParamMalfunctionGen
from flatland.utils.seeding import np_random

# (low, high) of `randint`: empty, small and large ranges, the boundaries of numpy's 32-bit and 64-bit paths
RANDINT_RANGES = [
    (0, 1), (0, 2), (3, 7), (-5, 5), (1, 2 ** 31), (0, 2 ** 32 - 1), (0, 2 ** 32), (0, 2 ** 32 + 1),
    (-(2 ** 40), 2 ** 40), (0, 2 ** 63 - 1),
]

# (rate, min_duration, max_duration): no malfunctions, the defaults of the tests, a single duration, long ones, and
# a rate whose probability is above 1/2
MALFUNCTION_PARAMETERS = [(0.0, 0, 0), (1 / 30, 2, 6), (0.5, 1, 1), (0.2, 0, 20), (2.0, 5, 9)]


def state(rs):
    """The Mersenne Twister state of `rs`: its 624 words and its position."""
    _, key, pos, _, _ = rs.get_state()
    return {"key": np.array(key, dtype=np.int64), "pos": int(pos)}


def draws(rs):
    """1500 `random_sample`s (more than one regeneration of the state), then `randint` for each range, 20 times."""
    initial = state(rs)
    samples = np.array([rs.random_sample() for _ in range(1500)])
    randints = [[low, high, int(rs.randint(low, high))] for _ in range(20) for low, high in RANDINT_RANGES]
    return {"initial": initial, "samples": samples, "randints": randints, "final": state(rs)}


def random_state_data():
    """Draws from `RandomState(seed)` for integer and array seeds, and from Flatland's `seeding.np_random`."""
    records = []
    for seed in [0, 1, 42, 2 ** 32 - 1, [1, 2, 3], [2 ** 32 - 1] * 700, list(range(624))]:
        records.append({"seed": seed, **draws(RandomState(seed))})
    for seed in [0, 7, 123456]:
        records.append({"seed": None, **draws(np_random(seed)[0])})
    return records


def duration(malfunction):
    return 0 if malfunction is None else int(malfunction.num_broken_steps)


def malfunction_data(calls=3000):
    """
    For each parameter set and two seeds, as the env does (`np_random`): the random state, then the durations of
    `calls` calls of `generate`, then the random state and the position in the cache. Plus, from a generator moved
    close to the end of its cache, the generator and random states and the durations of the next 40 calls.
    """
    records = []
    for rate, min_duration, max_duration in MALFUNCTION_PARAMETERS:
        for seed in (0, 1):
            rs = np_random(seed)[0]
            gen = ParamMalfunctionGen(MalfunctionParameters(rate, min_duration, max_duration))
            initial = state(rs)
            durations = np.array([duration(gen.generate(rs)) for _ in range(calls)])
            fresh = {"rate": rate, "min_duration": min_duration, "max_duration": max_duration, "probability": gen._malfunction_prob}
            records.append({**fresh, "initial": initial, "durations": durations, "final": state(rs), "index": gen._rand_idx})
            # resume close to the end of the cache, so that the position wraps around
            gen._rand_idx = NBR_CHACHED_RAND - 20
            resumed = {"initial": state(rs), "broken": np.array(gen._cached_broken, dtype=bool), "index": gen._rand_idx}
            durations = np.array([duration(gen.generate(rs)) for _ in range(40)])
            records.append({**fresh, **resumed, "durations": durations, "final": state(rs), "final_index": gen._rand_idx})
    return records
