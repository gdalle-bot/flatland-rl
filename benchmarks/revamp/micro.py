"""Micro-benchmarks backing specific claims of REVAMP.md (section 2.4).

Usage: python benchmarks/revamp/micro.py T0 [OUT_DIR]
  T0: step of scenario M at which to measure the observation builders (1240 in REVAMP.md, the step with the most agents
  on the map in flatbench's M/b run). Writes OUT_DIR/micro.json (default revamp_out).
"""
import json
import os
import sys
import timeit
from fractions import Fraction
from time import perf_counter_ns as now

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import flatbench  # noqa: E402
from flatland.envs import observations as obs_mod  # noqa: E402
from flatland.envs.agent_chains import MotionCheck  # noqa: E402
from flatland.envs.step_utils.speed_counter import SpeedCounter, cached_cell_exit  # noqa: E402

OUT = {}


def best(stmt, number, repeat=7, glb=None):
    return min(timeit.repeat(stmt, number=number, repeat=repeat, globals=glb)) / number


# 1. Fraction vs integer ticks in the speed-counter pattern -------------------------------------------
def fraction_vs_int():
    D = 12
    samples_f, samples_i = [], []
    for den in (1, 2, 3, 4):
        m = Fraction(1, den)
        for k in range(den + 1):
            s = Fraction(k, den)
            for j in range(D):
                d = Fraction(j, D)
                samples_f.append((m, s, d))
                samples_i.append((D // den, k * (D // den), j))
    one = Fraction(1)

    def frac_cached():
        for m, s, d in samples_f:
            cached_cell_exit(m, s, d)
            SpeedCounter.speed_after_acceleration(s, m, one)
            SpeedCounter.distance_after_crossing(d, s)
            SpeedCounter.distance_without_crossing(d, s)
            nd = (d + s) % 1
            _ = d + s >= 1 and nd < 1

    def frac_uncached():
        for m, s, d in samples_f:
            v = max(Fraction(0), min(m, s))
            _ = v != 0 and d + v >= 1
            min(s + one, m)
            nd = (d + s) % 1
            min(d + s, one)
            _ = d + s >= 1 and nd < 1

    def ints():
        for m, s, d in samples_i:
            v = s if s < m else m
            _ = v != 0 and d + v >= D
            min(s + D, m)
            nd = (d + s) % D
            min(d + s, D)
            _ = d + s >= D and nd < D

    n = len(samples_f)
    frac_cached()  # warm caches
    res = {k: best(f, 50) / n * 1e9 for k, f in
           [("fraction_lru_ns", frac_cached), ("fraction_uncached_ns", frac_uncached), ("int_ticks_ns", ints)]}
    F = Fraction(1, 3)
    res["F_eq_int0_ns"] = best("F == 0", 200000, glb={"F": F}) * 1e9
    res["F_eq_float0_ns"] = best("F == 0.0", 200000, glb={"F": F}) * 1e9
    res["F_plus_F_ns"] = best("F + G", 200000, glb={"F": F, "G": Fraction(1, 4)}) * 1e9
    res["int_plus_int_ns"] = best("a + b", 200000, glb={"a": 4, "b": 3}) * 1e9
    res["hash_F_ns"] = best("hash(F)", 200000, glb={"F": F}) * 1e9
    res["samples"] = n
    OUT["fraction_vs_int"] = res


# 2. The 1M-number malfunction cache -------------------------------------------------------------------
def rand1m():
    rs = np.random.RandomState(1)
    t = best("rs.rand(1_000_000) < 0.0001", 3, glb={"rs": rs})
    OUT["rand1M_ms"] = t * 1e3


def env_at(scen, cfg, t0):
    env = flatbench.make_env(scen, cfg, 1)
    env.reset(random_seed=1)
    actions = np.random.default_rng(12345).integers(0, 5, (env._max_episode_steps + 2, env.get_num_agents()), dtype=np.int8)
    flatbench.fast_forward(env, actions, t0)
    return env


# 3. Dead conflict-check branch in TreeObsForRailEnv ---------------------------------------------------
def dead_branch(t0):
    env = env_at("M", "c", t0)
    ob = env.obs_builder
    handles = env.get_agent_handles()
    counts = {"fast_delete": 0, "fast_where": 0}
    orig_del, orig_where = obs_mod.fast_delete, obs_mod.fast_where

    def counting_del(lis, index):
        counts["fast_delete"] += 1
        return orig_del(lis, index)

    def counting_where(x):
        counts["fast_where"] += 1
        return orig_where(x)

    obs_mod.fast_delete, obs_mod.fast_where = counting_del, counting_where
    ref = repr(ob.get_many(handles))
    obs_mod.fast_delete, obs_mod.fast_where = orig_del, orig_where

    def timed(patched):
        obs_mod.fast_delete = (lambda lis, index: ()) if patched else orig_del
        ts = []
        for _ in range(5):
            a = now()
            o = ob.get_many(handles)
            ts.append(now() - a)
        obs_mod.fast_delete = orig_del
        return min(ts), repr(o)

    t_orig, r_orig = timed(False)
    t_patch, r_patch = timed(True)
    # predictor share of get_many
    pt = []
    for _ in range(5):
        a = now()
        ob.predictor.get()
        pt.append(now() - a)
    OUT["dead_branch"] = dict(t0=t0, n_agents=len(handles), on_map=flatbench.covariates(env)["n_on_map"],
                              get_many_ms=t_orig / 1e6, get_many_without_branch_ms=t_patch / 1e6,
                              predictor_ms=min(pt) / 1e6, identical_obs=(ref == r_orig == r_patch),
                              fast_delete_calls=counts["fast_delete"], fast_where_calls=counts["fast_where"])


# 4. GlobalObsForRailEnv per-agent asserts -------------------------------------------------------------
def global_obs(t0):
    env = env_at("M", "a", t0)
    ob = env.obs_builder
    h = env.get_agent_handles()
    a = now()
    res = [ob.get(i) for i in h]
    t_get = now() - a
    rail_obs, s, tg = res[0]
    stmt = ("np.count_nonzero(~np.isfinite(r)); np.count_nonzero(np.isnan(r)); np.count_nonzero(~np.isfinite(s)); "
            "np.count_nonzero(np.isnan(s)); np.count_nonzero(~np.isfinite(t)); np.count_nonzero(np.isnan(t))")
    t_asserts = best(stmt, 20, glb={"np": np, "r": rail_obs, "s": s, "t": tg})
    OUT["global_obs"] = dict(n_agents=len(h), get_all_ms=t_get / 1e6, asserts_per_agent_ms=t_asserts * 1e3,
                             asserts_all_agents_ms=t_asserts * 1e3 * len(h))


# 5. Distance map ---------------------------------------------------------------------------------------
def distance_map():
    env = flatbench.make_env("M", "b", 1)
    env.reset(random_seed=1)
    dm = env.distance_map
    ts = []
    for _ in range(3):
        a = now()
        dm._compute(env.agents, env.rail)
        ts.append(now() - a)
    n_unique = len({frozenset(dm._valid_targets(a, env.rail)) for a in env.agents})
    dedupe = {}
    rng = np.random.default_rng(0)
    for n in (200, 500, 2000):
        pool = [[((int(r), int(c)), d) for d in range(4)] for r, c in rng.integers(0, 100, (60, 2))]
        targets = [pool[i] for i in rng.integers(0, len(pool), n)]

        def lst():
            seen = []
            for t in targets:
                if t not in seen:
                    pass
                seen.append(t)

        def dct():
            seen = {}
            for t in targets:
                k = frozenset(t)
                if k not in seen:
                    seen[k] = len(seen)

        dedupe[n] = dict(list_ms=best(lst, 1, 3) * 1e3, dict_ms=best(dct, 1, 3) * 1e3)
    OUT["distance_map"] = dict(compute_ms=min(ts) / 1e6, n_agents=env.get_num_agents(), unique_target_sets=n_unique,
                               array_mb=dm.distance_map.nbytes / 2 ** 20, dedupe=dedupe)


# 6. MotionCheck growth with contention ----------------------------------------------------------------
def motion_check():
    res = {}
    for m in (10, 25, 50, 100, 200):
        def run():
            mc = MotionCheck()
            for i in range(m):
                mc.add_agent(i, (i + 1, 0), (0, 0))
            mc.find_conflicts()

        res[m] = best(run, 1, 3) * 1e3
    ms = sorted(res)
    slope = np.polyfit(np.log(ms[1:]), np.log([res[m] for m in ms[1:]]), 1)[0]
    OUT["motion_check"] = dict(ms_by_contenders=res, growth_exponent=float(slope))


# 7. lru-cached transition lookups vs plain dict / numpy table ------------------------------------------
def lookups():
    env = flatbench.make_env("M", "b", 1)
    env.reset(random_seed=1)
    rail = env.rail
    eps = [(a.initial_entry_point[0], a.initial_entry_point[1]) for a in env.agents]
    for ep in eps:
        rail.get_transitions(ep)
    table = {ep: rail.get_transitions(ep) for ep in eps}
    arr = np.zeros((rail.height * rail.width * 4, 4), dtype=np.int8)
    idx = [((r * rail.width + c) * 4 + d) for (r, c), d in eps]
    OUT["lookups_ns"] = dict(
        lru_get_transitions=best("for ep in eps: rail.get_transitions(ep)", 200, glb={"eps": eps, "rail": rail}) / len(eps) * 1e9,
        dict_lookup=best("for ep in eps: table[ep]", 200, glb={"eps": eps, "table": table}) / len(eps) * 1e9,
        list_index=best("for i in idx: rows[i]", 200, glb={"idx": idx, "rows": list(arr)}) / len(eps) * 1e9,
    )


if __name__ == "__main__":
    t0 = int(sys.argv[1])
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "revamp_out"
    os.makedirs(out_dir, exist_ok=True)
    for f in (fraction_vs_int, rand1m, motion_check, lookups, distance_map):
        f()
        print(f.__name__, "done", flush=True)
    dead_branch(t0)
    print("dead_branch done", flush=True)
    global_obs(t0)
    print("global_obs done", flush=True)
    with open(os.path.join(out_dir, "micro.json"), "w") as fh:
        json.dump(OUT, fh, indent=1, default=str)
    print(json.dumps(OUT, indent=1, default=str))
