"""Phase-level timing of Flatland's RailEnv.reset()/step(), as reported in REVAMP.md section 2.

Usage: python benchmarks/revamp/flatbench.py SCEN CFG MODE [--reps R] [--early N] [--late N] [--t0 T] [--seed S]
                                             [--out DIR] [--tag TAG]
  SCEN in S, M, L (L2, L3: smaller fallbacks of L); CFG in a (the constructor defaults of commit 51f5621c: global
  obs, checks on), b (dummy obs, checks off), c (tree obs, checks off), d (dummy obs, checks on); MODE in timing
  (wall-clock phases), cprofile (.prof files). The step checks are always passed explicitly, since their default
  changed after 51f5621c.
Writes OUT/SCEN_CFG_MODE[TAG].json. See benchmarks/revamp/README.md.
"""
import argparse
import cProfile
import gc
import hashlib
import json
import os
import resource
import sys
import time
from time import perf_counter_ns as now

import numpy as np

from flatland.core.distance_map import AgentSourceTargetDistanceMap
from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.envs import agent_chains
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.malfunction_generators import MalfunctionParameters, ParamMalfunctionGen
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator

SCEN = {  # width, height, agents, cities
    "S": (30, 30, 7, 2),
    "M": (100, 100, 200, 5),
    "L": (158, 158, 500, 41),
    "L2": (150, 150, 500, 37),
    "L3": (120, 120, 500, 25),
}
SPEEDS = {1.: .25, 1. / 2.: .25, 1. / 3.: .25, 1. / 4.: .25}


def make_env(scen, cfg, seed):
    w, h, n, c = SCEN[scen]
    kw = dict(width=w, height=h, number_of_agents=n, random_seed=seed,
              rail_generator=sparse_rail_generator(max_num_cities=c, max_rails_between_cities=2, max_rail_pairs_in_city=2),
              line_generator=sparse_line_generator(SPEEDS),
              malfunction_generator=ParamMalfunctionGen(MalfunctionParameters(malfunction_rate=1 / 10000, min_duration=15, max_duration=50)))
    if cfg == "a":
        kw.update(check_step_pre_post_conditions=True)
    elif cfg == "b":
        kw.update(obs_builder_object=DummyObservationBuilder(), check_step_pre_post_conditions=False)
    elif cfg == "c":
        kw.update(obs_builder_object=TreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv(max_depth=50)),
                  check_step_pre_post_conditions=False)
    elif cfg == "d":
        kw.update(obs_builder_object=DummyObservationBuilder(), check_step_pre_post_conditions=True)
    return RailEnv(**kw)


class Clock:
    """Records (label, t0, t1, parent_label) for wrapped calls."""

    def __init__(self):
        self.stack = ["-"]
        self.ev = []

    def wrap(self, fn, label):
        def wrapped(*a, **k):
            self.stack.append(label)
            t0 = now()
            try:
                return fn(*a, **k)
            finally:
                t1 = now()
                self.stack.pop()
                self.ev.append((label, t0, t1, self.stack[-1]))

        return wrapped


CLK = Clock()
INSTANCE = ["_call_rail_generator", "line_generator", "timetable_generator", "_agents_from_line", "reset_agents",
            "_get_observations", "get_info_dict", "_capture_pre_step_snapshot", "end_of_episode_update",
            "_update_agent_positions_map", "_check_pre_post_invariants", "_check_speed_distance_speedup_postconditions",
            "_check_position_update_postconditions"]
CHECKS = {"_check_pre_post_invariants", "_check_speed_distance_speedup_postconditions", "_check_position_update_postconditions"}


def patch_classes():
    def defining(cls, name):
        for k in cls.__mro__:
            if name in vars(k):
                return k

    k = defining(AgentSourceTargetDistanceMap, "_compute")
    k._compute = CLK.wrap(k._compute, "dmap")
    mc = agent_chains.MotionCheck
    mc.find_conflicts = CLK.wrap(mc.find_conflicts, "find_conflicts")
    ParamMalfunctionGen.generate_rand_numbers = CLK.wrap(ParamMalfunctionGen.generate_rand_numbers, "rand1M")


def instrument(env):
    for name in INSTANCE:
        setattr(env, name, CLK.wrap(getattr(env, name), name))
    eg, ob = env.effects_generator, env.obs_builder
    eg.on_episode_step_start = CLK.wrap(eg.on_episode_step_start, "malf")
    eg.on_episode_step_end = CLK.wrap(eg.on_episode_step_end, "effects_end")
    ob.reset = CLK.wrap(ob.reset, "obs_reset")
    if getattr(ob, "predictor", None) is not None:
        ob.predictor.get = CLK.wrap(ob.predictor.get, "pred")


def dur(evs, label):
    return sum(t1 - t0 for (l, t0, t1, _) in evs if l == label)


def step_phases(evs, t0, t1):
    first = {}
    for (l, a, b, _) in evs:
        first.setdefault(l, (a, b))
    p = {"total": t1 - t0}
    checks = sum(dur(evs, c) for c in CHECKS)
    pre_check = sum(b - a for (l, a, b, _) in evs if l == "_check_pre_post_invariants" and a < first.get("malf", (t1, t1))[0])
    p["checks"] = checks
    p["malf"] = dur(evs, "malf") - dur(evs, "rand1M")
    p["rand1M"] = dur(evs, "rand1M")
    p["head"] = first["malf"][0] - t0 - pre_check
    p["snapshot"] = dur(evs, "_capture_pre_step_snapshot")
    p["collect_loop"] = first["find_conflicts"][0] - first["_capture_pre_step_snapshot"][1]
    p["motion_check"] = dur(evs, "find_conflicts")
    p["distribute_loop"] = first["end_of_episode_update"][0] - first["find_conflicts"][1]
    p["eoe"] = dur(evs, "end_of_episode_update")  # includes the lazy distance map on the last step
    p["effects_end"] = dur(evs, "effects_end")
    p["pred"] = dur(evs, "pred")
    p["obs"] = dur(evs, "_get_observations") - p["pred"]
    p["info"] = dur(evs, "get_info_dict")
    p["posmap"] = dur(evs, "_update_agent_positions_map")
    p["dmap"] = dur(evs, "dmap")
    p["residual"] = p["total"] - sum(v for k, v in p.items() if k not in ("total", "dmap"))
    return p


def reset_phases(evs, t0, t1):
    p = {"total": t1 - t0}
    dm = [(t1_ - t0_, par) for (l, t0_, t1_, par) in evs if l == "dmap"]
    p["dmap_timetable"] = sum(d for d, par in dm if par == "timetable_generator")
    p["dmap_other"] = sum(d for d, par in dm if par != "timetable_generator")
    p["rail"] = dur(evs, "_call_rail_generator")
    p["line"] = dur(evs, "line_generator")
    p["agents"] = dur(evs, "_agents_from_line")
    p["timetable"] = dur(evs, "timetable_generator") - p["dmap_timetable"]
    p["reset_agents"] = dur(evs, "reset_agents")
    p["obs_reset"] = dur(evs, "obs_reset") - sum(d for d, par in dm if par == "obs_reset")
    p["init_obs"] = dur(evs, "_get_observations") - sum(d for d, par in dm if par in ("_get_observations", "pred"))
    p["info"] = dur(evs, "get_info_dict")
    p["other"] = p["total"] - sum(v for k, v in p.items() if k != "total")
    return p


def covariates(env):
    on_map = moving = malf = done = waiting = 0
    for a in env.agents:
        if a.target_entry_point is not None:
            done += 1
        elif a.current_entry_point is None:
            waiting += 1
        else:
            on_map += 1
            moving += int(bool(a.speed_counter.speed))
            malf += int(a.malfunction_handler.in_malfunction)
    return dict(n_on_map=on_map, n_moving=moving, n_malf=malf, n_done=done, n_waiting=waiting)


def fingerprint(env):
    return hashlib.sha1(repr([(a.current_entry_point, a.speed_counter.speed, a.speed_counter.distance,
                               a.malfunction_handler.malfunction_down_counter, a.target_entry_point)
                              for a in env.agents]).encode()).hexdigest()[:12]


def act(actions, env):
    return {h: int(x) for h, x in enumerate(actions[env._elapsed_steps])}


GC_NS = [0]
_gc_t0 = [0]


def _gc_cb(phase, info):
    if phase == "start":
        _gc_t0[0] = now()
    else:
        GC_NS[0] += now() - _gc_t0[0]


def fast_forward(env, actions, t0):
    ob, chk = env.obs_builder, env.check_step_pre_post_conditions
    env.obs_builder, env.check_step_pre_post_conditions = DummyObservationBuilder(), False
    env.obs_builder.reset(env)
    while env._elapsed_steps < t0 and not env.dones["__all__"]:
        env.step(act(actions, env))
    env.obs_builder, env.check_step_pre_post_conditions = ob, chk
    a = now()
    ob.reset(env)
    env.distance_map.get()
    return now() - a


def maxrss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run(args):
    out = dict(scen=args.scen, cfg=args.cfg, mode=args.mode, seed=args.seed, reps=[])
    gc.callbacks.append(_gc_cb)
    if args.mode == "timing":
        patch_classes()
    for rep in range(args.reps):
        CLK.ev.clear()
        GC_NS[0] = 0
        env = make_env(args.scen, args.cfg, args.seed)
        prof = cProfile.Profile() if args.mode == "cprofile" else None
        if args.mode == "timing":
            instrument(env)
        t0 = now()
        if prof:
            prof.runcall(env.reset, random_seed=args.seed)
        else:
            env.reset(random_seed=args.seed)
        t1 = now()
        r = dict(rep=rep, reset_ns=t1 - t0, max_episode_steps=env._max_episode_steps, n_agents=env.get_num_agents(),
                 rss_after_reset_mb=maxrss_mb(), gc_reset_ns=GC_NS[0])
        if args.mode == "timing":
            r["reset_phases"] = reset_phases(CLK.ev, t0, t1)
        if prof:
            prof.dump_stats(f"{args.out}/{args.scen}_{args.cfg}_reset.prof")
        rng = np.random.default_rng(12345)
        actions = rng.integers(0, 5, (env._max_episode_steps + 2, env.get_num_agents()), dtype=np.int8)
        windows = [("early", args.early)]
        if args.late:
            windows.append(("late", args.late))
        r["windows"] = {}
        for wname, nsteps in windows:
            if wname == "late":
                r["attach_ns"] = fast_forward(env, actions, args.t0)
            if prof:
                prof = cProfile.Profile()
            steps = []
            GC_NS[0] = 0
            while not env.dones["__all__"] and (nsteps <= 0 or len(steps) < nsteps):
                acts = act(actions, env)
                CLK.ev.clear()
                if prof:
                    prof.enable()
                a = now()
                env.step(acts)
                b = now()
                if prof:
                    prof.disable()
                rec = dict(t=env._elapsed_steps, total=b - a)
                if args.mode == "timing":
                    rec.update(step_phases(CLK.ev, a, b))
                    if env._elapsed_steps % 5 == 0 or env.dones["__all__"]:
                        rec.update(covariates(env))
                steps.append(rec)
                if maxrss_mb() > 2800:
                    raise MemoryError(f"RSS {maxrss_mb()} MB")
            if prof:
                prof.dump_stats(f"{args.out}/{args.scen}_{args.cfg}_steps_{wname}.prof")
            r["windows"][wname] = dict(steps=steps, gc_ns=GC_NS[0], fingerprint=fingerprint(env))
        r["rss_mb"] = maxrss_mb()
        out["reps"].append(r)
        del env
        gc.collect()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scen")
    ap.add_argument("cfg")
    ap.add_argument("mode")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--early", type=int, default=0, help="0 = until done")
    ap.add_argument("--late", type=int, default=0)
    ap.add_argument("--t0", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="revamp_out")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t = time.time()
    res = run(args)
    res["wall_s"] = time.time() - t
    res["python"] = sys.version
    name = f"{args.out}/{args.scen}_{args.cfg}_{args.mode}{args.tag}.json"
    with open(name, "w") as f:
        json.dump(res, f)
    print("wrote", name, f"{res['wall_s']:.1f}s", f"rss={res['reps'][-1]['rss_mb']:.0f}MB")
