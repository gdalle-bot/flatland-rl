"""Turn flatbench outputs into Markdown tables for REVAMP.md. Usage: python benchmarks/revamp/analyze.py [OUT_DIR]"""
import glob
import json
import os
import pstats
import statistics as st
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "revamp_out"
PH = ["head", "malf", "snapshot", "collect_loop", "motion_check", "distribute_loop", "eoe", "effects_end", "checks", "obs", "pred",
      "info", "posmap", "residual"]
RPH = ["rail", "line", "agents", "timetable", "dmap_timetable", "dmap_other", "reset_agents", "obs_reset", "init_obs", "info", "other"]


def load(name):
    p = f"{OUT}/{name}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def reps(d):
    r = d["reps"]
    return r[1:] if len(r) > 1 else r


def mean_reset(d):
    rs = reps(d)
    return {k: st.mean(r["reset_phases"][k] for r in rs) / 1e6 for k in ["total"] + RPH}


def window_stats(d, w):
    """Per-step means (µs) over the window, steady steps only (not first/last), plus first/last step totals."""
    out = []
    for r in reps(d):
        s = r["windows"][w]["steps"]
        steady = s[1:-1] if w == "early" and len(s) > 2 else s
        m = {k: st.mean(x[k] for x in steady) / 1e3 for k in ["total"] + PH}
        m["n_steps"] = len(s)
        m["sum_ms"] = sum(x["total"] for x in s) / 1e6
        m["first_ms"] = s[0]["total"] / 1e6
        m["last_ms"] = s[-1]["total"] / 1e6
        m["gc_ms"] = r["windows"][w]["gc_ns"] / 1e6
        m["fp"] = r["windows"][w]["fingerprint"]
        out.append(m)
    return {k: (st.mean(o[k] for o in out) if k != "fp" else sorted({o["fp"] for o in out})) for k in out[0]}


def cv(d, w="early"):
    xs = [sum(x["total"] for x in r["windows"][w]["steps"][1:-1]) for r in reps(d)]
    return st.pstdev(xs) / st.mean(xs) if len(xs) > 1 else float("nan")


def fmt(x, nd=1):
    return f"{x:.{nd}f}"


def reset_table(scens):
    print("\n### reset() phases (ms, mean over reps)\n")
    print("| scenario | total | " + " | ".join(RPH) + " |")
    print("|---|" + "---|" * (len(RPH) + 1))
    for sc in scens:
        d = load(f"{sc}_b_timing")
        if not d:
            continue
        m = mean_reset(d)
        print(f"| {sc} | {fmt(m['total'])} | " + " | ".join(fmt(m[k]) for k in RPH) + " |")


def step_table(scens, cfgs=("b", "d", "a", "c")):
    print("\n### step() phases (µs per step, steady-state mean; share of step in parentheses)\n")
    print("| scen/cfg/window | steps | step µs | steps/s | " + " | ".join(PH) + " |")
    print("|---|" + "---|" * (len(PH) + 3))
    for sc in scens:
        for c in cfgs:
            d = load(f"{sc}_{c}_timing")
            if not d:
                continue
            for w in d["reps"][0]["windows"]:
                m = window_stats(d, w)
                tot = m["total"]
                cells = " | ".join(f"{fmt(m[k], 0)} ({fmt(100 * m[k] / tot, 0)}%)" if m[k] / tot >= 0.005 else fmt(m[k], 0) for k in PH)
                print(f"| {sc}/{c}/{w} | {m['n_steps']} | {fmt(tot, 0)} | {fmt(1e6 / tot, 0)} | {cells} |")


def episode_table(scens):
    print("\n### Full episode (config b and d): reset + all steps\n")
    print("| scen/cfg | reset ms | steps | Σ steps ms | first step ms | last step ms | GC ms | rep CV | fingerprints |")
    print("|---|---|---|---|---|---|---|---|---|")
    for sc in scens:
        for c in ("b", "d"):
            d = load(f"{sc}_{c}_timing")
            if not d:
                continue
            m = window_stats(d, "early")
            rm = st.mean(r["reset_ns"] for r in reps(d)) / 1e6
            print(f"| {sc}/{c} | {fmt(rm)} | {m['n_steps']} | {fmt(m['sum_ms'])} | {fmt(m['first_ms'], 2)} | {fmt(m['last_ms'], 2)} | "
                  f"{fmt(m['gc_ms'])} | {fmt(100 * cv(d), 1)}% | {','.join(m['fp'])} |")


def prof_table():
    print("\n### cProfile highlights\n")
    keys = {
        "fractions.py (Σ tottime)": None,
        "fast_delete": ("fast_methods.py", "fast_delete"),
        "coordinate_to_position": ("grid_utils.py", "coordinate_to_position"),
        "_explore_branch": ("observations.py", "_explore_branch"),
        "ShortestPathPredictor.get": ("predictions.py", "get"),
        "_reconstruct_shortest_path": ("distance_map.py", "_reconstruct_shortest_path"),
        "obs builder .get (Global or Tree)": ("observations.py", "get"),
        "find_conflicts": ("agent_chains.py", "find_conflicts"),
        "step_reward": ("rewards.py", "step_reward"),
        "__hash__ (transition_map)": ("transition_map.py", "__hash__"),
        "derived_state": ("agent_utils.py", "derived_state"),
        "_compute (dmap)": ("distance_map.py", "_compute"),
        "_walk_to_target": ("distance_map_walker.py", "_walk_to_target"),
        "a_star": ("grid4_astar.py", "a_star"),
        "check_path_exists": ("transition_map.py", "check_path_exists"),
        "_fix_transitions": ("rail_generators.py", "_fix_transitions"),
    }
    for f in sorted(glob.glob(f"{OUT}/*.prof")):
        s = pstats.Stats(f)
        total = s.total_tt
        print(f"\n**{os.path.basename(f)}** (total {fmt(total, 2)} s profiled)\n")
        rows = []
        frac = sum(v[2] for k, v in s.stats.items() if k[0].endswith("fractions.py"))
        rows.append(("fractions.py (Σ tottime)", "", frac))
        for label, key in keys.items():
            if key is None:
                continue
            best = None
            for k, v in s.stats.items():
                if k[0].endswith(key[0]) and k[2] == key[1]:
                    if best is None or v[3] > best[3]:
                        best = v
            if best is not None:
                rows.append((label, best[1], best[3]))
        print("| function | ncalls | cumtime s | % of profiled |")
        print("|---|---|---|---|")
        for label, n, ct in rows:
            print(f"| {label} | {n} | {fmt(ct, 3)} | {fmt(100 * ct / total, 1)}% |")


if __name__ == "__main__":
    scens = [s for s in ("S", "M", "L", "L2", "L3") if load(f"{s}_b_timing")]
    reset_table(scens)
    episode_table(scens)
    step_table(scens)
    for extra in ("M_b_timing_seed2", "M_b_timing_seed3", "M_b_timing_pass2"):
        d = load(extra)
        if d:
            m = window_stats(d, "early")
            print(f"\n{extra}: steps={m['n_steps']} step µs={fmt(m['total'], 0)} Σ={fmt(m['sum_ms'])} ms cv={fmt(100 * cv(d), 1)}%")
    prof_table()
