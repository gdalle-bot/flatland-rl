# Flatland revamp: algorithmic speed-ups and a Julia rewrite

This document plans how to make the Flatland simulator fast. It has two prongs, implemented as successive PRs into
the `revamp` branch of this repository:

1. **Python speed-ups** (§3): algorithmic and data-structure changes inside this repository, in the spirit of
   [#68](https://github.com/flatland-association/flatland-rl/pull/68).
2. **A Julia rewrite** (§4): a Julia core in `julia/Flatland/` in this repository, with a thin Python wrapper that
   keeps the `RailEnv` / PettingZoo / RLlib API.

Everything below refers to `main` at `51f5621c`. Claims are tagged:

- **[measured]**: profiled on this commit (§2, Appendix A).
- **[read]**: verified by reading the code at the cited `file:line`.
- **[estimate]**: an extrapolation, to be confirmed by the benchmark of the corresponding PR.

## 0. TL;DR

- **Observation building dominates everything else** [measured]. With 200 agents on a 100×100 grid:
  - the dynamics alone (`step()` without observations) take **3.2 ms per step**;
  - the default `GlobalObsForRailEnv` takes **110–140 ms**;
  - the canonical RL setup (`TreeObsForRailEnv(3)` + `ShortestPathPredictorForRailEnv(50)`) takes **210 ms**.

  With 500 agents on a 158×158 map, the dynamics fit easily: about 34 s for `reset()` plus a full 3122-step
  episode. The built-in observation builders do not. With the default builder (0.8–1.15 s per step, from single
  10-step windows), the same episode would take 40–60 minutes [estimate]. That is above the 30-minute per-scenario
  budget of ECML 2026, whose levels go up to 532 agents (on a different, 120×150 map).
- **The biggest Python wins are not in the dynamics.**
  - Rewrite the tree observation, predictor and global observation around precomputed tables.
  - Remove a conflict check in the tree observation that never fires. Deleting it alone cuts the tree observation's
    time by at least 44% (38% predictor included), with identical outputs. That check is also the only reader of the
    predictor's output inside the tree observation. Together, the check and the predictor are about half of a step
    with a tree observation [estimate].
  - Make the step pre/post-condition checks opt-in: they add 40–50% to a step without observations.
  - Compute the distance map once per episode instead of twice: the timetable generator computes one, and the env
    computes its own lazily at the last step (2.6 s at 500 agents).
- **The dynamics are about 16–18 µs per agent per step** in Python [measured]. Each item below is bounded by its
  measured share, so only their combination can give a few × [estimate]:
  - integer speed ticks instead of `Fraction`: 38× on the speed-counter arithmetic, but `fractions.py` is 15% of
    profiled step time [measured];
  - integer entry-point tables instead of the transition maps' `lru_cache`s;
  - skipping inactive agents.

  A Julia core with struct-of-arrays state should reach two orders of magnitude on the dynamics alone [estimate].
  End to end, it only pays off once the observation builders are ported too (Amdahl: the dynamics are 2–4% of a
  step with a built-in observation builder).
- **The Julia rewrite is a dynamics-parity port, not a fork.**
  - Python keeps scenario generation and stays the reference implementation.
  - Julia must reproduce Python's trajectories bit for bit, checked on golden traces in CI.
  - The plan is 16 milestones (M0a–M12), each one or two PRs of under 1000 lines.
- **The two prongs share one data model.** Integer entry points, integer ticks, multi-source distance maps and a
  segment graph are both the main Python speed-ups and the Julia core's design. Doing them in Python first speeds
  up users now and shrinks the gap the conformance tests must cover.

## 1. Why speed matters

**The literature names the simulator as the bottleneck.** Chen, Li, Harabor & Stuckey
([arXiv:2306.06455](https://arxiv.org/abs/2306.06455)) won the Flatland 3 challenge with a MAPF approach. They
write:

> "A significant hidden challenge of the competition is to overcome the slow execution speed of the competition
> environment."

> "As discussed in the introduction, this environment consumes the majority of the execution time."

They also report that in NeurIPS 2020 the winning software "only used 30% of the 8 hours for planning agents, the
rest was spent in executing the simulation environment". For Flatland 3, "the total planning time is only 5% of the
total time".

The Flatland 2020 competition report ([arXiv:2103.16511](https://arxiv.org/abs/2103.16511)) gave submissions 8 hours
in total, "10 minutes per environment for initial planning, then 10 seconds … per timestep", on 4 CPU cores. The
original Flatland paper ("Flatland-RL : Multi-Agent Reinforcement Learning on Trains",
[arXiv:2012.05893](https://arxiv.org/abs/2012.05893)) promised "fast experimentation" as
the reason to simplify the physical simulation in the first place.

**Current challenges are larger.** The
[ECML 2026 levels](https://flatland-association.github.io/flatland-book/challenges/ecml2026/levelconfig.html) go up
to 532 agents on a single 120×150 map with 28 stations. The [evaluation rules](https://flatland-association.github.io/flatland-book/challenges/ecml2026/eval.html)
allow "up to 30 minutes per scenario", 5 hours in total, on 4 CPU cores with 15 GB.

**RL training multiplies the cost.** At 4.6–4.8 steps/s (§2, 200 agents, tree observation), one million environment
steps take about 60 hours on one core. At the dynamics-only speed (316 steps/s), they take under an hour.

**Upstream has worked on speed for years, with mixed results.**
- [#6](https://github.com/flatland-association/flatland-rl/issues/6) found `MotionCheck.find_conflicts` at about 50%
  of `step()`. [#87](https://github.com/flatland-association/flatland-rl/pull/87) fixed it by replacing networkx with
  dicts. It is based on "Bochatay (2024)", cited in `flatland/envs/agent_chains.py:270`; no public copy of that report was
  found.
- [#68](https://github.com/flatland-association/flatland-rl/pull/68) moved A* to a heap.
- A September 2026 series of micro-optimisations (PRs
  [#520](https://github.com/flatland-association/flatland-rl/pull/520) and
  [#521](https://github.com/flatland-association/flatland-rl/pull/521), landed via
  [#499](https://github.com/flatland-association/flatland-rl/pull/499)) gained from about 1% to 50% on individual
  sub-functions (`step_reward` halved).
- Upstream CI's [profiling chart](https://github.com/flatland-association/flatland-rl/pull/550#issuecomment-6495645677)
  (Python 3.13, 200 agents, checks off, no observations) shows that `step` time swung 3× across releases: 1.02 s in
  v4.1.1, 3.07 s in v4.2.5, 1.26 s on `main`. These are cProfile cumulative times over a 200-step run, not comparable
  to §2's per-step wall-clock times. The Cython build of
  [#449](https://github.com/flatland-association/flatland-rl/pull/449) gives 1.25 s, i.e. nothing, because
  `step()` no longer drives the compiled state machine.
- [#206](https://github.com/flatland-association/flatland-rl/issues/206) ("Fraction", "speed counter", "rewards") is
  still open.

## 2. Baseline measurements

**Setup.**
- Commit `51f5621c`, CPython 3.13.15, pure-Python sources, one pinned core of a 4-core Linux VM.
  - The Cython build doesn't change `step()` (§1).
  - It may speed up `reset()`: the line generator calls `get_k_shortest_paths`, one of the compiled modules. So the
    reset numbers are upper bounds.
- Scenarios: `sparse_rail_generator(max_rails_between_cities=2, max_rail_pairs_in_city=2)`, speeds
  {1, ½, ⅓, ¼} uniform, `ParamMalfunctionGen(rate=1e-4, 15–50 steps)`, seed 1. Actions are seeded uniform-random,
  pre-generated, as in upstream's profiling script.
- A fresh env for every repetition.
- **The random policy gridlocks.** At M, none of the 200 agents reaches its target, 105 never depart, and at most 95
  are on the map at once. At L, 3 of 500 arrive and 94 never depart. This shapes station queues (MotionCheck),
  tree-observation density and reward bookkeeping; a trained policy's profile may differ.
- **Repetitions.** S and M/b/d: 5 or 3 repetitions of full episodes, with a coefficient of variation of the
  per-step mean of 3% (0.7% on a second pass). M/a and M/c: 3 repetitions of 50- and 30-step windows. L: one
  repetition, and only 10-step windows for L/a and L/c.
- Timings are wall-clock between timestamps on the bracketing calls. cProfile was used only for within-phase splits
  and call counts. The driver and analysis scripts are in Appendix A.

| scenario | grid | agents | cities | max episode steps | basis |
|---|---|---|---|---|---|
| S | 30×30 | 7 | 2 | 278 | Flatland 3 `Test_00` |
| M | 100×100 | 200 | 5 | 1656 | upstream CI profiling setup |
| L | 158×158 | 500 | 41 | 3122 | Flatland 3 `Test_14` geometry, about the ECML 2026 agent count (not its map) |

| config | observation builder | `check_step_pre_post_conditions` |
|---|---|---|
| a | constructor default (`GlobalObsForRailEnv`) | constructor default (`True` at this commit) |
| b | `DummyObservationBuilder` | `False` (upstream CI) |
| c | `TreeObsForRailEnv(3, ShortestPathPredictorForRailEnv(50))` | `False` |
| d | `DummyObservationBuilder` | `True` |

All configs follow the same trajectory: fingerprints of all agents' final state are equal across configs and
repetitions.

### 2.1 Step time

Per-step mean over steady-state steps; the first and last steps are reported separately in §2.3.

| scen/cfg | ms / step | steps / s | where the time goes |
|---|---|---|---|
| S/b | 0.18 | 5 700 | collect 34%, distribute 34%, info dict 11%, snapshot 6% |
| S/d | 0.24 | 4 100 | checks 29% |
| S/a | 0.61 | 1 600 | global obs 60%, checks 12% |
| S/c | 3.6 | 280 | tree obs 76%, predictor 19% |
| M/b | 3.2 | 316 | collect 38%, distribute 28%, MotionCheck 13%, info dict 10%, snapshot 5% |
| M/d | 4.7 | 211 | checks 27% |
| M/a | 109 → 142 | 9 → 7 | global obs 97% |
| M/c | 217 → 211 | 5 | tree obs 80–84%, predictor 15–18% |
| L/b | 8.9 | 112 | collect 40%, distribute 33%, info dict 10%, snapshot 7%, MotionCheck 5% |
| L/a | 790 → 1 150 | 1 | global obs 99% |
| L/c | 778 → 765 | 1 | tree obs 83–84%, predictor 15% |

- "x → y" is the early window (from step 1) versus the late window (around peak on-map occupancy: step 1240 for M,
  2795 for L).
- Other seeds for M/b: seed 2 gives 3.18 ms/step, seed 3 gives 3.22 ms/step.
- The dynamics cost about **16–18 µs per agent per step** (M/b and L/b), nearly independent of the scenario size.

### 2.2 Reset time (config b)

| scen | total | rail | line | timetable | timetable's distance map | other |
|---|---|---|---|---|---|---|
| S | 35 ms | 29 | 1 | 2 | 3 | 0.5 |
| M | 297 ms | 146 | 39 | 59 | 50 | 3 |
| L | 6 108 ms | 668 | 2 523 | 287 | 2 621 | 8 |

### 2.3 Hidden spikes

| | S | M | L | cause |
|---|---|---|---|---|
| first step | 8.1 ms (46× a normal step) | 9.8 ms | 14.8 ms | `ParamMalfunctionGen` draws `rand(1_000_000)` on its first call, even at rate 0 |
| last step | 3.1 ms | 85 ms (46 ms distance map) | **2 761 ms** (2.6 s distance map) | the env's own distance map is computed lazily, the first time a reward reads it at episode end; it is a second full computation after the timetable's; the rest is end-of-episode rewards |

### 2.4 Micro-benchmarks

Run in-process, best of several repetitions; `benchmarks/revamp/micro.py` (Appendix A).

| what | result |
|---|---|
| tree obs at M, step 1240 (95 agents on map): `get_many` with the dead conflict block (§3.1) vs with its `fast_delete` stubbed out | **182.5 → 113.5 ms (−38%) predictor included, 156 → 87 ms (−44%) without the predictor's 26 ms**. Lower bounds: the block's other work still ran. Observations identical (`repr` equal); 24 930 `fast_delete` list copies, 0 `fast_where` calls |
| global obs at M: the six `isfinite`/`isnan` asserts | 0.12 ms per agent, i.e. 24 ms of the 145 ms the builder needs per step |
| speed-counter arithmetic per agent-step (cell exit, acceleration, both distance formulas, the entry test) | `Fraction` with the existing `lru_cache`s: 6.8 µs; `Fraction` uncached: 6.7 µs; **integer ticks: 0.18 µs (38×)** |
| `Fraction + Fraction` vs `int + int` / `hash(Fraction)` | 479 ns vs 18 ns / 179 ns |
| transition lookup: `rail.get_transitions(ep)` (cache hit) vs dict vs list index | 353 ns vs 53 ns vs 16 ns |
| `RandomState.rand(1_000_000) < p` (first malfunction draw) | 6.5 ms |
| distance map at M: one `_compute` (warm) | 37 ms; **6 unique target sets for 200 agents**, stored as 200 copies (64 MB) |
| target dedupe with list membership vs dict | 0.24 vs 0.06 ms at n = 200; 3.0 vs 0.6 ms at n = 2000 (real but small) |
| `MotionCheck.find_conflicts`, m agents queued for one cell (synthetic; in the env, large m comes from off-map agents departing into one start cell, since an on-map cell has at most 4 predecessors) | 0.04 / 0.3 / 2.4 / 32 / 551 ms for m = 10 / 25 / 50 / 100 / 200: **grows as m^3.6** |

### 2.5 What cProfile adds

cProfile inflates call-heavy code, so these are relative shares of *profiled* time, used to split phases.

- **M/b dynamics** (1656 steps):
  - `fractions.py` 15% of self time;
  - `step_reward` 12%;
  - `derived_state()` 9%, with 994k calls, about 3 per agent per step;
  - `find_conflicts` 9%;
  - `Fraction.__hash__` 2.3M calls.
- **M/a global obs:** 1.6M `derived_state()` calls in 20 steps, i.e. 2·n² per step: 21% of profiled time early,
  39% late.
- **M/c and L/c tree obs:**
  - `_explore_branch` is 75% of profiled time;
  - the dead conflict block's helpers (`fast_delete` + `coordinate_to_position`) take 15%.
- **L/b reset** (cumulative times overlap):
  - the line generator's `check_path_exists` DFS is 45%;
  - `_reconstruct_shortest_path` is 44%, but 40 points of it are the nested lazy distance-map computation
    (`geometric_distance` → `get()` → `_compute`); building the paths themselves is about 4%. The timetable
    generator builds every agent's full shortest path only to take its `len()`, which equals the distance plus 1;
  - the distance map's `_compute` is 40% (the same computation as above);
  - A* (after #68) is 3%.

## 3. Prong 1: speed-ups inside the Python codebase

Each item gives:
- where (`file:line`);
- the current complexity and the proposed one;
- its measured share;
- its **behaviour impact**: *none* (bit-identical trajectories and observations), *obs* (changes observation
  values), *rng* (changes RNG consumption, hence seeded scenarios or malfunctions), or *api*;
- how to test it.

They are ordered by measured impact. Notation: n agents, H×W grid, V ≤ 4HW entry points, T steps, D tree depth,
P predictor depth.

### 3.1 Observation builders (the largest cost by far)

**Tree observation** (`flatland/envs/observations.py`, `TreeObsForRailEnv`). It costs 80–84% of a step at M, about
0.85 ms per agent per step [measured]. Each agent walks its tree cell by cell in Python. At every visited cell, it:
- parses the cell's 16-bit code as a string: `bin(...)`, `.count("1")` and `int(transition_bit, 2) == int('1000010000100001', 2)`
  (`observations.py:323-328`) [read];
- builds `OrderedSet`s of visited cells that only the renderer reads (`dev_obs_dict`) [read];
- runs a conflict check that **never fires**:
  - `coordinate_to_position(self.env.width, [position])` returns a one-element list
    (`flatland/core/grid/grid_utils.py:265-291`);
  - `predicted_pos[t]` is a list of ints;
  - so `int_position in fast_delete(self.predicted_pos[...], handle)` (`observations.py:331-368`) is always false.

  The three `elif` branches still copy and scan an n-element list each, which makes the check O(n) per explored
  cell and O(n²·cells) per step. The feature `dist_potential_conflict` is always `inf` [read; the A/B in §2.4
  confirms identical observations].

**The check is also the predictor's only consumer.** Inside `TreeObsForRailEnv`, the predictor's output is used only
by this check (`observations.py:70-86,331-368`) [read]. So the 15–18% of each step spent in the predictor feeds a
constant feature.

Fixes, in order:

| # | change | impact |
|---|---|---|
| T1 | Delete the dead conflict block: **at least −44% of the tree observation's own time at M** (−38% predictor included) [measured, §2.4]. | none |
| T2 | Replace the string parsing with `int(code).bit_count()` and an integer comparison (or per-cell tables built at `reset()`). | none |
| T3 | Merge visited cells into `dev_obs_dict` only when rendering is requested (`observations.py:246,250,477,487`). The per-segment `visited` set also detects loops (`:379-382`), so it stays, as a plain `set`. | none (`dev_obs_dict` is only for rendering) |
| T4 | Walk a **segment graph**: contract runs of cells without a decision into edges, with length, target and switch metadata precomputed at reset, and bucket agents by segment once per step. Per-agent cost drops from O(b^D·ℓ) to O(b^D), with ℓ the mean segment length. `DecisionPointGraph` (`flatland/envs/graph/graph_simplification.py:28-98`) already builds this contraction for graph envs. | none if feature semantics are kept |
| T5 | Restore `dist_potential_conflict` correctly with a `{(t, cell): [handles]}` index built once per step. Separate PR. | obs |
| T6 | Until T5 lands, call the predictor only when something reads its output. Besides the dead check, its readers are `obs_builder.predictions`, the renderer's `env.dev_pred_dict` and the persisted snapshots that store it (`persistence.py:427`). | api |

**Shortest-path predictor** (`flatland/envs/predictions.py:96-175`). It costs 15–18% of a step at M [measured].
Every step, it rebuilds every agent's path to depth P from the distance map:
- `_reconstruct_shortest_path` (`flatland/core/distance_map.py:149-183`) allocates an attrs `Waypoint` per cell;
- `shortest_path = shortest_path[1:]` inside the loop (`predictions.py:151,168`) makes each path O(P²);
- `if handle:` (`predictions.py:46,121`) treats handle 0 as "all agents" (latent: no caller in the repository passes a
  handle).

Fixes:
- Precompute a **next-hop table** per unique target set when the distance map is built. Reconstruction becomes P
  array lookups, with tie-breaking kept identical to `_reconstruct_shortest_path`.
- Reuse last step's path when the agent has only advanced along it.

Impact: none.

**Global observation** (`observations.py:552-604`, the **constructor default**: `rail_env.py:177,1603`). It costs
96–99% of a step at M and L [measured]. For each agent `get()`:
- allocates fresh H×W×2 and H×W×5 arrays;
- loops over **all** agents, calling `derived_state()` twice per pair. cProfile counts 1.6M `derived_state` calls in
  20 steps at M, i.e. n² × 2;
- runs six full-array `isfinite`/`isnan` asserts (`observations.py:598-603`), two of which rescan the constant H×W×16
  rail layer. They take 24 ms of the builder's 145 ms per step at M [measured].

`reset()` fills that rail layer cell by cell via `bin()` strings (`observations.py:552-559`). The default argument
`GlobalObsForRailEnv()` is evaluated once, so **every `RailEnv` built without an explicit builder shares one
instance** [read].

Fixes:
- Default to `None` and build a new instance per env: a bug fix of its own.
- Build the shared layers once per step, in O(n + HW), then give each agent a copy with its own cells set. The total
  stays O(n·HW) because of the per-agent copies. On a cell shared by several agents (level-free crossings), the
  second channel must keep the other occupant's direction, and the other channels are last-writer-wins in handle
  order.
- Check the rail layer once at reset.
- Vectorise the bit unpacking with `(grid[..., None] >> np.arange(15, -1, -1)) & 1`.

Impact: none for the speed-up.

**Gym flattening** (`flatland/ml/observations/flatten_tree_observation_for_rail_env.py:64-124`). It concatenates
small numpy arrays recursively per node, and normalises with Python while-loops. Fix: fill one preallocated array at
precomputed offsets. Impact: none.

### 3.2 Defaults and per-step work that does nothing

| # | where | what | fix | impact |
|---|---|---|---|---|
| D1 | `rail_env.py:185,1612` | `check_step_pre_post_conditions=True` by default: +39% per step at S, +50% at M [measured] | Default to `None`, deferring to an environment variable that the repository-root `conftest.py` sets for the test suite. | api (default change) |
| D2 | `rail_env.py:1687-1696`, called at `:1707` | `_update_agent_positions_map` is an O(n) loop every step; `agent_positions` is read nowhere in `flatland/` [read] | Deprecate, compute lazily on access. | api (deprecation) |
| D3 | `rail_env.py:590,609,699-707` | per agent and step: `state = agent.state` (a `derived_state()` call) used only in a warning; `frozenset(agent.targets)` rebuilt although targets are fixed per episode; `any(self.rail.get_transitions(...))` computed only to decide whether to warn | Hoist or drop. | none |
| D4 | `rail_env.py:425-448` | `get_info_dict` is 10% of the dynamics [measured]; it calls `derived_state()` twice per agent | Compute the state once per agent. | none |

### 3.3 Distance maps and `reset()`

- **Computed twice per episode** [measured, read]:
  - the timetable generator builds its own `DistanceMap` with "fake agents" and computes it
    (`timetable_generators.py:60-79`). The branch is always taken, since sparse lines have at least two waypoint
    groups;
  - the env's own map is computed lazily on first read, which is the last step for configs b/d (2.6 s at L).

  Fix: a cache keyed by the target set, shared by the timetable generator and the env. The timetable's fake agents
  target `{(waypoints[i+1][0].position, d) for all d}` (`timetable_generators.py:72`). For lines of two stops, that
  equals the env agents' targets; for longer lines, it is a different set per leg.
- **Computed twice after every load and clone.** `RailEnvPersister` assigns the stored map and then calls `_compute`
  anyway (`persistence.py:312-314`), and the next `get()` computes it again [read].
- **Data structures** (`core/distance_map.py:66-88`, `core/distance_map_walker.py`, `envs/grid/distance_map.py:30-41`)
  [read]:

  | today | proposal |
  |---|---|
  | K ≤ 4 sequential backward BFS walks per target, with tuple keys in a `defaultdict` | **one multi-source BFS** per unique target set, over a precomputed int predecessor table |
  | an `(n, H, W, 4)` float64 array, with a full copy per agent sharing a target: 64 MB at M for **6 unique target sets** [measured], 400 MB at L | `(U, H, W, 4)` int32 plus an agent-to-row index, with U the number of unique target sets (0.96 MB at M) and a sentinel for unreachable entry points |
  | targets deduplicated with list membership and `.index`, O(n²); measured small: 3 ms at n = 2000 | a dict keyed by `frozenset(targets)` |

  The walker's comment says a *shared visited set across sequential walks* broke looped layouts. A true
  multi-source BFS does not have that problem, but needs a looped-layout regression test.

  Impact: none for the walk, the dedupe and computing once. The compact storage is **api and file format**:
  `distance_map.get()` returns the `(n, H, W, 4)` float64 array with `inf` today, which `observations.py:278`,
  `rail_env_shortest_paths.py:220,224` and the persisted state (`persistence.py:98,466`) read. Keep a lazily
  materialised full-array getter for them.

  Open PR [#542](https://github.com/flatland-association/flatland-rl/pull/542) adds fractional edge lengths and
  turns the BFS into a label-correcting walk over an object array of `int`/`Fraction`. With integer ticks (§3.4),
  that array stays integer.
- **Line generator** (2.5 s at L, 41% of reset [measured]):
  - `check_path_exists` runs a DFS per agent and orientation (`line_generators.py:85-88`);
  - `get_k_shortest_paths(k=1)` serves as a reachability test, up to 16 calls per leg (`line_generators.py:194-201`).

  Fix: one cached BFS per source. [#462](https://github.com/flatland-association/flatland-rl/issues/462) proposes using
  the distance map as an A* heuristic for k = 1. Impact: none if the same orientation is chosen.
- **Rail generator** [read]:
  - A* evaluates the expensive `validate_new_transition` before testing `respect_transition_validity`, which is
    `False` here, so the result is discarded (`core/grid/grid4_astar.py:116-118`). Swap the operands. Impact: none.
  - `_fix_transitions` iterates `list(city_cells) + inter_city_lines` with duplicates and builds a fresh
    `RailEnvTransitions()` per fixed cell (`rail_generators.py:874,899`). Impact: deduplicating changes the generated
    rail if it changes the iteration order. `list(city_cells)` iterates a set, so the order is CPython-specific.
- **`get_k_shortest_paths`** (`rail_env_shortest_paths.py:60,100-104`) [read]:
  - `min(heap)` scans all bucket keys on every pop;
  - `v in pu` / `pu + (v,)` copy whole path tuples per edge.

  Fix: parent pointers, and a monotone cursor over costs. Impact: none if tie-breaking is kept.

### 3.4 Numbers: integer speed ticks instead of `Fraction`

Speeds and distances are `fractions.Fraction` (`step_utils/speed_counter.py`). Fraction arithmetic is pure Python,
and the caches built to hide it hash Fractions on every lookup:
- `_cap_speed` (`:55`), `cached_cell_exit` (`:64`), the four `speed_after_*` / `distance_*` formulas;
- `_candidate_distance`, with a 10-keyword cache key (`rail_env.py:1323`).

In the M/b profile, `Fraction.__hash__` is called 2.3M times in 1656 steps, and `fractions.py` takes 15% of
profiled self time [measured]. The per-agent speed-counter pattern costs:
- 6.8 µs with `Fraction`s and the existing caches;
- 6.7 µs with `Fraction`s and no caches, so **the caches buy nothing**: hashing the key costs as much as the
  arithmetic;
- 0.18 µs with integer ticks [measured, §2.4].

Proposal: store speed and distance as integers in units of 1/Dᵢ, with a **per-agent** denominator
Dᵢ = lcm(den(max_speedᵢ), den(acceleration_delta), den(braking_delta)), plus edge lengths once #542 lands.
- A per-agent Dᵢ is needed because a global LCM overflows Int64 quickly: lcm(1..50) > 2⁶³.
- Every quantity of agent i stays in (1/Dᵢ)ℤ ∩ [0, 1], so the arithmetic is exact. Comparisons become int
  comparisons, and the caches can go.
- `Fraction` properties remain at the API edge (`info["speed"]`, `SpeedCounter.speed`).

Caveats:
- `RailEnv` defaults the deltas to floats 1.0 / −1.0 (`rail_env.py:1610-1611`), unlike `AbstractRailEnv`. Convert
  floats exactly at the constructor rather than rejecting them, which would be an api change.
- `_pseudo_fractional` snaps floats to "nice" fractions; keep it only at the API edge.

Impact: none, checked by bit-identical trajectories on the regression corpus.

### 3.5 Topology: integer entry points and precomputed tables

Today, entry points are nested tuples `((r, c), d)`. The transition maps answer queries through 29 `lru_cache`
decorators (`transition_map.py` 10, `grid4.py` 8, the grid and graph transition maps 5 and 6; 56 in the whole
package). Instance-method caches are keyed by `self` through a Python-level `__hash__`
(`core/transition_map.py:207`), so every lookup calls back into the interpreter. The caches hold strong references
to old rails, up to `maxsize` = 1–4M entries [read]. `GridTransitionMap.__hash__` is called 600k times in 15
tree-observation steps at M [measured].

Proposal: when a rail is finalised, number entry points `ep = 4·(r·W + c) + d` and build flat tables:
- `succ[action, ep] → ep | −1`, covering all 5 actions and storing invalid moves explicitly.
  `apply_action_independent` is a pure function of the cell code, the entry point and the action, so it fits a
  table: the single-switch fallback to forward and symmetric-switch invalidity included
  (`flatland/envs/rail_grid_transition_map.py:46-99,143-149`).
- `resource[ep]`: the cell, or (cell, axis) for level-free crossings (`flatland/core/grid/grid_resource_map.py:13-20`).
- `valid[ep]`, and predecessors in CSR form for the BFS.

Tuples remain derived properties at the API boundary, in the spirit of
[#281](https://github.com/flatland-association/flatland-rl/issues/281) (de-couple public and internal state). The
same tables describe a `GraphTransitionMap`, so `step()` no longer depends on the back end. Graph tables
must be built in Python, because `apply_action_independent` takes the first successor in networkx insertion order
(`envs/graph/rail_graph_transition_map.py:144-151`). Impact: none.

### 3.6 Skip inactive agents

Every step runs the whole collect, MotionCheck, distribute and reward pipeline for every agent. "Inactive" means
off the map and not departing this step, or removed after `DONE`. Agents idling on the map stay active, since they
hold their resource in MotionCheck. `RailEnvStateMachineWrapper` reconstructs signals for every agent, so it must
get the inactive agents' trivial signals too. Keep index sets of active, waiting (sorted by `earliest_departure`)
and done agents, and loop over active ones only. Precondition: check, per reward class, that per-step rewards are zero for inactive agents. In
`BaseDefaultRewards.step_reward` an agent off the map with `old_entry_point is None` records nothing
(`rewards.py:189-240`). Impact: none.

### 3.7 MotionCheck

`_fix_conflicts` pops the head of its conflict list with `target_conflicts = target_conflicts[1:]`
(`agent_chains.py:368`), and `_stop_and_update_target_conflicts` calls `list.remove` (`:379`). That is O(p²) in the
number p of conflicting pairs, and p grows quadratically with the number m of agents queued for one cell [read]. In
the env, large m comes from off-map agents departing into one start cell (an on-map cell has at most 4
predecessors).
§2.4 measures m^3.6: 551 ms with 200 agents queued [measured]. In the random-policy runs, MotionCheck is 13% of
the dynamics at M and 5% at L [measured].

Fix: an index cursor (or `deque`) over the queue, which makes it O(n + p). O(n + m) would need a per-resource
reformulation, which the next paragraph rules out.

**The queue order is part of the semantics.** For example, with three agents a < b < c asking for one free cell,
the current order lets a move. Processing the pairs in the order (b, c), (a, c), (a, b) stops all three, because a
stopped agent then blocks the others. So any rewrite must keep the exact order: pairs built in handle order, each
resource's agents in insertion order, FIFO with appends. It must also be accepted by differential fuzzing against
the current implementation. Iteration over the `deadlocked` set (`:325`) follows CPython's set order; `sorted()`
there is cheap insurance. Impact: none.

### 3.8 Malfunctions

`ParamMalfunctionGen` caches `rand(1_000_000)` on its first call, even for `NoMalfunctionGen`
(`malfunction_generators.py:63-79,112-114`). The cache survives `reset()` and is rebuilt on every clone and load.
After that, it makes one Python call chain per agent per step.

Fixes:
- a rate-0 fast path;
- per-step draws as one slice plus `np.flatnonzero`;
- a stateless generator ([#364](https://github.com/flatland-association/flatland-rl/issues/364)).

Impact: **rng**. Skipping the 1M draws changes the RNG stream for anything drawn afterwards without reseeding, so it
needs a compatibility flag or regenerated golden data.

### 3.9 Peripheral costs

| where | what | fix | impact |
|---|---|---|---|
| `trajectories/policy_runner.py:109,135` | `snapshot_interval=1`: a full env pickle every step, including the static grid | a default of about 50 (resume already replays from the nearest snapshot); write the grid once | api (default) |
| `trajectories/trajectory_observation_callbacks.py:32,34` | recomputes `env._get_observations()` instead of reusing the step's observations | reuse `env.obs_dict` | none for deterministic builders; a fix for stochastic ones (`observations_perturbed.py` draws from `np_random`), which today record observations that differ from the policy's |
| `trajectories/trajectories.py:181-193` | `info` is stored as Python `repr` (`Fraction(1, 2)`, `<TrainState.MOVING: 3>`) and parsed back with regexes and `ast.literal_eval` per row | typed columns, or Parquet | file format |
| `evaluators/client.py:167,186` | two synchronous Redis round trips per step | pipeline them | none |
| `envs/rail_env_shortest_paths.py:5` | `import matplotlib.pyplot` at module level, for a debug plot: 0.29 s of the 0.48 s it takes to import `rail_env` [measured] | lazy import | none |

### 3.10 Bugs and parity hazards found along the way

Found by reading [read]. Each needs a failing test before its fix.

1. **TreeObs conflict check never fires** (§3.1). `dist_potential_conflict` is always `inf`.
2. **The predictors treat handle 0 as "all agents"**: `if handle:` at `predictions.py:46,121`. Latent: no caller in
   the repository passes a handle.
3. **`coordinate_to_position` collides when H > W.** It encodes `col·width + row` (`grid_utils.py:290`). No current
   effect, since its only callers feed the dead conflict check; it matters for T5.
4. **The recorded deadlock flag is always 0.** `RecordStepsEffectsGenerator` tests
   `position in env.resource_check.deadlocked` (`record_steps_effects_generator.py:36`), but `deadlocked` holds
   agent handles.
5. **Every `RailEnv` without an explicit builder shares one `GlobalObsForRailEnv`** (`rail_env.py:177,1603`).
6. **Reward bookkeeping leaks across episodes.** `Rewards` has no `reset()`, and `RailEnv.reset()` keeps the same
   object. `arrivals` / `departures` / `stopped_waypoints` (`rewards.py:183-187`) leak, and the reward state is not
   persisted either.
7. **`ConditionalMalfunctionEffectsGenerator._num_malfunctions` is never reset** (`malfunction_effects_generators.py:103`).
   `max_num_malfunctions` is therefore a lifetime cap.
8. **`DummyPredictorForRailEnv` mutates the agents it predicts for.** It overwrites `agent.current_entry_point` on every
   call (`predictions.py:73`), and restores it only at the end. The `raise` at `predictions.py:79` leaves it changed,
   and a read-only state view (§4.5) cannot support it.
9. **PettingZoo `render()` cannot work.** It calls `self._wrap.render()`
   (`ml/pettingzoo/pettingzoo_parallel_rail_env.py:95`), but `RailEnv` has no `render` method.
10. **A tie depends on set order** (parity hazard): `np.argmax` over `list(stopped_wps)`, a set
    (`rewards.py:323-336`).
11. **MotionCheck needs exclusive occupancy** (contract hazard). It assumes each resource is held by at most one
    agent; its termination argument ("in every round, one agent is stopped", `agent_chains.py:320`) relies on it.
    With two agents holding the same cell, `find_conflicts` loops forever. With three or more, its conflict queue also
    grows until memory runs out: a fuzz test hit a `MemoryError` [measured]. The cause is that re-stopping an
    already-stopped agent re-adds pairs for every other agent listed under its cell (`:375`). The env guarantees
    exclusivity, but legacy snapshots from the old same-cell bug (`tests/test_known_flatland_bugs.py`) or states edited
    by hand do not. Then `step()` hangs instead of failing. Assert exclusivity on `add_agent` with a
    resource → holder dict.
12. **`reset(random_seed=s)` on a used env does not reproduce a fresh env's malfunctions.** `ParamMalfunctionGen`'s
    `_cached_rand` and `_rand_idx` survive `reset()` (`malfunction_generators.py:57-79`).
13. **`PolicyRunner.create_from_policy` shares the caller's state with a second env.** It hands `env.obs_builder`,
    `env.rewards` and `env.effects_generator` to an env loaded from a snapshot (`policy_runner.py:105-129`), which
    rebinds the builder to itself and accumulates its rewards into the caller's per-episode reward state.

### 3.11 PR sequence

Successive PRs against `revamp`. Each is under 1000 added lines, has non-regression tests (bit-identical
trajectories, observations or distance maps where the impact is "none"), and gets an entry in the profiling table:

| PR | content | impact |
|---|---|---|
| P1 | D1: checks opt-in; enabled in the test suite | api (default change) |
| P2 | Global obs: shared layers, vectorised reset, asserts on the shared layers (the default-instance fix, bug 5, is its own PR) | none |
| P3 | Tree obs T1–T3 | none |
| P3b | Tree obs T6: call the predictor only when its output is read | api |
| P4 | Predictor next-hop table and incremental paths (with bug 2) | none |
| P5 | Distance map: dict dedupe, multi-source BFS, compute once per load/clone; then the shared cache and the compact rows behind a compatible getter | none, then api |
| P6 | MotionCheck: process the queue in place, O(n + p), with a differential fuzz test | none |
| P7 | D2–D4, inactive-agent index sets (§3.6) | none / api |
| P8 | Integer speed ticks, in 2–3 PRs | none |
| P9 | Integer entry-point tables (grid, then graph), in 2–3 PRs | none |
| P10 | Tree obs T4 (segment graph) | none |
| P11 | Reset: line-generator BFS, A* operand swap, `get_k_shortest_paths` | none / rng |
| P12 | Malfunction fast path behind a flag | rng |
| P13 | Trajectories: snapshot interval, observation reuse, typed info | api / format |
| P14 | Bug fixes 3, 4 and 6–13, one PR each (6–8 are also milestone M0a) | obs / api |
| P15 | CI profiling covers the defaults, observation builders, load/clone and L-size scenarios; today only config b at M is profiled | — |

The "none" PRs are also checked against the existing regression corpus: `benchmarks/benchmark_episodes.py`, 113
episodes replayed step by step.

## 4. Prong 2: a Julia rewrite

### 4.1 Goals and non-goals

Goals:
- **Bit-identical dynamics.** Given the same initial state, actions and effects, Julia produces the same positions,
  speeds, distances, states, rewards and dones as Python at every step.
- **End-to-end seeded parity** while generation stays in Python. A small port of numpy's legacy `RandomState`
  (MT19937, gym-style seeding as in `utils/seeding.py`, `rand`, masked `randint`) is enough to reproduce
  `ParamMalfunctionGen`.
- **No allocations in `step!`** beyond preallocated buffers sized at reset (MotionCheck's conflict pairs and the
  reward bookkeeping grow with conflicts and history, so their capacity is bounded and checked), type-stable code,
  and threaded batches of envs.
- **Python users keep their code.** A wrapper exposes the Julia env behind the `RailEnv` protocol that the ml
  adapters and observation builders read (§4.5).

Non-goals, at least initially:
- Julia-native rail/line/timetable generators. Bit-parity is unrealistic, because generation depends on numpy
  legacy algorithms, CPython set iteration order (`rail_generators.py:874`) and heap tie-breaking. A later port would
  be statistically equivalent only.
- The AIcrowd evaluator service, the InteractiveAI client, the Jupyter editor, rendering, and reading pickles in
  Julia.

### 4.2 Prerequisites on the Python side

These help both prongs:

1. **A language-neutral state format** (JSON, versioned; continues
   [#129](https://github.com/flatland-association/flatland-rl/issues/129)):
   - the grid as a uint16 matrix, and level-free positions;
   - agents: initial / current / next / target entry points, waypoints with alternatives and time windows,
     `max_speed`, speed and distance;
   - malfunction counters, elapsed / max steps;
   - **RNG state**, **effects-generator state** and **reward accumulator state**;
   - the malfunction probability `p = 1 − exp(−rate)` itself (`malfunction_generators.py:37`), since Julia's `exp`
     may differ from numpy's by one ulp;
   - rationals stored as `[num, den]`.

   Round-trip pickle ↔ JSON on every committed fixture.
2. **A golden-trace exporter**, `flatland-trace/1`: `{semantics_version, scenario, steps: [{inject, actions, post}]}`.
   - It records Replay tables by instrumenting `run_replay_config` (`tests/test_utils.py`): those tables are Python
     expressions with imperative setup, so the trace stores the post-injection state, not a recipe.
   - It also records random-policy episodes and the realised malfunction schedule.
   - The traces are generated from `HEAD` in CI rather than taken from the archived benchmark episodes. The step
     semantics still move fast: 157 commits touched the step code in six months, and
     `tests/test_flatland_regression_episodes.py` is skipped because a timing fix made its episodes diverge.
3. **A `SEMANTICS_VERSION` constant** in both implementations, bumped by any PR that changes trajectories.
4. **A `RailEnvView` protocol** (`typing.Protocol`). It lists the members the ml adapters, built-in observation
   builders, predictors and renderer read. `RailEnv` already satisfies it; the Julia-backed env will too.

### 4.3 Architecture

The intuition: once a rail is built, the network is static. Compile it into flat integer tables, keep the agents in
parallel arrays, and the step becomes table lookups and integer arithmetic.

```julia
struct EntryPointNetwork{L}         # grid and graph alike, built once per rail
    next::Matrix{Int32}             # (5 actions, nEP) → next entry point, 0 = none
    resource::Vector{Int32}         # entry point → resource id (cell, or cell+axis for level-free crossings)
    pred_ptr::Vector{Int32}         # CSR predecessors (distance maps)
    pred_idx::Vector{Int32}
    labels::Vector{L}               # (r, c, d) or graph node string, for I/O only
end

struct Agents                       # struct of arrays, one entry per agent
    initial_ep::Vector{Int32}
    ep::Vector{Int32}               # current entry point (0 = off map)
    next_ep::Vector{Int32}
    old_ep::Vector{Int32}
    target_ep::Vector{Int32}        # entry point where it arrived (0 = not done)
    target_set::Vector{Int32}       # row in the distance map, and index of its set of target entry points
    speed::Vector{Int32}            # ticks of 1/den[i]
    distance::Vector{Int32}
    max_speed::Vector{Int32}
    den::Vector{Int32}
    malfunction::Vector{Int32}      # remaining steps
    earliest_departure::Vector{Int32}
    latest_arrival::Vector{Int32}
    arrival_time::Vector{Int32}
    waypoints::WaypointTable        # stops with their alternatives and time windows (CSR)
end

mutable struct RailEnv{Checks,L,R<:AbstractRewards,E<:AbstractEffects,O<:AbstractObservation,RNG<:AbstractRNG}
    net::EntryPointNetwork{L}
    agents::Agents
    rewards::R
    effects::E
    obs::O
    rng::RNG
    buffers::StepBuffers            # preallocated: candidates, MotionCheck queues, reward vectors
    t::Int
end

step!(env, actions::AbstractVector{<:Integer})  # mutates env, returns nothing
```

(`EntryPointNetwork` without the predecessor tables and labels exists since milestone M2.)

- **`step!` mirrors the Python control flow.** It follows effects (step start) → collect → MotionCheck →
  distribute → rewards → effects (step end), branch for branch as specified in `design_by_contract.md`. The `_check_*`
  invariant net is compiled in or out through the `Checks` type parameter.
- **MotionCheck** uses preallocated arrays: an occupant per resource, a FIFO queue in the Python order (§3.7), and
  an O(n + p) resolution in the number p of conflicting pairs.
- **Distance maps** are multi-source BFS over the CSR predecessors, stored as a plain `Array{Int32, 4}` of size
  `(4, W, H, U)`, with `typemax(Int32)` as the unreachable sentinel. Never permute it in Julia (a
  `PermutedDimsArray` gets copied); numpy's `.T` of it is a zero-copy view in Python's `(U, H, W, 4)` layout. Note that
  dtype and `inf` differ from Python's float64 array, so the facade converts lazily.
- **Graphs.jl** is used for the decision-point graph and at the I/O edge of graph envs, never in `step!`; the distance
  maps use the CSR tables directly.
- **Observations** fill preallocated buffers (`observe!(buf, obs, env)`). The tree observation walks the segment
  graph (§3.1 T4).
- **Randomness** comes from an explicit `rng` field: `NumpyLegacyRNG` for parity, `Xoshiro` otherwise. There is no
  global RNG.
- **Batches.** A concretely typed `Vector{E}` of envs (`E` the concrete `RailEnv{...}` type) stepped with
  OhMyThreads.jl `tforeach`. Python calls it without the GIL (`_jl_call_nogil`), which only lets other Python threads
  run.

### 4.4 Dependencies: Python → Julia

The versions are the latest in the General registry as of 2026-10-30.

| Python | used for | Julia |
|---|---|---|
| numpy arrays | grid, distance maps, observations | `Base.Array`; no extra dependency |
| numpy `RandomState` | generators, malfunctions | a small `NumpyLegacyRNG` (MT19937) for parity; `Random.Xoshiro`; StableRNGs.jl 1.1 in tests |
| `fractions.Fraction` | speed, distance | `Int32` ticks; `Rational{Int}` only at the API edge |
| attrs, NamedTuple, dataclasses, fastenum | data classes, enums | `struct`; EnumX.jl 1.0 |
| `functools.lru_cache` | memoisation | precomputed tables (no memoisation) |
| `heapq`, `deque`, `OrderedSet` | A*, BFS | DataStructures.jl 0.19 |
| networkx | `GraphRailEnv`, decision-point graph | Graphs.jl 1.15 (+ MetaGraphsNext.jl 0.8 at the I/O edge only) |
| pickle / msgpack | env snapshots | JSON.jl 1.13 (state schema, §4.2); MsgPack.jl 1.2 if needed |
| pandas TSV | trajectories | CSV.jl 1.2 + DataFrames.jl 1.8; Arrow.jl 2.8 for a typed format |
| importlib_resources / `env_data` | packaged scenarios | Pkg `Artifacts.toml` / LazyArtifacts |
| matplotlib, pyglet, Pillow, svgutils | rendering | a Makie.jl 0.24 / CairoMakie 0.15 package extension, reusing `flatland/png`/`svg` tiles (deferred) |
| gymnasium, pettingzoo, ray | RL adapters | the Python wrapper (§4.5); CommonRLInterface.jl 0.3 as an optional extension |
| redis, crowdai, boto3 | evaluator | stays in Python |
| click | CLIs | deferred; `@main` |
| Cython | acceleration | not needed |
| pytest, pytest-xdist | tests | Test (`@testset` files included by `runtests.jl`, as in DifferentiationInterface.jl) |
| flake8, deptry | lint | Runic.jl 1.13 (as in DifferentiationInterface.jl), Aqua.jl 0.8, ExplicitImports.jl 1.15, JET.jl 0.9–0.13 (0.13 needs Julia ≥ 1.12; the older versions cover the LTS) |
| cProfile, line_profiler | profiling | Profile stdlib, Chairmarks.jl 1.3, AirspeedVelocity.jl 0.6, AllocCheck.jl 0.2 |
| jupyter-book (flatland-book) | docs | Documenter.jl 1.19 + Literate.jl 2.21 + DocumenterCitations.jl 1.5; a page in the flatland-book |
| juliacall (new) | Python wrapper | PythonCall.jl 0.9.37, test-only (differential tests): juliacall pins its own exact PythonCall version, so Flatland.jl must not depend on it. With `JULIA_CONDAPKG_BACKEND=Null` and `JULIA_PYTHONCALL_EXE`, CI reuses the Python venv instead of a Conda one |
| — | latency | PrecompileTools.jl 1.2–1.3 (1.3 needs Julia ≥ 1.12); optional PackageCompiler sysimage |

### 4.5 Python interop

**Layout.**
- `julia/Flatland/` holds the Julia package.
- `flatland/julia/` holds the wrapper code. It is behind a `[julia]` extra that adds `juliacall`.
- `flatland/juliapkg.json` declares the Julia dependency. It must sit at `flatland/`, not `flatland/julia/`.
  [pyjuliapkg](https://github.com/JuliaPy/pyjuliapkg) only scans the top of each `sys.path` entry and one level
  down, so a deeper file works in editable installs and silently fails in regular ones.
  [PySR](https://github.com/astroautomata/PySR/blob/master/pysr/juliapkg.json) uses the same layout.
- Before registration, `juliapkg.json` points at the repository with `url` + `subdir` + `rev`. In development and
  CI, there are two options:
  - set `PYTHON_JULIACALL_EXE` and `PYTHON_JULIACALL_PROJECT` together (juliacall refuses one without the other), with
    a project that `dev`s the checkout and pins PythonCall to juliacall's exact version;
  - or set `PYTHON_JULIAPKG_PROJECT` with `PYTHON_JULIAPKG_OFFLINE=yes`, without which juliapkg replaces a `dev`
    entry by URL tracking.

**Data path.**
- `reset()` still runs Python generation, exports the state (§4.2) and loads it into Julia.
- `step(action_dict)` converts the dict to an `Int8` vector and calls `step!`.
- Observations and the distance map come back as zero-copy views. juliacall's `ArrayValue` gives a numpy view for
  strided arrays of supported element types
  ([docs](https://juliapy.github.io/PythonCall.jl/stable/juliacall-reference/)).
- An array-returning `step_arrays` next to the dict-based `step`. Building Python dicts of infos, rewards and dones
  costs about as much as `get_info_dict` alone, 0.3 ms per step at M [measured], which would cap the end-to-end gain
  of the dict API at a few × over today's 3.2 ms [estimate].
- A batched `step_many` amortises the call overhead.

**The `RailEnvView` facade.**
- Observation builders keep `self.env` from `reset(env)`, so the facade keeps a stable identity and is refreshed in
  place once per step.
- Agent views index the shared arrays and build `Fraction`s lazily. They return plain Python `int`s, not the
  `np.int32` that numpy views over Julia arrays yield (`AGENTS.md`'s dtype-taint rule for entry points).
- `targets` is the same set as in Python. Builders only read `next(iter(agent.targets))[0]`, a position shared by all
  alternatives of the target, so the iteration order does not matter.
- The only mutation API is `set_malfunction(handle, n, phase)`, which is all that the effects generators of the tests
  need. Tests that set up states by hand (the `Replay` tables, `tests/test_utils.py:105-134`) run their setup on a
  Python env, export the state (§4.2) and load it into Julia. `DummyPredictorForRailEnv`, which mutates agents (bug
  8), is unsupported until fixed.

**Startup latency.**
- A PrecompileTools workload.
- Docker images bake the depot and set `PYTHON_JULIAPKG_OFFLINE=yes`, with an optional sysimage.
- Fork crashes. A child forked from a process running Julia segfaults (PythonCall issue
  [#628](https://github.com/JuliaPy/PythonCall.jl/issues/628)), and Python ≤ 3.13 on Linux forks by default. Ray and
  multiprocessing workers must use `spawn`, and the wrapper imports juliacall lazily.
- juliacall starts Julia with one thread by default; batched stepping needs `PYTHON_JULIACALL_THREADS` (and
  `PYTHON_JULIACALL_HANDLE_SIGNALS=yes`).

### 4.6 Testing

| layer | what | tools |
|---|---|---|
| unit | port the transition, speed, MotionCheck and reward unit tests (`test_flatland_core_transitions.py`, `test_speed_counter.py`, `test_agent_chains.py`, `test_rewards.py`) | `@testset` |
| tables | `EntryPointNetwork` against Python, exhaustively: every cell type × 4 headings × 5 actions, plus off-grid neighbours | golden JSON |
| conformance | every `flatland-trace/1` trace replays bit-identically, comparing rationals exactly (stricter than the Python tests' `np.allclose`) | golden traces from CI |
| differential | fuzz MotionCheck (about 10⁵ random configurations including swaps, chains and level-free crossings), then whole episodes, against the live Python implementation | PythonCall.jl + CondaPkg.jl, nightly |
| properties | invariants of the Python check net: at most one agent per resource except level-free crossings, speed ≤ max speed, distance in [0, 1], done agents off map | Supposition.jl |
| quality | stale/missing deps, method ambiguities, type stability of `step!`, explicit imports, zero allocations | Aqua.jl, JET.jl (`test_package`, `@test_opt`), ExplicitImports.jl, AllocCheck.jl / `@allocated` |
| Python wrapper | `tests/ml` smoke tests, and the Replay tests with their setup run in Python and exported (§4.5), parametrised over the backend | pytest `-m julia` |
| performance | `benchmark/benchmarks.jl` SUITE over S/M/L | AirspeedVelocity.jl |

### 4.7 CI

A path-filtered workflow `julia.yml` triggers on `julia/**` and on the exporters of the conformance data
(`tests/conformance/**`). A cheap `conformance.yml` re-runs those exporters whenever `flatland/**`, the exporters or
the data change, and fails if the committed data is stale.

- **Test matrix:** Julia `lts`, `1` and `pre` on ubuntu; macOS and Windows on `1`.
  - Steps: `julia-actions/setup-julia@v3`, `julia-actions/cache@v3`, `julia-buildpkg` and `julia-runtest` with
    `project: julia/Flatland`, `julia-processcoverage` with `directories: julia/Flatland/src`, Codecov.
- **Lower compat bounds:** `julia-actions/julia-downgrade-compat@v2` with `projects: julia/Flatland,julia/Flatland/test`
  and `mode: forcedeps`, then the tests run from that resolved environment, since `Pkg.test` re-resolves split test
  environments before Julia 1.12.
- **Formatting check** (Runic).
- **Docs:** built with `julia/Flatland/docs` as project, and `deploydocs(dirname = "julia", tag_prefix = "Flatland-")`
  so that the docs land in a subfolder of `gh-pages` and TagBot's subdirectory tags deploy versioned docs.
- **Benchmarks:** the AirspeedVelocity action has no subdirectory input, and the `benchpkg` CLI's `--path` only works
  with `--rev dirty` (named revisions go through `Pkg.add`, which fails for a subdirectory). So check out the base and
  the head in two worktrees and run `benchpkg --rev dirty --path <worktree>/julia/Flatland --script ...` in each, with
  separate output directories.
- **Conformance:** a Python job exports golden traces from `HEAD` as an artifact, and a Julia job replays them.
- **Wrapper:** install `.[julia]` and run `pytest -m julia`.
- **Bots:**
  - CompatHelper with `subdirs=["julia/Flatland", "julia/Flatland/docs"]`.
  - TagBot with `subdir: julia/Flatland`.
  - Registration via `@JuliaRegistrator register subdir=julia/Flatland`
    ([Registrator](https://github.com/JuliaRegistries/Registrator.jl)). The name `Flatland` is free in General.
- **Version tags:** TagBot names subdirectory tags `Flatland-v0.1.0`. setuptools-scm's default `tag_regex` accepts a
  `Name-` prefix ([docs](https://setuptools-scm.readthedocs.io/en/latest/config/)), and `[tool.setuptools_scm]` is
  empty here (`pyproject.toml:128`). A Julia tag could therefore become the Python package's version. Before the
  first Julia tag, set `scm.git.describe_command` to match `v[0-9]*` only, which needs setuptools-scm ≥ 9.2 (9.0 and
  9.1 are yanked), or `tag = {prefix = "v"}`, which needs ≥ 10.1. So bump `[build-system]`'s `setuptools_scm>=8`.
  Tags `v4.3.0` then `Flatland-v0.1.0` gave the Python version `0.1.1.dev1` today (reproduced). M1 already excludes
  `julia/` from release-please's root package (`exclude-paths`) and from the Python sdist (`MANIFEST.in`), which
  otherwise includes the git-tracked tree.

### 4.8 Documentation

- Documenter.jl docs in `julia/Flatland/docs`, with Literate.jl tutorials replacing notebooks and
  DocumenterCitations.jl for the references above.
- A "Julia backend" page in the flatland-book.
- A contributor guide on changing semantics in both implementations: bump `SEMANTICS_VERSION`, regenerate traces,
  both sides green. `CODEOWNERS` for `julia/` and the step code.

### 4.9 Milestones

Each milestone is one PR of under 1000 added lines (generated data excluded), except where it says "2 PRs".

| # | milestone | acceptance criterion |
|---|---|---|
| M0a | Python: `sorted()` over order-dependent set iterations; fix bugs 6–8 (§3.10) | failing tests first; regression corpus unchanged or updated knowingly |
| M0b | Python: JSON state schema v1 (§4.2) | pickle ↔ JSON round trip on every committed fixture |
| M0c | Python: `flatland-trace/1` exporter | re-running it gives byte-identical output |
| M1 | `julia/Flatland` scaffold, `julia.yml`, Aqua/JET/ExplicitImports/format/docs | CI green |
| M1b | spike: juliacall calls a trivial function inside the Docker image | cold-start time and image size recorded; fork/spawn behaviour documented |
| M2 | `EntryPointNetwork` for grid (resources included) and graph | exhaustive table equality with Python |
| M3 | integer ticks and the three `_candidate_*` functions | every case of `design_by_contract.md` Table 1 matches; zero allocations |
| M4 | MotionCheck | differential fuzzing passes; O(n + p) |
| M5 | `step!` + state import/export, malfunctions from an explicit schedule | positions, speeds, distances, states and dones of all golden traces bit-identical (rewards come with M6, seeded malfunctions with M7); no allocations once buffers are sized |
| M6 | distance map; rewards (`DefaultRewards`, `ECML2026Rewards`, `DelayRewards`, `PunctualityRewards`, `BasicMultiObjectiveRewards`), 2 PRs | distance arrays and reward traces equal |
| M7 | `NumpyLegacyRNG` + `ParamMalfunctionGen` | streams equal numpy's; seeded episodes identical without a schedule |
| M8 | Python wrapper and facade (`flatland/juliapkg.json`, `[julia]` extra) | `tests/ml` PettingZoo/RLlib smoke tests pass on the Julia backend; built-in Python observations identical through the facade |
| M9 | Julia tree observation (segment graph) + predictor, 2 PRs; before registration, since without observations the Julia backend gives users no end-to-end speed-up (§5) | observation arrays equal to the Python versions after §3.1 |
| M10 | register Flatland.jl 0.1.0 | registered in General; `pip install flatland-rl[julia]` resolves the registered version, offline in the baked image |
| M11 | batched `step_many!` without the GIL, PrecompileTools, AirspeedVelocity, `-julia` Docker tag | budgets set in M1b's spike, e.g. first `step` within 10 s of import in the baked image and batched throughput at least the end-to-end target of §5 [estimate] |
| M12 | docs, flatland-book page, dual-implementation guide; optionally an evaluator `--backend julia` | docs deployed |

Expected size [estimate]: the simulation core is about 7.5k lines of Python code
(`envs`, `core`, `step_utils`, `trajectories`, `env_generation`, excluding docstrings). The Julia core through M10
should be 4–6k lines plus tests.

### 4.10 Risks

- **Drift between the two implementations.** This is the main risk: Python semantics still change weekly (#542,
  #545, the pending 5.0.0 release). Mitigations: `SEMANTICS_VERSION`, conformance on every PR touching either side,
  `CODEOWNERS`, and an explicit decision on when, if ever, the Python core is retired.
- **Startup latency** in evaluation pods and Ray workers, and the `fork` safety of a process with Julia running.
  Measured in M1b before anything else depends on it.
- **Docker.** Image size, and multi-arch builds: precompiling arm64 under QEMU is slow, so use native arm64
  runners.
- **Contributors.** Fewer of them read Julia. Keep the Julia surface small (step, rewards, distance map,
  observations), and keep Python the reference.
- **Custom Python observation builders** keep working through the facade, but at Python speed. Only the built-in
  ones get the Julia speed-up.
- **Amdahl.** With a built-in observation builder, the dynamics are 2–4% of a step today [measured]. Until the
  observations are ported (M9) and the wrapper avoids building Python dicts every step (`step_arrays`), a faster
  `step!` barely shows end to end.

## 5. How the two prongs combine

Recommended order:

1. **Python quick wins: P1–P7.** Apart from P1's default change, they don't change outputs. They remove most of the
   measured cost for observation users: the tree observation, global observation, predictor, checks and the
   duplicate distance map.
2. **Python representation changes: P8, P9, P10.** Integer ticks, integer entry-point tables and the segment graph
   *are* the Julia data model (§4.3). Landing them in Python first speeds up today's users. It also reduces the
   Julia port to translating table-driven code, and makes conformance failures easier to localise.
3. **Julia milestones M0a–M5** in parallel with step 2, since M0a–M0c are Python-side anyway. Then M6–M12.
4. **The remaining Python items** (P11–P15) whenever convenient.

The decision point is after M5, measured end to end through the Python API (`step_arrays`, then `step`) for configs b
and c, against Python with P1–P9 landed. If the Julia backend is not at least 10× faster at M and L for config b
[estimate: the dynamics alone should gain two orders of magnitude], stop after Prong 1 and keep Python only. Config c
only counts after M9.

## 6. References

- Z. Chen, J. Li, D. Harabor, P. J. Stuckey. *Scalable Rail Planning and Replanning with Soft Deadlines.*
  [arXiv:2306.06455](https://arxiv.org/abs/2306.06455), 2023. The quotes in §1 are from the
  [HTML version](https://ar5iv.labs.arxiv.org/html/2306.06455).
- F. Laurent et al. *Flatland Competition 2020: MAPF and MARL for Efficient Train Coordination on a Grid World.*
  [arXiv:2103.16511](https://arxiv.org/abs/2103.16511), 2021.
- S. Mohanty et al. *Flatland-RL: Multi-Agent Reinforcement Learning on Trains.*
  [arXiv:2012.05893](https://arxiv.org/abs/2012.05893), 2020.
- [Flatland book](https://flatland-association.github.io/flatland-book/): key concepts, persistence, ECML 2026
  [levels](https://flatland-association.github.io/flatland-book/challenges/ecml2026/levelconfig.html) and
  [evaluation](https://flatland-association.github.io/flatland-book/challenges/ecml2026/eval.html).
- Upstream issues and PRs:
  - performance: [#6](https://github.com/flatland-association/flatland-rl/issues/6),
    [#68](https://github.com/flatland-association/flatland-rl/pull/68),
    [#87](https://github.com/flatland-association/flatland-rl/pull/87),
    [#182](https://github.com/flatland-association/flatland-rl/pull/182),
    [#206](https://github.com/flatland-association/flatland-rl/issues/206),
    [#439](https://github.com/flatland-association/flatland-rl/issues/439),
    [#449](https://github.com/flatland-association/flatland-rl/pull/449),
    [#465](https://github.com/flatland-association/flatland-rl/issues/465);
  - design: [#129](https://github.com/flatland-association/flatland-rl/issues/129),
    [#281](https://github.com/flatland-association/flatland-rl/issues/281),
    [#364](https://github.com/flatland-association/flatland-rl/issues/364),
    [#462](https://github.com/flatland-association/flatland-rl/issues/462),
    [#542](https://github.com/flatland-association/flatland-rl/pull/542),
    [#549](https://github.com/flatland-association/flatland-rl/issues/549).
- Julia/Python tooling:
  - [PythonCall.jl / juliacall](https://juliapy.github.io/PythonCall.jl/stable/), [pyjuliapkg](https://github.com/JuliaPy/pyjuliapkg),
    [PySR](https://github.com/astroautomata/PySR) (precedent for a Python package backed by a Julia package);
  - [Registrator](https://github.com/JuliaRegistries/Registrator.jl), [TagBot](https://github.com/JuliaRegistries/TagBot),
    [CompatHelper](https://juliaregistries.github.io/CompatHelper.jl/dev/),
    [AirspeedVelocity.jl](https://github.com/MilesCranmer/AirspeedVelocity.jl),
    [julia-downgrade-compat](https://github.com/julia-actions/julia-downgrade-compat),
    [setuptools-scm configuration](https://setuptools-scm.readthedocs.io/en/latest/config/).
- `gdalle/Flatland.jl` ([GitHub](https://github.com/gdalle/Flatland.jl)): an earlier, unregistered skeleton (21 lines
  of source) with CI scaffolding. `julia/Flatland` reuses its name and UUID, so it can be archived with a pointer here.

## Appendix A. Reproducing the measurements

The scripts are in [`benchmarks/revamp/`](benchmarks/revamp/README.md):
- `flatbench.py` times every phase of `reset()` and `step()` with timestamps around the calls that bracket them, or
  records cProfile files;
- `run_all.sh` runs the jobs of §2 one after the other, with timeouts;
- `micro.py` runs the micro-benchmarks of §2.4;
- `analyze.py` turns the outputs into the tables of §2.

The numbers above were produced on commit `51f5621c` with CPython 3.13, the pure-Python sources, and one pinned core
(`taskset -c 3`, `prlimit --as=3.5GB`); peak memory was 2.0 GB (L/a).
