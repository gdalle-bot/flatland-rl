# Flatland revamp: algorithmic speed-ups and a Julia rewrite

This document plans how to make the Flatland simulator fast. It has two prongs:

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

  With 500 agents (ECML 2026 scale), the default config needs 0.8–1.15 s per step. A full 3122-step episode would
  take 40–60 minutes, more than ECML's 30-minute budget per scenario.
- **The biggest Python wins are not in the dynamics.**
  - Rewrite the tree observation, predictor and global observation around precomputed tables.
  - Remove a conflict check in the tree observation that never fires. Deleting it alone cuts the tree observation's
    time by 38%, with identical outputs.
  - Make the step pre/post-condition checks opt-in (+40–50% per step).
  - Stop computing the distance map twice per `reset()` (up to 2.8 s at 500 agents).
- **The dynamics are about 16–18 µs per agent per step** in Python [measured]. Each of these should give a few ×
  [estimate]:
  - integer speed ticks instead of `Fraction` (38× on the speed-counter arithmetic alone [measured]);
  - integer entry-point tables instead of 56 `lru_cache`s;
  - skipping inactive agents. A Julia core with struct-of-arrays state and zero allocations should reach two orders of
  magnitude [estimate].
- **The Julia rewrite is a dynamics-parity port, not a fork.**
  - Python keeps scenario generation and stays the reference implementation.
  - Julia must reproduce Python's trajectories bit for bit, checked on golden traces in CI.
  - The plan is 16 milestones (M0a–M12), each a PR of under 1000 lines.
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

They also report that in NeurIPS 2020 "the winning software only used 30%" of the budget for planning. For
Flatland 3, "the total planning time is only 5% of the total time".

The Flatland 2020 competition report ([arXiv:2103.16511](https://arxiv.org/abs/2103.16511)) gave submissions 8 hours
in total, "10 minutes per environment for initial planning, then 10 seconds … per timestep", on 4 CPU cores. The
original Flatland paper ([arXiv:2012.05893](https://arxiv.org/abs/2012.05893)) promised "fast experimentation" as
the reason to simplify the physical simulation in the first place.

**Current challenges are larger.** The
[ECML 2026 levels](https://flatland-association.github.io/flatland-book/challenges/ecml2026/levelconfig.html) go up
to 532 agents. The [evaluation rules](https://flatland-association.github.io/flatland-book/challenges/ecml2026/eval.html)
allow "up to 30 minutes per scenario", 5 hours in total, on 4 CPU cores with 15 GB.

**RL training multiplies the cost.** At 5 steps/s (§2, 200 agents, tree observation), one million environment steps
take about 55 hours on one core. At the dynamics-only speed (316 steps/s), they take under an hour.

**Upstream has worked on speed for years, with mixed results.**
- [#6](https://github.com/flatland-association/flatland-rl/issues/6) found `MotionCheck.find_conflicts` at about 50%
  of `step()`. [#87](https://github.com/flatland-association/flatland-rl/pull/87) fixed it by replacing networkx with
  dicts. It is based on "Bochatay (2024)", cited in `flatland/envs/agent_chains.py:270`; I could not find that
  report online.
- [#68](https://github.com/flatland-association/flatland-rl/pull/68) moved A* to a heap.
- A September 2026 series of micro-optimisations each gained 1–18% on sub-functions.
- Upstream CI's profiling chart (Python 3.13, 200 agents, checks off, no observations) shows that cumulative `step`
  time swung 3× across releases: 1.02 s in v4.1.1, 3.07 s in v4.2.5, 1.26 s on `main`. The Cython build of
  [#449](https://github.com/flatland-association/flatland-rl/pull/449) gives 1.25 s, i.e. nothing, because
  `step()` no longer drives the compiled state machine.
- [#206](https://github.com/flatland-association/flatland-rl/issues/206) ("Fraction", "speed counter", "rewards") is
  still open.

## 2. Baseline measurements

**Setup.**
- Commit `51f5621c`, CPython 3.13.15, pure-Python sources (no Cython build; see §1 for why that doesn't matter for
  `step()`). One pinned core of a 4-core Linux VM.
- Scenarios: `sparse_rail_generator(max_rails_between_cities=2, max_rail_pairs_in_city=2)`, speeds
  {1, ½, ⅓, ¼} uniform, `ParamMalfunctionGen(rate=1e-4, 15–50 steps)`, seed 1. Actions are seeded uniform-random,
  pre-generated, as in upstream's profiling script.
- A fresh env for every repetition.
- Timings are wall-clock between timestamps on the bracketing calls. cProfile was used only for within-phase splits
  and call counts. The driver and analysis scripts are in Appendix A.

| scenario | grid | agents | cities | max episode steps | basis |
|---|---|---|---|---|---|
| S | 30×30 | 7 | 2 | 278 | Flatland 3 `Test_00` |
| M | 100×100 | 200 | 5 | 1656 | upstream CI profiling setup |
| L | 158×158 | 500 | 41 | 3122 | Flatland 3 `Test_14` geometry, ECML 2026 agent count |

| config | observation builder | `check_step_pre_post_conditions` |
|---|---|---|
| a | constructor default (`GlobalObsForRailEnv`) | constructor default (`True`) |
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
| first step | 8.1 ms (45× a normal step) | 9.8 ms | 14.8 ms | `ParamMalfunctionGen` draws `rand(1_000_000)` on its first call, even at rate 0 |
| last step | 3.1 ms | 85 ms | **2 761 ms** | the env's own distance map is computed lazily, the first time a reward reads it at episode end; it is a second full computation after the timetable's |

### 2.4 Micro-benchmarks

Run in-process, best of several repetitions; `bench/micro.py` in Appendix A.

| what | result |
|---|---|
| tree obs at M, step 1240 (95 agents on map): `get_many` with vs without the dead conflict block (§3.1) | **182.5 → 113.5 ms (−38%)**; observations identical (`repr` equal); 24 930 `fast_delete` list copies, 0 `fast_where` calls |
| global obs at M: the six `isfinite`/`isnan` asserts | 0.12 ms per agent, i.e. 24 ms of the 145 ms the builder needs per step |
| speed-counter arithmetic per agent-step (cell exit, acceleration, both distance formulas, the entry test) | `Fraction` with the existing `lru_cache`s: 6.8 µs; `Fraction` uncached: 6.7 µs; **integer ticks: 0.18 µs (38×)** |
| `Fraction(1,3) == 0` vs `== 0.0` | 88 ns vs 1 551 ns |
| `Fraction + Fraction` vs `int + int` / `hash(Fraction)` | 479 ns vs 18 ns / 179 ns |
| transition lookup: `rail.get_transitions(ep)` (cache hit) vs dict vs list index | 353 ns vs 53 ns vs 16 ns |
| `RandomState.rand(1_000_000) < p` (first malfunction draw) | 6.5 ms |
| distance map at M: one `_compute` (warm) | 37 ms; **6 unique target sets for 200 agents**, stored as 200 copies (61 MB) |
| target dedupe with list membership vs dict | 0.24 vs 0.06 ms at n = 200; 3.0 vs 0.6 ms at n = 2000 (real but small) |
| `MotionCheck.find_conflicts`, m agents queued for one cell | 0.04 / 0.3 / 2.4 / 32 / 551 ms for m = 10 / 25 / 50 / 100 / 200: **grows as m^3.6** |

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
  - `_reconstruct_shortest_path` is 44%: the timetable generator builds every agent's full shortest path only to
    take its `len()`;
  - the distance map's `_compute` is 40%;
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
| T1 | Delete the dead conflict block: **−38% of tree obs time at M** [measured, §2.4]. | none |
| T2 | Replace the string parsing with `int.bit_count()` and an integer comparison (or per-cell tables built at `reset()`). | none |
| T3 | Record visited cells only when rendering is requested. | none (`dev_obs_dict` is only for rendering) |
| T4 | Walk a **segment graph**: contract runs of cells without a decision into edges, with length, target and switch metadata precomputed at reset, and bucket agents by segment once per step. Per-agent cost drops from O(b^D·ℓ) to O(b^D), with ℓ the mean segment length. `DecisionPointGraph` (`flatland/envs/graph/graph_simplification.py:28-98`) already builds this contraction for graph envs. | none if feature semantics are kept |
| T5 | Restore `dist_potential_conflict` correctly with a `{(t, cell): [handles]}` index built once per step. Separate PR. | obs |
| T6 | Until T5 lands, call the predictor only when something reads its output. Today `obs_builder.predictions` and the renderer's `env.dev_pred_dict` are the only readers besides the dead check. | api |

**Shortest-path predictor** (`flatland/envs/predictions.py:96-175`). It costs 15–18% of a step at M [measured].
Every step, it rebuilds every agent's path to depth P from the distance map:
- `_reconstruct_shortest_path` (`flatland/core/distance_map.py:149-183`) allocates an attrs `Waypoint` per cell;
- `shortest_path = shortest_path[1:]` inside the loop (`predictions.py:151,168`) makes each path O(P²);
- `if handle:` (`predictions.py:121`) treats handle 0 as "all agents".

Fixes:
- Precompute a **next-hop table** per unique target set when the distance map is built. Reconstruction becomes P
  array lookups, with tie-breaking kept identical to `_reconstruct_shortest_path`.
- Reuse last step's path when the agent has only advanced along it.

Impact: none.

**Global observation** (`observations.py:552-604`, the **constructor default**: `rail_env.py:177,1603`). It costs
97–99% of a step at M and L [measured]. For each agent `get()`:
- allocates fresh H×W×2 and H×W×5 arrays;
- loops over **all** agents, calling `derived_state()` twice per pair. cProfile counts 1.6M `derived_state` calls in
  20 steps at M, i.e. n² × 2;
- runs six full-array `isfinite`/`isnan` asserts (`observations.py:598-603`), two of which rescan the constant H×W×16
  rail layer. They take 24 ms of the builder's 145 ms per step at M [measured].

`reset()` fills that rail layer cell by cell via `bin()` strings (`observations.py:552-559`). The default argument
`GlobalObsForRailEnv()` is evaluated once, so **every `RailEnv` built without an explicit builder shares one
instance** [read].

Fixes:
- Default to `None` and build a new instance per env (bug fix, api).
- Build the shared layers once per step in O(n + HW), then per agent copy them and patch O(1) cells.
- Check the rail layer once at reset.
- Vectorise the bit unpacking with `(grid[..., None] >> np.arange(15, -1, -1)) & 1`.

Impact: none, apart from the default-instance fix.

**Gym flattening** (`flatland/ml/observations/flatten_tree_observation_for_rail_env.py:64-124`). It concatenates
small numpy arrays recursively per node, and normalises with Python while-loops. Fix: fill one preallocated array at
precomputed offsets. Impact: none.

### 3.2 Defaults and per-step work that does nothing

| # | where | what | fix | impact |
|---|---|---|---|---|
| D1 | `rail_env.py:185,1612` | `check_step_pre_post_conditions=True` by default: +39% per step at S, +50% at M [measured] | Default to `False`; turn it on in the test suite via `conftest.py`. | none |
| D2 | `rail_env.py:1687-1696`, called at `:1707` | `_update_agent_positions_map` is an O(n) loop every step; `agent_positions` is read nowhere in `flatland/` [read] | Deprecate, compute lazily on access. | api (deprecation) |
| D3 | `rail_env.py:590,609,699-707` | per agent and step: `state = agent.state` (a `derived_state()` call) used only in a warning; `frozenset(agent.targets)` rebuilt although targets are fixed per episode; `any(self.rail.get_transitions(...))` computed only to decide whether to warn | Hoist or drop. | none |
| D4 | `rail_env.py:425-448` | `get_info_dict` is 10% of the dynamics [measured]; it calls `derived_state()` twice per agent | Compute the state once per agent. | none |

### 3.3 Distance maps and `reset()`

- **Computed twice per reset** [measured, read]:
  - the timetable generator builds its own `DistanceMap` with "fake agents" and computes it
    (`timetable_generators.py:60-79`). The branch is always taken, since sparse lines have at least two waypoint
    groups;
  - the env's own map is computed lazily on first read, which is the last step for configs a/b/d (2.8 s at L).

  Fix: compute once and share it.
- **Recomputed after every load and clone.** `RailEnvPersister` assigns the stored map and then calls `_compute`
  anyway (`persistence.py:312-314`) [read].
- **Data structures** (`core/distance_map.py:66-88`, `core/distance_map_walker.py`, `envs/grid/distance_map.py:30-41`)
  [read]:

  | today | proposal |
  |---|---|
  | K ≤ 4 sequential backward BFS walks per target, with tuple keys in a `defaultdict` | **one multi-source BFS** per unique target set, over a precomputed int predecessor table |
  | an `(n, H, W, 4)` float64 array, with a full copy per agent sharing a target: 61 MB at M for **6 unique target sets** [measured], 400 MB at L | `(U, H, W, 4)` int32 plus an agent-to-row index, with U the number of unique target sets (1.8 MB at M) |
  | targets deduplicated with list membership and `.index`, O(n²); measured small: 3 ms at n = 2000 | a dict keyed by `frozenset(targets)` |

  The walker's comment says a *shared visited set across sequential walks* broke looped layouts. A true
  multi-source BFS does not have that problem, but needs a looped-layout regression test.

  Open PR [#542](https://github.com/flatland-association/flatland-rl/pull/542) adds fractional edge lengths and
  turns the BFS into a label-correcting walk over an object array of `int`/`Fraction`. With integer ticks (§3.4),
  that array stays integer.
- **Line generator** (2.5 s at L, 41% of reset [measured]):
  - `check_path_exists` runs a DFS per agent and orientation (`line_generators.py:85-88`);
  - `get_k_shortest_paths(k=1)` serves as a reachability test, up to 16 calls per leg (`line_generators.py:194-201`).

  Fix: one cached BFS per source; [#462](https://github.com/flatland-association/flatland-rl/issues/462) proposes the
  same for k = 1. Impact: none if the same orientation is chosen.
- **Rail generator** [read]:
  - A* evaluates the expensive `validate_new_transition` before testing `respect_transition_validity`, which is
    `False` here, so the result is discarded (`core/grid/grid4_astar.py:116-118`). Swap the operands. Impact: none.
  - `_fix_transitions` iterates `list(city_cells) + inter_city_lines` with duplicates and builds a fresh
    `RailEnvTransitions()` per fixed cell (`rail_generators.py:874,899`). Impact: rng if the iteration order
    changes. `list(city_cells)` iterates a set, so the order is CPython-specific.
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
- `RailEnv` defaults the deltas to floats 1.0 / −1.0 (`rail_env.py:1610-1611`), unlike `AbstractRailEnv`. Accept
  only exact rationals.
- `_pseudo_fractional` snaps floats to "nice" fractions; keep it only at the API edge.

Impact: none, checked by bit-identical trajectories on the regression corpus.

### 3.5 Topology: integer entry points and precomputed tables

Today, entry points are nested tuples `((r, c), d)`. The transition map answers queries through 56 `lru_cache`
decorators. Instance-method caches are keyed by `self` through a Python-level `__hash__`
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

Tuples remain derived properties at the API boundary. The same tables describe a `GraphTransitionMap`, which unifies
the grid and graph back ends ([#281](https://github.com/flatland-association/flatland-rl/issues/281)). Graph tables
must be built in Python, because `apply_action_independent` takes the first successor in networkx insertion order
(`envs/graph/rail_graph_transition_map.py:144-151`). Impact: none.

### 3.6 Skip inactive agents

Every step runs the whole collect, MotionCheck, distribute and reward pipeline for every agent: waiting, removed
`DONE`, or idle. Keep index sets of active, waiting (sorted by `earliest_departure`) and done agents, and loop over
active ones only. Precondition: check, per reward class, that per-step rewards are zero for inactive agents. In
`BaseDefaultRewards.step_reward` an agent off the map with `old_entry_point is None` records nothing
(`rewards.py:189-240`). Impact: none.

### 3.7 MotionCheck

`_fix_conflicts` pops the head of its conflict list with `target_conflicts = target_conflicts[1:]`
(`agent_chains.py:368`), and `_stop_and_update_target_conflicts` calls `list.remove` (`:379`). That is O(p²) in the
number p of conflicting pairs, and p grows quadratically with the number m of agents queued for one cell [read].
§2.4 measures m^3.6: 551 ms with 200 agents queued [measured]. In the random-policy runs, MotionCheck is 13% of
the dynamics at M and 5% at L [measured].

Fix: an index cursor (or `deque`) and per-resource sets.

**The queue order is part of the semantics.** In a fuzzing run of the design review, shuffling the initial conflict
queue changed the outcome in 538 of 100 000 random configurations. In one example, three agents queue for one cell
and the shuffled order stops all three. So any rewrite must keep the exact order: pairs built in handle order, each
resource's agents in insertion order, FIFO with appends. It must also be accepted by differential fuzzing against
the current implementation. Iteration over the `deadlocked` set (`:325`) showed no effect, but `sorted()` there is
cheap insurance. Impact: none.

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
| `trajectories/trajectory_observation_callbacks.py:32,34` | recomputes `env._get_observations()` instead of reusing the step's observations | reuse `env.obs_dict` | none |
| `trajectories/trajectories.py:181-193` | `info` is stored as Python `repr` (`Fraction(1, 2)`, `<TrainState.MOVING: 3>`) and parsed back with regexes and `ast.literal_eval` per row | typed columns, or Parquet | file format |
| `evaluators/client.py:167,186` | two synchronous Redis round trips per step | pipeline them | none |
| `envs/rail_env_shortest_paths.py:5` | `import matplotlib.pyplot` at module level, for a debug plot: 0.29 s of the 0.48 s it takes to import `rail_env` [measured] | lazy import | none |

### 3.10 Bugs and parity hazards found along the way

Found by reading [read]. Each needs a failing test before its fix.

1. **TreeObs conflict check never fires** (§3.1). `dist_potential_conflict` is always `inf`.
2. **The predictor treats handle 0 as "all agents"**: `if handle:` at `predictions.py:121`.
3. **`coordinate_to_position` collides when H > W.** It encodes `col·width + row` (`grid_utils.py:288`).
4. **The recorded deadlock flag is always 0.** `RecordStepsEffectsGenerator` tests
   `position in env.resource_check.deadlocked` (`record_steps_effects_generator.py:36`), but `deadlocked` holds
   agent handles.
5. **Every `RailEnv` without an explicit builder shares one `GlobalObsForRailEnv`** (`rail_env.py:177,1603`).
6. **Reward bookkeeping leaks across episodes.** `Rewards` has no `reset()`, and `RailEnv.reset()` keeps the same
   object. `arrivals` / `departures` / `stopped_waypoints` (`rewards.py:183-187`) leak, and the reward state is not
   persisted either.
7. **`ConditionalMalfunctionEffectsGenerator._num_malfunctions` is never reset** (`malfunction_effects_generators.py:103`).
   `max_num_malfunctions` is therefore a lifetime cap.
8. **`DummyPredictorForRailEnv` leaves an agent corrupted on error.** It overwrites `agent.current_entry_point` while
   predicting and only restores it at the end, so the `raise` at `predictions.py:79` leaves it changed.
9. **PettingZoo `render()` cannot work.** It calls `self._wrap.render()`
   (`ml/pettingzoo/pettingzoo_parallel_rail_env.py:95`), but `RailEnv` has no `render` method.
10. **A tie depends on set order** (parity hazard): `np.argmax` over `list(stopped_wps)`, a set
    (`rewards.py:323-336`).
11. **MotionCheck needs exclusive occupancy** (contract hazard). It assumes each resource is held by at most one
    agent. When fuzzing with two agents holding the same cell, `find_conflicts` kept appending conflict pairs until
    it ran out of memory [measured]. The env guarantees exclusivity, but a port or a caller building MotionCheck
    inputs by hand must too. Assert it on `add_agent`.

### 3.11 PR sequence

Stacked PRs against `revamp`. Each is under 1000 added lines, has non-regression tests, and gets an entry in the
profiling table:

| PR | content | impact |
|---|---|---|
| P1 | D1: checks opt-in; enabled in the test suite | none |
| P2 | Global obs: per-env default instance, shared layers, vectorised reset, asserts at reset | api (bug fix) |
| P3 | Tree obs T1–T3 | none |
| P4 | Predictor next-hop table and incremental paths | none |
| P5 | Distance map: dict dedupe, multi-source BFS, int32 rows, compute once per reset/load/clone, reused by the timetable | none |
| P6 | MotionCheck order-preserving O(n + m), with a differential fuzz test | none |
| P7 | D2–D4, inactive-agent index sets (§3.6) | none / api |
| P8 | Integer speed ticks, in 2–3 PRs | none |
| P9 | Integer entry-point tables (grid, then graph), in 2–3 PRs | none |
| P10 | Tree obs T4 (segment graph) | none |
| P11 | Reset: line-generator BFS, A* operand swap, `get_k_shortest_paths` | none / rng |
| P12 | Malfunction fast path behind a flag | rng |
| P13 | Trajectories: snapshot interval, observation reuse, typed info | api / format |
| P14 | Bug fixes 2–10, one PR each | obs / api |
| P15 | CI profiling covers the defaults, observation builders, load/clone and L-size scenarios; today only config b at M is profiled | — |

The "none" PRs are checked against the existing regression corpus: `benchmarks/benchmark_episodes.py`, 133
episodes replayed step by step.

## 4. Prong 2: a Julia rewrite

### 4.1 Goals and non-goals

Goals:
- **Bit-identical dynamics.** Given the same initial state, actions and effects, Julia produces the same positions,
  speeds, distances, states, rewards and dones as Python at every step.
- **End-to-end seeded parity** while generation stays in Python. A small port of numpy's legacy `RandomState`
  (MT19937, gym-style seeding as in `utils/seeding.py`, `rand`, masked `randint`) is enough to reproduce
  `ParamMalfunctionGen`.
- **Zero allocations per `step!`**, type-stable code, and threaded batches of envs.
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
struct EntryPointNetwork            # grid and graph alike, built once per rail
    succ::Matrix{Int32}             # (5 actions, nEP) → next entry point, 0 = invalid
    resource::Vector{Int32}         # entry point → resource id (cell, or cell+axis for level-free crossings)
    pred_ptr::Vector{Int32}         # CSR predecessors (distance maps)
    pred_idx::Vector{Int32}
    labels::Vector{EPLabel}         # ((r,c),d) or graph node string, for I/O only
end

struct Agents                       # struct of arrays, one entry per agent
    ep::Vector{Int32}               # current entry point (0 = off map)
    next_ep::Vector{Int32}
    target_set::Vector{Int32}       # row in the distance map
    speed::Vector{Int32}            # ticks of 1/den[i]
    distance::Vector{Int32}
    max_speed::Vector{Int32}
    den::Vector{Int32}
    malfunction::Vector{Int32}      # remaining steps
    earliest_departure::Vector{Int32}
    done_at::Vector{Int32}
end

mutable struct RailEnv{R<:AbstractRewards,E<:AbstractEffects,O<:AbstractObservation,RNG<:AbstractRNG}
    net::EntryPointNetwork
    agents::Agents
    rewards::R
    effects::E
    obs::O
    rng::RNG
    buffers::StepBuffers            # preallocated: candidates, MotionCheck queues, reward vectors
    t::Int
end

step!(env, actions::AbstractVector{<:Integer})  # mutates env, returns nothing, allocates nothing
```

- **`step!` mirrors the Python control flow.** It follows effects (step start) → collect → MotionCheck →
  distribute → rewards → effects (step end), branch for branch as specified in `design_by_contract.md`. The `_check_*`
  invariant net is compiled in or out through a type parameter.
- **MotionCheck** uses preallocated arrays: an occupant per resource, a FIFO queue in the Python order (§3.7), and
  an O(n + m) resolution.
- **Distance maps** are multi-source BFS over the CSR predecessors, stored as `Array{Int32}(4, W, H, U)`. Its
  transpose is a zero-copy view in Python's `(U, H, W, 4)` layout.
- **Graphs.jl** is used for distance maps and the decision-point graph only, never in `step!`.
- **Observations** fill preallocated buffers (`observe!(buf, obs, env)`). The tree observation walks the segment
  graph (§3.1 T4).
- **Randomness** comes from an explicit `rng` field: `NumpyLegacyRNG` for parity, `Xoshiro` otherwise. There is no
  global RNG.
- **Batches.** A `Vector{RailEnv}` stepped with OhMyThreads.jl `tforeach`. Python calls it without the GIL
  (`_jl_call_nogil`).

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
| pytest, pytest-xdist | tests | Test + TestItems.jl 1.1 / TestItemRunner.jl 1.3 |
| flake8, deptry | lint | JuliaFormatter.jl 2.14 (blue style, as in `gdalle/Flatland.jl`), Aqua.jl 0.8, ExplicitImports.jl 1.15, JET.jl 0.13 |
| cProfile, line_profiler | profiling | Profile stdlib, Chairmarks.jl 1.3, AirspeedVelocity.jl 0.6, AllocCheck.jl 0.2 |
| jupyter-book (flatland-book) | docs | Documenter.jl 1.19 + Literate.jl 2.21 + DocumenterCitations.jl 1.5; a page in the flatland-book |
| juliacall (new) | Python wrapper | PythonCall.jl 0.9.37 + CondaPkg.jl 0.2 (test-only, for differential tests) |
| — | latency | PrecompileTools.jl 1.3; optional PackageCompiler sysimage |

### 4.5 Python interop

**Layout.**
- `julia/Flatland/` holds the Julia package.
- `flatland/julia/` holds the wrapper code. It is behind a `[julia]` extra that adds `juliacall`.
- `flatland/juliapkg.json` declares the Julia dependency. It must sit at `flatland/`, not `flatland/julia/`.
  [pyjuliapkg](https://github.com/JuliaPy/pyjuliapkg) only scans the top of each `sys.path` entry and one level
  down, so a deeper file works in editable installs and silently fails in regular ones.
  [PySR](https://github.com/MilesCranmer/PySR/blob/master/pysr/juliapkg.json) uses the same layout.
- Before registration, `juliapkg.json` points at the repository with `url` + `subdir` + `rev`. In development and
  CI, `PYTHON_JULIACALL_PROJECT` points at a project that `dev`s the checkout.

**Data path.**
- `reset()` still runs Python generation, exports the state (§4.2) and loads it into Julia.
- `step(action_dict)` converts the dict to an `Int8` vector and calls `step!`.
- Observations and the distance map come back as zero-copy views. juliacall's `ArrayValue` gives a numpy view for
  strided arrays of supported element types
  ([docs](https://juliapy.github.io/PythonCall.jl/stable/juliacall-reference/)).
- A batched `step_many` amortises the call overhead.

**The `RailEnvView` facade.**
- Observation builders keep `self.env` from `reset(env)`, so the facade keeps a stable identity and is refreshed in
  place once per step.
- Agent views index the shared arrays and build `Fraction`s lazily.
- `targets` keeps waypoint order, because builders use `next(iter(agent.targets))`.
- The only mutation API is `set_malfunction(handle, n, phase)`, which is all that the test effects generators need.

**Startup latency.**
- A PrecompileTools workload.
- Docker images bake the depot and set `PYTHON_JULIAPKG_OFFLINE=yes`, with an optional sysimage.
- Ray / multiprocessing workers must use spawn rather than fork after Julia has started. This is to verify in M1b.

### 4.6 Testing

| layer | what | tools |
|---|---|---|
| unit | port the transition, speed, MotionCheck and reward unit tests (`test_flatland_core_transitions.py`, `test_speed_counter.py`, `test_agent_chains.py`, `test_rewards.py`) | `@testitem` (TestItems.jl), run by TestItemRunner.jl |
| tables | `EntryPointNetwork` against Python, exhaustively: every cell type × 4 headings × 5 actions, plus off-grid neighbours | golden JSON |
| conformance | every `flatland-trace/1` trace replays bit-identically, comparing rationals exactly (stricter than the Python tests' `np.allclose`) | golden traces from CI |
| differential | fuzz MotionCheck (about 10⁵ random configurations including swaps, chains and level-free crossings), then whole episodes, against the live Python implementation | PythonCall.jl + CondaPkg.jl, nightly |
| properties | invariants of the Python check net: at most one agent per resource except level-free crossings, speed ≤ max speed, distance in [0, 1], done agents off map | Supposition.jl |
| quality | stale/missing deps, method ambiguities, type stability of `step!`, explicit imports, zero allocations | Aqua.jl, JET.jl (`test_package`, `@test_opt`), ExplicitImports.jl, AllocCheck.jl / `@allocated` |
| Python wrapper | `tests/ml` smoke tests and the Replay tests, parametrised over the backend | pytest `-m julia` |
| performance | `benchmark/benchmarks.jl` SUITE over S/M/L | AirspeedVelocity.jl |

### 4.7 CI

A new path-filtered workflow `julia.yml` triggers on `julia/**`, `conformance/**`, `flatland/envs/**` and
`flatland/core/**`:

- **Test matrix:** Julia `lts`, `1` and `pre` on ubuntu; macOS and Windows on `1`.
  - Steps: `julia-actions/setup-julia@v2`, `julia-actions/cache@v2`, `julia-buildpkg` and `julia-runtest` with
    `project: julia/Flatland`, `julia-processcoverage` with `directories: julia/Flatland/src`, Codecov.
- **Lower compat bounds:** `julia-actions/julia-downgrade-compat@v2` with `projects: julia/Flatland`.
- **Formatting check** (JuliaFormatter).
- **Docs:** `julia-docdeploy` with `project: julia/Flatland`, and `deploydocs(dirname="julia")` so the docs land
  in a subfolder of `gh-pages`.
- **Benchmarks:** the AirspeedVelocity action has no subdirectory input, so call the `benchpkg` CLI with
  `--path julia/Flatland`.
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
  first Julia tag:
  - set `scm.git.describe_command` to match `v[0-9]*` only;
  - exclude `julia/` from release-please's root package.

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
| M4 | MotionCheck | differential fuzzing passes; O(n + m) |
| M5 | `step!` + state import/export, malfunctions from an explicit schedule | all golden traces bit-identical; `@allocated step!(env, a) == 0` |
| M6 | distance map; rewards (`DefaultRewards`, `ECML2026Rewards`, `DelayRewards`, `PunctualityRewards`, `BasicMultiObjectiveRewards`), 2 PRs | distance arrays and reward traces equal |
| M7 | `NumpyLegacyRNG` + `ParamMalfunctionGen` | streams equal numpy's; seeded episodes identical without a schedule |
| M8 | Python wrapper and facade (`flatland/juliapkg.json`, `[julia]` extra) | `tests/ml` PettingZoo/RLlib smoke tests pass on the Julia backend; built-in Python observations identical through the facade |
| M9 | register Flatland.jl 0.1.0 | `pip install flatland-rl[julia]` works offline in the baked image |
| M10 | Julia tree observation (segment graph) + predictor, 2 PRs | observation arrays equal to the Python versions after §3.1 |
| M11 | batched `step_many!` without the GIL, PrecompileTools, AirspeedVelocity, `-julia` Docker tag | throughput and time-to-first-step budgets met |
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

## 5. How the two prongs combine

Recommended order:

1. **Python quick wins: P1–P6.** They don't change behaviour and remove most of the measured cost for observation
   users: the tree observation, global observation, predictor, checks and the duplicate distance map.
2. **Python representation changes: P8, P9, P10.** Integer ticks, integer entry-point tables and the segment graph
   *are* the Julia data model (§4.3). Landing them in Python first speeds up today's users. It also reduces the
   Julia port to translating table-driven code, and makes conformance failures easier to localise.
3. **Julia milestones M0–M5** in parallel with step 2, since M0 is Python-side anyway. Then M6–M12.

The decision point is after M5. If the Julia `step!` is not at least 30× faster than the Python dynamics at M and L
[estimate: two orders of magnitude], stop after Prong 1 and keep Python only.

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
    [PySR](https://github.com/MilesCranmer/PySR) (precedent for a Python package backed by a Julia package);
  - [Registrator](https://github.com/JuliaRegistries/Registrator.jl), [TagBot](https://github.com/JuliaRegistries/TagBot),
    [CompatHelper](https://juliaregistries.github.io/CompatHelper.jl/dev/),
    [AirspeedVelocity.jl](https://github.com/MilesCranmer/AirspeedVelocity.jl),
    [julia-downgrade-compat](https://github.com/julia-actions/julia-downgrade-compat),
    [setuptools-scm configuration](https://setuptools-scm.readthedocs.io/en/latest/config/).
- `gdalle/Flatland.jl` ([GitHub](https://github.com/gdalle/Flatland.jl)): an earlier, unregistered 42-line skeleton
  with CI scaffolding (JuliaFormatter blue style, Aqua, JET, Documenter). It can be archived with a pointer here.

## Appendix A. Reproducing the measurements

The scripts live outside the repository; nothing below writes into the checkout.

```bash
uv venv --python 3.13 venv
uv pip install --python venv/bin/python -c requirements.txt numpy networkx matplotlib attrs fastenum cython importlib-resources msgpack msgpack-numpy
export PYTHONPATH=<repo root> PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
bash bench/run_all.sh            # timing + cProfile jobs, about 8 minutes on one core
python bench/micro.py 1240       # micro-benchmarks (1240 = peak on-map step of scenario M)
python bench/analyze.py          # Markdown tables
```

Every run goes through `timeout`, `taskset -c 3` and `prlimit --as=3.5GB`. Peak RSS was 2.0 GB, for L/a.

<details><summary><code>bench/flatbench.py</code></summary>

```python
"""Phase-level timing of Flatland's RailEnv.reset()/step().

Usage: flatbench.py SCEN CFG MODE [--reps R] [--early N] [--late N] [--t0 T] [--seed S] [--out DIR]
  SCEN in S, M, L, L2, L3; CFG in a (defaults), b (dummy obs, checks off), c (tree obs, checks off),
  d (dummy obs, checks on); MODE in timing, cprofile.
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
    if cfg == "b":
        kw.update(obs_builder_object=DummyObservationBuilder(), check_step_pre_post_conditions=False)
    elif cfg == "c":
        kw.update(obs_builder_object=TreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv(max_depth=50)),
                  check_step_pre_post_conditions=False)
    elif cfg == "d":
        kw.update(obs_builder_object=DummyObservationBuilder())
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
    ap.add_argument("--out", default=os.environ.get("S", ".") + "/out")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    t = time.time()
    res = run(args)
    res["wall_s"] = time.time() - t
    res["python"] = sys.version
    name = f"{args.out}/{args.scen}_{args.cfg}_{args.mode}{args.tag}.json"
    with open(name, "w") as f:
        json.dump(res, f)
    print("wrote", name, f"{res['wall_s']:.1f}s", f"rss={res['reps'][-1]['rss_mb']:.0f}MB")
```

</details>

<details><summary><code>bench/micro.py</code></summary>

```python
"""Micro-benchmarks backing specific claims of the revamp report. Usage: micro.py T0_M"""
import json
import math
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
    for f in (fraction_vs_int, rand1m, motion_check, lookups, distance_map):
        f()
        print(f.__name__, "done", flush=True)
    dead_branch(t0)
    print("dead_branch done", flush=True)
    global_obs(t0)
    print("global_obs done", flush=True)
    with open(os.environ["S"] + "/out/micro.json", "w") as fh:
        json.dump(OUT, fh, indent=1, default=str)
    print(json.dumps(OUT, indent=1, default=str))
```

</details>

<details><summary><code>bench/analyze.py</code></summary>

```python
"""Turn flatbench outputs into Markdown tables for REVAMP.md."""
import glob
import json
import os
import pstats
import statistics as st

OUT = os.environ["S"] + "/out"
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
```

</details>

<details><summary><code>bench/run_all.sh</code></summary>

```bash
#!/bin/bash
# Sequential profiling job list; one core (3), <=3.5 GB address space, per-job timeouts.
source "$(dirname "$0")/env.sh"
cd "$S"
LOG=$S/out/run_all.log
run() { # timeout args...
  local t=$1; shift
  echo "$(date +%T) START $*" >> $LOG
  timeout -k 10 $t taskset -c 3 prlimit --as=3500000000 -- $PY $S/bench/flatbench.py "$@" >> $LOG 2>&1
  echo "$(date +%T) END rc=$? $*" >> $LOG
}
t0_of() { # scenario -> step with max on-map agents in its b timing run
  $PY -c "
import json,sys
d=json.load(open('$S/out/$1_b_timing.json'))
s=[x for x in d['reps'][-1]['windows']['early']['steps'] if 'n_on_map' in x]
print(max(s,key=lambda x:x['n_on_map'])['t'])"
}
# --- S
for c in b d a c; do run 300 S $c timing --reps 5; done
# --- M
run 900 M b timing --reps 5
run 600 M b timing --reps 2 --seed 2 --tag _seed2
run 600 M b timing --reps 2 --seed 3 --tag _seed3
T0=$(t0_of M); echo "M T0=$T0" >> $LOG
run 900 M d timing --reps 3
run 900 M a timing --reps 3 --early 50 --late 50 --t0 $T0
run 900 M c timing --reps 3 --early 30 --late 30 --t0 $T0
# --- cProfile S/M
for c in a b c d; do run 300 S $c cprofile; done
run 600 M b cprofile
run 600 M d cprofile --early 200
run 600 M a cprofile --early 20 --late 20 --t0 $T0
run 600 M c cprofile --early 15 --late 15 --t0 $T0
# --- L (fallbacks if L fails)
LS=L
run 900 L b timing
if ! grep -q "wrote $S/out/L_b_timing.json" $LOG; then LS=L2; run 900 L2 b timing; fi
if ! grep -q "wrote $S/out/${LS}_b_timing.json" $LOG; then LS=L3; run 900 L3 b timing; fi
T0L=$(t0_of $LS); echo "$LS T0=$T0L" >> $LOG
run 900 $LS a timing --early 10 --late 10 --t0 $T0L
run 1200 $LS c timing --early 10 --late 10 --t0 $T0L
run 900 $LS b cprofile --early 200
run 1200 $LS c cprofile --early 5
# --- pass 2 drift check
run 900 M b timing --reps 3 --tag _pass2
echo "$(date +%T) ALL DONE" >> $LOG
```

</details>

