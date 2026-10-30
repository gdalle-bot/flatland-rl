# Profiling scripts of `REVAMP.md`

The measurements in `REVAMP.md` §2 come from these scripts, run on commit `51f5621c` with the pure-Python sources
(CPython 3.13, no Cython build), one pinned core, and seeded random actions.

| script | what it does |
|---|---|
| `flatbench.py` | builds an env for a scenario (S, M, L) and configuration (a–d, see `REVAMP.md` §2), times `reset()` and every `step()` phase with timestamps around the calls that bracket them, or records cProfile `.prof` files |
| `run_all.sh` | runs all `flatbench.py` jobs of §2 one after the other, with timeouts (about 8 minutes on one core) |
| `micro.py` | the micro-benchmarks of §2.4. Some target code that later changes removed (e.g. the tree observation's conflict check), so they reproduce on `51f5621c` only |
| `analyze.py` | turns the outputs into the Markdown tables of §2 |

From the repository root:

```bash
export PYTHONPATH=$PWD
WRAP="taskset -c 3 prlimit --as=3500000000 --" benchmarks/revamp/run_all.sh revamp_out
python benchmarks/revamp/micro.py 1240 revamp_out   # 1240: step of M with the most agents on the map
python benchmarks/revamp/analyze.py revamp_out
```

Scenario L needs about 2 GB of memory with the default observation builder.
