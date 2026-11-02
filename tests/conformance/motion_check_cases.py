"""
Random inputs of `MotionCheck` and the Python resolution of their conflicts, as plain lists and dicts (see
`motion_check_data`), that Flatland.jl's tests compare against.
"""
import numpy as np

from flatland.envs.agent_chains import MotionCheck


def random_inputs(rng, max_agents, max_resources):
    """
    Random agents on a few resources numbered from 1, as `[held, wanted]` pairs with `None` for none. As in the env,
    each resource is held by at most one agent. Few resources for many agents make swaps, chains and same-target
    conflicts frequent.
    """
    n_agents = int(rng.integers(1, max_agents + 1))
    resources = [int(r) for r in rng.integers(1, max_resources + 1, size=int(rng.integers(2, max_resources + 1)))]
    occupied = set()
    inputs = []
    for _ in range(n_agents):
        held = None
        if rng.random() < 0.8:
            held = resources[int(rng.integers(len(resources)))]
            if held in occupied:
                held = None
            else:
                occupied.add(held)
        u = rng.random()
        wanted = held if u < 0.3 else (None if u < 0.4 else resources[int(rng.integers(len(resources)))])
        inputs.append([held, wanted])
    return inputs


def resolve(inputs):
    """
    Python's resolution of `inputs`, with agents numbered from 0 in the order of `inputs`: whether each agent may move,
    and the stopped and deadlocked agents.
    """
    mc = MotionCheck()
    for i, (held, wanted) in enumerate(inputs):
        mc.add_agent(i, held, wanted)
    mc.find_conflicts()
    return {
        "inputs": inputs,
        "check_resource": [bool(mc.check_resource(i)) for i in range(len(inputs))],
        "stopped": sorted(int(i) for i in mc.stopped),
        "deadlocked": sorted(int(i) for i in mc.deadlocked),
    }


def motion_check_data(n, seed, max_agents, max_resources):
    """The resolution of `n` random inputs of `random_inputs`, from the generator seeded with `seed`."""
    rng = np.random.default_rng(seed)
    return [resolve(random_inputs(rng, max_agents=max_agents, max_resources=max_resources)) for _ in range(n)]
