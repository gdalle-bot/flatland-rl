"""
    Flatland

Julia implementation of the [Flatland](https://github.com/flatland-association/flatland-rl) railway simulator.

The Python package `flatland-rl` is the reference implementation: this package aims at reproducing its trajectories
exactly, and at simulating them much faster. See `REVAMP.md` at the root of the repository for the plan.
"""
module Flatland

include("actions.jl")
include("grid.jl")
include("network.jl")

export RailEnvAction, DO_NOTHING, MOVE_LEFT, MOVE_FORWARD, MOVE_RIGHT, STOP_MOVING
export GridRail, is_valid_entry_point, check_action, apply_action
export EntryPointNetwork, entry_point_index, entry_point, num_entry_points, num_resources, next_entry_point, successors

end
