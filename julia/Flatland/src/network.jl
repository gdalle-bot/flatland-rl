"""
    EntryPointNetwork

All entry points of a rail as flat integer tables, built once per rail. Entry point `(r, c, d)` (0-based row, column
and heading) has index `entry_point_index(net, r, c, d) = 4 * (r * width + c) + d + 1`, and index `0` stands for "no
entry point".

# Fields

- `height`, `width`: size of the grid
- `valid::BitVector`: whether each entry point is on the rail
- `next::Matrix{Int32}`: `next[a, i]` is the entry point reached from `i` with action `a` (`Int(a) + 1`-th row), `0` if
  the action is invalid there or leads off the rail
- `resource::Vector{Int32}`: the resource each entry point occupies, the cell `r * width + c + 1`, except on level-free
  crossings, where each axis is a resource of its own: `height * width + 2 * (r * width + c) + d % 2 + 1`

# Constructor

    EntryPointNetwork(rail::GridRail)
"""
struct EntryPointNetwork
    height::Int
    width::Int
    valid::BitVector
    next::Matrix{Int32}
    resource::Vector{Int32}
end

"""
    entry_point_index(net, r, c, d)

Index of the entry point `(r, c, d)` (0-based row, column and heading) in the tables of `net`.
"""
function entry_point_index end

entry_point_index(height_width::Tuple{Int, Int}, r::Integer, c::Integer, d::Integer) =
    4 * (Int(r) * height_width[2] + Int(c)) + Int(d) + 1
entry_point_index(net::EntryPointNetwork, r::Integer, c::Integer, d::Integer) =
    entry_point_index((net.height, net.width), r, c, d)

"""
    entry_point(net, i)

The `(r, c, d)` entry point with index `i`.
"""
function entry_point(net::EntryPointNetwork, i::Integer)
    cell, d = divrem(Int(i) - 1, 4)
    r, c = divrem(cell, net.width)
    return (r, c, d)
end

num_entry_points(net::EntryPointNetwork) = length(net.valid)

function EntryPointNetwork(rail::GridRail)
    h, w = height(rail), width(rail)
    n = 4 * h * w
    valid = falses(n)
    next = zeros(Int32, length(ALL_ACTIONS), n)
    resource = zeros(Int32, n)
    for r in 0:(h - 1), c in 0:(w - 1), d in 0:3
        i = entry_point_index((h, w), r, c, d)
        cell = r * w + c
        resource[i] = (r, c) in rail.level_free ? h * w + 2 * cell + d % 2 + 1 : cell + 1
        is_valid_entry_point(rail, r, c, d) || continue
        valid[i] = true
        for action in ALL_ACTIONS
            target = apply_action(rail, action, r, c, d)
            if !isnothing(target)
                next[Int(action) + 1, i] = entry_point_index((h, w), target...)
            end
        end
    end
    return EntryPointNetwork(h, w, valid, next, resource)
end

"""
    next_entry_point(net, i, action)

Index of the entry point reached from entry point `i` with `action`, `0` if there is none.
"""
next_entry_point(net::EntryPointNetwork, i::Integer, action::RailEnvAction) = @inbounds net.next[Int(action) + 1, i]

"""
    successors(net, i)

Distinct entry points reachable from entry point `i` with `MOVE_LEFT`, `MOVE_FORWARD` and `MOVE_RIGHT`, in this order,
as `get_successor_entry_points` in Python.
"""
function successors(net::EntryPointNetwork, i::Integer)
    result = Int32[]
    for action in (MOVE_LEFT, MOVE_FORWARD, MOVE_RIGHT)
        j = next_entry_point(net, i, action)
        if j != 0 && !(j in result)
            push!(result, j)
        end
    end
    return result
end
