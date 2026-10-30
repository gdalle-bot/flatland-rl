#=
Rail grid and cell transitions.

Positions `(r, c)` are 0-based row and column and directions `d` are 0 (north), 1 (east), 2 (south), 3 (west), as in
Python. An entry point `(r, c, d)` is a cell together with the heading of an agent that entered it.
=#

# (row, column) offset when leaving a cell towards each direction, as `MOVEMENT_ARRAY` in Python
const MOVEMENT = ((-1, 0), (0, 1), (1, 0), (0, -1))

"""
    GridRail

Rail grid: `transitions[r + 1, c + 1]` is the 16-bit transition code of the cell in (0-based) row `r` and column `c`,
encoded as `flatland.core.grid.grid4.Grid4Transitions` in Python, and `level_free` contains the (0-based) positions of
level-free crossings, where the two crossing tracks are distinct resources.

# Constructor

    GridRail(transitions::AbstractMatrix, level_free=())

The constructor copies `transitions`.
"""
struct GridRail
    transitions::Matrix{UInt16}
    level_free::Set{Tuple{Int, Int}}
end

GridRail(transitions::AbstractMatrix, level_free = ()) =
    GridRail(Matrix{UInt16}(transitions), Set{Tuple{Int, Int}}(level_free))

height(rail::GridRail) = size(rail.transitions, 1)
width(rail::GridRail) = size(rail.transitions, 2)

in_bounds(rail::GridRail, r::Integer, c::Integer) = 0 <= r < height(rail) && 0 <= c < width(rail)
cell_code(rail::GridRail, r::Integer, c::Integer) = rail.transitions[r + 1, c + 1]

# 4-bit mask of the exit directions allowed when heading `orientation` in a cell: bit `3 - d` for exit direction `d`
transition_bits(code::UInt16, orientation::Integer) = (code >> (4 * (3 - orientation))) & 0x000f
has_transition(code::UInt16, orientation::Integer, direction::Integer) =
    isone((transition_bits(code, orientation) >> (3 - direction)) & 0x0001)
count_transitions(code::UInt16, orientation::Integer) = count_ones(transition_bits(code, orientation))

# first allowed exit direction, 3 if there is none (as `fast_argmax` in Python)
function first_transition(code::UInt16, orientation::Integer)
    for direction in 0:2
        has_transition(code, orientation, direction) && return direction
    end
    return 3
end

"""
    is_valid_entry_point(rail, r, c, d)

Whether `(r, c, d)` is in the grid and has at least one allowed exit, as `is_valid_entry_point` in Python.
"""
is_valid_entry_point(rail::GridRail, r::Integer, c::Integer, d::Integer) =
    in_bounds(rail, r, c) && count_transitions(cell_code(rail, r, c), d) > 0

"""
    check_action(action, code, direction)

Exit direction taken by `action` in a cell with transition code `code` entered heading `direction`, as
`RailGridTransitionMap._check_action_new` in Python. Returns `(new_direction, transition_valid, preprocessed_action,
action_valid)`:
- `MOVE_LEFT`/`MOVE_RIGHT` fall back to `MOVE_FORWARD` (with `transition_valid = false`) where the turn is impossible;
- any action is invalid (`action_valid = false`, `STOP_MOVING`) where neither it nor going forward is possible, e.g.
  `MOVE_FORWARD` facing a symmetric switch.

Throws an `ArgumentError` if `direction` is not in `0:3`.
"""
function check_action(action::RailEnvAction, code::UInt16, direction::Integer)
    0 <= direction <= 3 || throw(ArgumentError("direction must be in 0:3, got $direction"))
    direction = Int(direction)
    if count_transitions(code, direction) == 1
        # dead-end, straight or curved rail: the only exit
        new_direction = first_transition(code, direction)
        if action == MOVE_LEFT && new_direction != mod(direction - 1, 4)
            return new_direction, false, MOVE_FORWARD, true
        elseif action == MOVE_RIGHT && new_direction != mod(direction + 1, 4)
            return new_direction, false, MOVE_FORWARD, true
        end
        return new_direction, true, action, true
    end
    if action == MOVE_LEFT
        new_direction = mod(direction - 1, 4)
        if has_transition(code, direction, new_direction)
            return new_direction, true, MOVE_LEFT, true
        elseif has_transition(code, direction, direction)
            return direction, false, MOVE_FORWARD, true
        end
    elseif action == MOVE_RIGHT
        new_direction = mod(direction + 1, 4)
        if has_transition(code, direction, new_direction)
            return new_direction, true, MOVE_RIGHT, true
        elseif has_transition(code, direction, direction)
            return direction, false, MOVE_FORWARD, true
        end
    elseif has_transition(code, direction, direction)
        return direction, true, action, true
    end
    return direction, false, STOP_MOVING, false
end

"""
    apply_action(rail, action, r, c, d)

Entry point `(r′, c′, d′)` an agent at entry point `(r, c, d)` moves to with `action`, or `nothing` if the action is
invalid there or leads off the rail, as `RailGridTransitionMap.apply_action_independent` in Python. `DO_NOTHING` and
`STOP_MOVING` lead where `MOVE_FORWARD` does. Throws a `BoundsError` if `(r, c)` is outside the grid.
"""
function apply_action(rail::GridRail, action::RailEnvAction, r::Integer, c::Integer, d::Integer)
    new_direction, _, _, action_valid = check_action(action, cell_code(rail, r, c), d)
    dr, dc = MOVEMENT[new_direction + 1]
    new_r, new_c = Int(r) + dr, Int(c) + dc
    if action_valid && is_valid_entry_point(rail, new_r, new_c, new_direction)
        return (new_r, new_c, new_direction)
    else
        return nothing
    end
end
