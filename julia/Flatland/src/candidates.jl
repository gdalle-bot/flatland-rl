"""
    PreStepAgent(; current_entry_point, next_entry_point, initial_entry_point, done, in_malfunction, speed, distance,
                 earliest_departure)

The state of one agent at the start of a step, as read by the collect phase. Entry points are indices of an
[`EntryPointNetwork`](@ref), `0` for none; speeds and distances are in ticks of the agent's [`SpeedProfile`](@ref),
and are only meaningful while the agent is on the map (`None` in Python while it is off the map).

# Fields

- `current_entry_point::Int32`: `0` while off the map
- `next_entry_point::Int32`: the entry point the agent moves to when it leaves its cell, `0` while off the map
- `initial_entry_point::Int32`: where the agent enters the map
- `done::Bool`: whether the agent reached its target in an earlier step
- `in_malfunction::Bool`: as read by the collect phase, after this step's malfunction counter update and new
  malfunctions
- `speed::Int`, `distance::Int`: ignored while off the map
- `earliest_departure::Int`: the first step at which the agent may enter the map
"""
Base.@kwdef struct PreStepAgent
    current_entry_point::Int32
    next_entry_point::Int32
    initial_entry_point::Int32
    done::Bool
    in_malfunction::Bool
    speed::Int
    distance::Int
    earliest_departure::Int
end

"""
    Candidate

What an agent asks for in a step, before conflicts with the other agents are resolved: the optimistic candidate of
the collect phase.

# Fields

- `entry_point::Int32`, `next_entry_point::Int32`: `0` for none (the agent stays off the map)
- `speed::Int`: in ticks
- `distance::Union{Nothing, Int}`: in ticks, `nothing` if the agent stays off or leaves the map
"""
struct Candidate
    entry_point::Int32
    next_entry_point::Int32
    speed::Int
    distance::Union{Nothing, Int}
end

is_moving_action(action::RailEnvAction) = action in (MOVE_LEFT, MOVE_FORWARD, MOVE_RIGHT)
is_left_right_action(action::RailEnvAction) = action in (MOVE_LEFT, MOVE_RIGHT)

"""
    candidate(agent::PreStepAgent, profile::SpeedProfile, action::RailEnvAction;
              independent, next_is_target, elapsed_steps, remove_agents_at_target)

The [`Candidate`](@ref) of `agent` for `action`, as computed by the collect phase of `RailEnv.step()` in Python.

# Keyword arguments

- `independent::Integer`: the entry point that `action` leads to from `agent.next_entry_point`, or from
  `agent.initial_entry_point` off the map, ignoring other agents ([`next_entry_point`](@ref)); `0` if `action` is
  invalid there
- `next_is_target::Bool`: whether `agent.next_entry_point` is one of the agent's targets
- `elapsed_steps::Integer`: the env's step counter as the collect phase reads it, already incremented for this step
  (`1` in the first step)
- `remove_agents_at_target::Bool`: the env parameter of the same name

The three parts of the candidate follow the case analysis of `design_by_contract.md` (Table 1), one function each.
"""
function candidate(
        agent::PreStepAgent, profile::SpeedProfile, action::RailEnvAction;
        independent::Integer, next_is_target::Bool, elapsed_steps::Integer, remove_agents_at_target::Bool,
    )
    (; done, in_malfunction, speed, distance) = agent
    off_map = agent.current_entry_point == 0
    transition_invalid = independent == 0
    if off_map
        cell_exit = true
        target_reached = false
        action_invalid_on_rail = false
        invalid_action_at_cell_exit = false
    else
        cell_exit = is_cell_exit(profile, speed, distance)
        target_reached = cell_exit && next_is_target
        action_invalid_on_rail = transition_invalid
        invalid_action_at_cell_exit = action_invalid_on_rail && cell_exit
    end
    stopped = !off_map && speed == 0
    entry_point_candidates = candidate_entry_points(
        agent, action; independent, elapsed_steps, off_map, cell_exit, target_reached, transition_invalid,
        action_invalid_on_rail, invalid_action_at_cell_exit,
    )
    stay_off_map = entry_point_candidates[1] == 0
    flags = (; done, in_malfunction, off_map, target_reached, invalid_action_at_cell_exit, stopped, stay_off_map)
    return Candidate(
        entry_point_candidates...,
        candidate_speed(profile, speed, action; flags...),
        candidate_distance(profile, speed, distance; remove_agents_at_target, flags...),
    )
end

function candidate_entry_points(
        agent::PreStepAgent, action::RailEnvAction;
        independent, elapsed_steps, off_map, cell_exit, target_reached, transition_invalid, action_invalid_on_rail,
        invalid_action_at_cell_exit,
    )
    (; current_entry_point, next_entry_point, initial_entry_point, done, in_malfunction) = agent
    unchanged = (current_entry_point, next_entry_point)
    movement_action_given = is_moving_action(action)
    ready_to_depart = !done && agent.earliest_departure <= elapsed_steps
    required_action_invalid_or_not_required_or_no_movement =
        transition_invalid || !movement_action_given || !ready_to_depart
    map_entry = off_map && !required_action_invalid_or_not_required_or_no_movement
    off_map_no_departure = off_map && required_action_invalid_or_not_required_or_no_movement
    on_map_cell_transition = !off_map && cell_exit && !action_invalid_on_rail

    # done
    done && return unchanged
    # stay off map
    off_map_no_departure && !done && !in_malfunction && return unchanged
    # keep moving mid-cell
    !done && !in_malfunction && !off_map && !cell_exit && return unchanged
    # on-map cell transition
    if on_map_cell_transition && !done && !in_malfunction && !target_reached
        return (next_entry_point, Int32(independent))
    end
    # malfunction
    in_malfunction && !done && !target_reached && return unchanged
    # map entry
    map_entry && !done && !in_malfunction && return (initial_entry_point, Int32(independent))
    # target reached
    target_reached && !done && return (next_entry_point, Int32(independent))
    # invalid action at cell exit
    invalid_action_at_cell_exit && !done && !in_malfunction && !target_reached && return unchanged
    return error("no candidate_entry_points branch matched")
end

function candidate_speed(
        p::SpeedProfile, speed::Int, action::RailEnvAction;
        done, in_malfunction, off_map, target_reached, invalid_action_at_cell_exit, stopped, stay_off_map,
    )
    no_earlier_case_applies = !(done || target_reached) && !in_malfunction && !off_map && !invalid_action_at_cell_exit
    # done
    done && return 0
    # stay off map
    off_map && stay_off_map && !done && !target_reached && !in_malfunction && return 0
    # keep moving mid-cell
    if no_earlier_case_applies && (action == DO_NOTHING || (!stopped && is_left_right_action(action)))
        return speed
    end
    # acceleration or start moving
    if (action == MOVE_FORWARD || (stopped && is_moving_action(action))) && no_earlier_case_applies
        return speed_after_acceleration(p, speed)
    end
    # braking
    action == STOP_MOVING && no_earlier_case_applies && return speed_after_braking(p, speed)
    # malfunction
    in_malfunction && !done && !target_reached && return 0
    # map entry
    off_map && !stay_off_map && !done && !target_reached && !in_malfunction && return cap_speed(p, p.acceleration)
    # target reached
    target_reached && !done && return 0
    # invalid action at cell exit
    invalid_action_at_cell_exit && !done && !target_reached && !in_malfunction && !off_map && return 0
    return error("no candidate_speed branch matched")
end

function candidate_distance(
        p::SpeedProfile, speed::Int, distance::Int;
        done, in_malfunction, remove_agents_at_target, off_map, target_reached, invalid_action_at_cell_exit, stopped,
        stay_off_map,
    )
    on_map_no_earlier_case = !done && !target_reached && !off_map && !in_malfunction && !invalid_action_at_cell_exit
    # done (a done agent removed at its target is off the map)
    done && return off_map ? nothing : distance_without_crossing(p, distance, speed)
    # stay off map
    off_map && stay_off_map && !done && !target_reached && !in_malfunction && return nothing
    # keep moving mid-cell
    !stopped && on_map_no_earlier_case && return distance_after_crossing(p, distance, speed)
    # stopped
    stopped && on_map_no_earlier_case && return distance
    # malfunction (on or off the map)
    in_malfunction && !done && !target_reached && return off_map ? nothing : distance
    # map entry
    off_map && !stay_off_map && !done && !target_reached && !in_malfunction && return 0
    # target reached
    if target_reached && !done
        return remove_agents_at_target ? nothing : distance_without_crossing(p, distance, speed)
    end
    # invalid action at cell exit
    if invalid_action_at_cell_exit && !done && !target_reached && !in_malfunction && !off_map
        return distance_without_crossing(p, distance, speed)
    end
    return error("no candidate_distance branch matched")
end
