"""
    SpeedProfile(max_speed::Rational, acceleration_delta::Rational, braking_delta::Rational)

The speed parameters of one agent, in integer ticks. A cell is `ticks_per_cell` ticks long: the least common multiple
of the denominators of the three arguments. Every speed and distance the agent can reach is then a whole number of
ticks, so the rational arithmetic of the Python package (`fractions.Fraction`) becomes integer arithmetic.

`acceleration_delta` and `braking_delta` are the env's parameters of the same names; `braking_delta` is
non-positive in practice (`-1` by default).

# Fields

- `ticks_per_cell::Int`
- `max_speed::Int`
- `acceleration::Int`: `acceleration_delta` in ticks
- `braking::Int`: `braking_delta` in ticks
"""
struct SpeedProfile
    ticks_per_cell::Int
    max_speed::Int
    acceleration::Int
    braking::Int
end

function SpeedProfile(max_speed::Rational, acceleration_delta::Rational, braking_delta::Rational)
    max_speed <= 1 || throw(ArgumentError("max_speed must be at most 1 cell per step, got $max_speed"))
    n = lcm(denominator(max_speed), denominator(acceleration_delta), denominator(braking_delta))
    return SpeedProfile(n, to_ticks(n, max_speed), to_ticks(n, acceleration_delta), to_ticks(n, braking_delta))
end

"""
    to_ticks(profile::SpeedProfile, x::Rational)

The speed or distance `x` (in cells) as a whole number of ticks of `profile`, `nothing` for `nothing` (off the map).
Throws an `InexactError` if `x` is not a multiple of a tick.
"""
to_ticks(profile::SpeedProfile, x) = to_ticks(profile.ticks_per_cell, x)

to_ticks(::Integer, ::Nothing) = nothing
function to_ticks(ticks_per_cell::Integer, x::Rational)
    t = x * ticks_per_cell
    isinteger(t) || throw(InexactError(:to_ticks, Int, x))
    return Int(t)
end

"""
    from_ticks(profile::SpeedProfile, t)

The speed or distance of `t` ticks of `profile`, in cells, as a `Rational{Int}`; `nothing` for `nothing`.
"""
from_ticks(profile::SpeedProfile, t::Integer) = t // profile.ticks_per_cell
from_ticks(::SpeedProfile, ::Nothing) = nothing

# The formulas below are those of `flatland.envs.step_utils.speed_counter` in ticks, for agents on the map.

cap_speed(p::SpeedProfile, speed::Int) = max(0, min(p.max_speed, speed))

speed_after_acceleration(p::SpeedProfile, speed::Int) = min(speed + p.acceleration, p.max_speed)

speed_after_braking(p::SpeedProfile, speed::Int) = max(speed + p.braking, 0)

distance_after_crossing(p::SpeedProfile, distance::Int, speed::Int) = mod(distance + speed, p.ticks_per_cell)

distance_without_crossing(p::SpeedProfile, distance::Int, speed::Int) = min(distance + speed, p.ticks_per_cell)

"""
    is_cell_exit(profile, speed, distance)

Whether an agent on the map, at `speed` and `distance` in its cell, crosses into the next cell in the coming step.
"""
function is_cell_exit(p::SpeedProfile, speed::Int, distance::Int)
    speed = cap_speed(p, speed)
    return speed != 0 && distance + speed >= p.ticks_per_cell
end
