"""
    SpeedProfile(max_speed, acceleration_delta, braking_delta)

The speed parameters of one agent, in integer ticks. A cell is `ticks_per_cell` ticks long: the least common multiple
of the denominators of the three arguments. Every speed and distance the agent can reach is then a whole number of
ticks, so the rational arithmetic of the Python package (`fractions.Fraction`) becomes integer arithmetic.

`acceleration_delta` and `braking_delta` are the env's parameters of the same names; `braking_delta` is
non-positive in practice (`-1` by default). The arguments are `Integer`s or `Rational`s. Floats are not accepted,
since most decimal speeds (e.g. `0.33`) are not exact binary fractions: convert them first, as the Python package's
`_pseudo_fractional` does (`0.33` becomes `1//3`). Throws an `ArgumentError` unless `0 <= max_speed <= 1`.

# Fields

- `ticks_per_cell::Int`
- `max_speed::Int`, in `0:ticks_per_cell`
- `acceleration::Int`: `acceleration_delta` in ticks
- `braking::Int`: `braking_delta` in ticks
"""
struct SpeedProfile
    ticks_per_cell::Int
    max_speed::Int
    acceleration::Int
    braking::Int
    function SpeedProfile(ticks_per_cell::Integer, max_speed::Integer, acceleration::Integer, braking::Integer)
        ticks_per_cell > 0 || throw(ArgumentError("ticks_per_cell must be positive, got $ticks_per_cell"))
        0 <= max_speed <= ticks_per_cell ||
            throw(ArgumentError("max_speed must be between 0 and 1 cell per step, got $max_speed // $ticks_per_cell"))
        return new(ticks_per_cell, max_speed, acceleration, braking)
    end
end

# exact numbers of cells
const Exact = Union{Integer, Rational}

function SpeedProfile(max_speed::Exact, acceleration_delta::Exact, braking_delta::Exact)
    n = lcm(denominator(max_speed), denominator(acceleration_delta), denominator(braking_delta))
    return SpeedProfile(n, exact_ticks(n, max_speed), exact_ticks(n, acceleration_delta), exact_ticks(n, braking_delta))
end

"""
    to_ticks(profile::SpeedProfile, x)

The speed or distance `x` (in cells, an `Integer` or a `Rational`) as a whole number of ticks of `profile`, `nothing`
for `nothing` (off the map). Throws an `InexactError` if `x` is not a multiple of a tick.
"""
to_ticks(profile::SpeedProfile, x::Union{Exact, Nothing}) = exact_ticks(profile.ticks_per_cell, x)

exact_ticks(::Integer, ::Nothing) = nothing
function exact_ticks(ticks_per_cell::Integer, x::Exact)
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
