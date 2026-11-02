"""
Number of uniform draws that [`ParamMalfunctionGen`](@ref) caches, as `NBR_CHACHED_RAND` in Python.
"""
const NUM_CACHED_DRAWS = 1_000_000

"""
    ParamMalfunctionGen(rate, min_duration, max_duration)
    ParamMalfunctionGen(rate, min_duration, max_duration, probability, broken, index)

The malfunction generator of the Python package (`flatland.envs.malfunction_generators.ParamMalfunctionGen`). Each
call to [`generate_malfunction!`](@ref) draws whether an agent breaks down, with probability `1 - exp(-rate)`, and if
so for how many steps, uniformly in `min_duration + 1:max_duration + 1`.

As in Python, the first call draws [`NUM_CACHED_DRAWS`](@ref) uniform numbers from the random state, whatever the
rate (so a zero rate also consumes them), and caches which of them are below the probability; each call then moves
to the next cached draw, cyclically, and draws the duration from the random state only when the agent breaks down.
The cache and the position in it are kept for the lifetime of the generator.

The second constructor restores a generator from its state in Python: the `probability` (`_malfunction_prob`, which
may differ from Julia's `1 - exp(-rate)` by one ulp), the cached draws below it (`broken`, empty if nothing was drawn
yet) and the position `index` (`_rand_idx`).
"""
mutable struct ParamMalfunctionGen
    rate::Float64
    min_duration::Int
    max_duration::Int
    probability::Float64
    broken::BitVector
    index::Int
    function ParamMalfunctionGen(
            rate::Real, min_duration::Integer, max_duration::Integer, probability::Real, broken::AbstractVector{Bool},
            index::Integer,
        )
        min_duration <= max_duration ||
            throw(ArgumentError("min_duration must be at most max_duration, got $min_duration and $max_duration"))
        0 <= probability <= 1 || throw(ArgumentError("the probability must be in [0, 1], got $probability"))
        length(broken) in (0, NUM_CACHED_DRAWS) ||
            throw(ArgumentError("broken must be empty or have $NUM_CACHED_DRAWS draws, got $(length(broken))"))
        0 <= index < NUM_CACHED_DRAWS || throw(ArgumentError("index must be in 0:$(NUM_CACHED_DRAWS - 1), got $index"))
        return new(rate, min_duration, max_duration, probability, BitVector(broken), index)
    end
end

function ParamMalfunctionGen(rate::Real, min_duration::Integer, max_duration::Integer)
    probability = rate <= 0 ? 0.0 : 1 - exp(-Float64(rate))
    return ParamMalfunctionGen(rate, min_duration, max_duration, probability, BitVector(), 0)
end

"""
    NoMalfunctionGen()

The generator of an env without malfunctions, as in Python: a [`ParamMalfunctionGen`](@ref) of rate 0, which still
consumes [`NUM_CACHED_DRAWS`](@ref) uniform draws on its first use.
"""
NoMalfunctionGen() = ParamMalfunctionGen(0.0, 0, 0)

"""
    generate_malfunction!(gen::ParamMalfunctionGen, rs::NumpyRandomState)

The number of steps for which an agent breaks down, `0` if it does not, as `ParamMalfunctionGen.generate` in Python.
"""
function generate_malfunction!(gen::ParamMalfunctionGen, rs::NumpyRandomState)
    if isempty(gen.broken)
        resize!(gen.broken, NUM_CACHED_DRAWS)
        for k in 1:NUM_CACHED_DRAWS
            gen.broken[k] = random_sample(rs) < gen.probability
        end
    end
    gen.index += 1
    gen.index >= NUM_CACHED_DRAWS && (gen.index %= NUM_CACHED_DRAWS)
    # Python's index is 0-based
    @inbounds gen.broken[gen.index + 1] || return 0
    return legacy_randint(rs, gen.min_duration, gen.max_duration + 1) + 1
end
