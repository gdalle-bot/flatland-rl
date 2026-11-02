const MT_N = 624
const MT_M = 397

"""
    NumpyRandomState(seed::Integer)
    NumpyRandomState(seed::AbstractVector{<:Integer})
    NumpyRandomState(key::AbstractVector{<:Integer}, pos::Integer)

The random number generator of the Python package: numpy's legacy `RandomState`, a 32-bit Mersenne Twister (MT19937),
with numpy's algorithms for [`random_sample`](@ref) and [`legacy_randint`](@ref). The same state draws the same
numbers as in Python.

The constructors match `RandomState(seed)` for an integer seed in `0:2^32-1` or a vector of such integers, and
`RandomState().set_state(("MT19937", key, pos))` for the 624 words `key` and the position `pos` of
`RandomState.get_state()`. The Gaussian cache of that state is not represented: the Python package draws no Gaussian
numbers during an episode.
"""
mutable struct NumpyRandomState
    key::Vector{UInt32}
    pos::Int
    function NumpyRandomState(key::AbstractVector{<:Integer}, pos::Integer)
        length(key) == MT_N || throw(ArgumentError("the key must have $MT_N words, got $(length(key))"))
        0 <= pos <= MT_N || throw(ArgumentError("the position must be in 0:$MT_N, got $pos"))
        all(k -> 0 <= k <= typemax(UInt32), key) || throw(ArgumentError("the key words must be in 0:2^32-1"))
        return new(UInt32.(key), Int(pos))
    end
end

function NumpyRandomState(seed::Integer)
    0 <= seed <= typemax(UInt32) || throw(ArgumentError("the seed must be in 0:2^32-1, got $seed"))
    return NumpyRandomState(init_genrand(UInt32(seed)), MT_N)
end

function NumpyRandomState(seed::AbstractVector{<:Integer})
    isempty(seed) && throw(ArgumentError("the seed must be non-empty"))
    all(s -> 0 <= s <= typemax(UInt32), seed) || throw(ArgumentError("the seed words must be in 0:2^32-1"))
    return NumpyRandomState(init_by_array(UInt32.(seed)), MT_N)
end

function init_genrand(seed::UInt32)
    key = Vector{UInt32}(undef, MT_N)
    key[1] = seed
    for i in 2:MT_N
        key[i] = 0x6c078965 * (key[i - 1] ⊻ (key[i - 1] >> 30)) + UInt32(i - 1)
    end
    return key
end

function init_by_array(init_key::Vector{UInt32})
    mt = init_genrand(0x012bd6aa)  # 19650218
    n = length(init_key)
    # 0-based indices i and j, as in the reference implementation
    i, j = 1, 0
    for _ in 1:max(MT_N, n)
        mt[i + 1] = (mt[i + 1] ⊻ ((mt[i] ⊻ (mt[i] >> 30)) * 0x0019660d)) + init_key[j + 1] + UInt32(j)
        i += 1
        j += 1
        if i >= MT_N
            mt[1] = mt[MT_N]
            i = 1
        end
        j >= n && (j = 0)
    end
    for _ in 1:(MT_N - 1)
        mt[i + 1] = (mt[i + 1] ⊻ ((mt[i] ⊻ (mt[i] >> 30)) * 0x5d588b65)) - UInt32(i)
        i += 1
        if i >= MT_N
            mt[1] = mt[MT_N]
            i = 1
        end
    end
    mt[1] = 0x80000000
    return mt
end

function generate!(rs::NumpyRandomState)
    key = rs.key
    upper, lower, matrix = 0x80000000, 0x7fffffff, 0x9908b0df
    @inbounds for i in 1:MT_N
        y = (key[i] & upper) | (key[mod1(i + 1, MT_N)] & lower)
        key[i] = key[mod1(i + MT_M, MT_N)] ⊻ (y >> 1) ⊻ (iszero(y & 0x00000001) ? 0x00000000 : matrix)
    end
    rs.pos = 0
    return rs
end

"""
    next_uint32!(rs::NumpyRandomState)

The next 32-bit output of the Mersenne Twister.
"""
function next_uint32!(rs::NumpyRandomState)
    rs.pos == MT_N && generate!(rs)
    rs.pos += 1
    y = @inbounds rs.key[rs.pos]
    y ⊻= y >> 11
    y ⊻= (y << 7) & 0x9d2c5680
    y ⊻= (y << 15) & 0xefc60000
    y ⊻= y >> 18
    return y
end

"""
    random_sample(rs::NumpyRandomState)

A float uniformly distributed in `[0, 1)` with 53 random bits, as numpy's `RandomState.random_sample()` (and each
number of `RandomState.rand(n)`): it consumes two 32-bit outputs.
"""
function random_sample(rs::NumpyRandomState)
    a = next_uint32!(rs) >> 5
    b = next_uint32!(rs) >> 6
    return (a * 67108864.0 + b) / 9007199254740992.0
end

"""
    legacy_randint(rs::NumpyRandomState, low, high)

An integer uniformly distributed in `low:high-1`, as numpy's `RandomState.randint(low, high)`: masked rejection
sampling on 32-bit outputs (on 64-bit ones beyond 32-bit ranges). It consumes no output if `high == low + 1`.
"""
function legacy_randint(rs::NumpyRandomState, low::Integer, high::Integer)
    low < high || throw(ArgumentError("low must be smaller than high, got $low and $high"))
    range = UInt64(high - 1 - low)
    offset = Int64(low)
    range == 0 && return offset
    return if range <= typemax(UInt32)
        range == typemax(UInt32) && return offset + Int64(next_uint32!(rs))
        mask = UInt32(bit_mask(range))
        while true
            value = next_uint32!(rs) & mask
            value <= range && return offset + Int64(value)
        end
    else
        mask = bit_mask(range)
        while true
            value = ((UInt64(next_uint32!(rs)) << 32) | next_uint32!(rs)) & mask
            value <= range && return offset + reinterpret(Int64, value)
        end
    end
end

# the smallest 2^k - 1 at least `range`
bit_mask(range::UInt64) = typemax(UInt64) >> leading_zeros(range)
