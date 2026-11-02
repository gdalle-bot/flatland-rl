using Flatland
using Flatland: NUM_CACHED_DRAWS, generate_malfunction!, legacy_randint, next_uint32!, random_sample
using JET: JET
using Test

@isdefined(julia_data) || include("python.jl")

const RNG_CASES = python_module("tests.conformance.rng_cases")

julia_state(state) = NumpyRandomState(state["key"], state["pos"])
same_state(rs::NumpyRandomState, state) = rs.key == state["key"] && rs.pos == state["pos"]

@testset "NumpyRandomState draws as numpy's RandomState" begin
    records = julia_data(RNG_CASES.random_state_data())
    @testset "seed $(record["seed"])" for record in records
        seed = record["seed"]
        if !isnothing(seed)
            # the seeding itself
            rs = seed isa Integer ? NumpyRandomState(seed) : NumpyRandomState(Int.(seed))
            @test same_state(rs, record["initial"])
        end
        rs = julia_state(record["initial"])
        @test [random_sample(rs) for _ in record["samples"]] == record["samples"]
        @test [legacy_randint(rs, low, high) for (low, high, _) in record["randints"]] == [r[3] for r in record["randints"]]
        @test same_state(rs, record["final"])
    end
end

@testset "ParamMalfunctionGen draws as Python's" begin
    records = julia_data(RNG_CASES.malfunction_data())
    @testset "rate $(r["rate"]), durations $(r["min_duration"]):$(r["max_duration"])$(haskey(r, "broken") ? ", resumed" : "")" for r in records
        rs = julia_state(r["initial"])
        gen = if haskey(r, "broken")
            ParamMalfunctionGen(r["rate"], r["min_duration"], r["max_duration"], r["probability"], r["broken"], r["index"])
        else
            ParamMalfunctionGen(r["rate"], r["min_duration"], r["max_duration"])
        end
        # Julia's 1 - exp(-rate) is Python's probability for these rates
        @test gen.probability == r["probability"]
        @test [generate_malfunction!(gen, rs) for _ in r["durations"]] == r["durations"]
        @test same_state(rs, r["final"])
        @test gen.index == get(r, "final_index", r["index"])
        if r["rate"] > 0 && !haskey(r, "broken")
            # 3000 draws break down at least once
            @test any(>(0), r["durations"])
        end
    end
    @test any(r -> haskey(r, "final_index") && r["final_index"] < r["index"], records)
end

@testset "NoMalfunctionGen consumes the cached draws" begin
    rs = NumpyRandomState(1)
    gen = NoMalfunctionGen()
    @test generate_malfunction!(gen, rs) == 0
    reference = NumpyRandomState(1)
    for _ in 1:NUM_CACHED_DRAWS
        random_sample(reference)
    end
    @test rs.key == reference.key && rs.pos == reference.pos
    @test generate_malfunction!(gen, rs) == 0
    @test rs.key == reference.key && rs.pos == reference.pos
end

@testset "Invalid arguments" begin
    @test_throws ArgumentError NumpyRandomState(-1)
    @test_throws ArgumentError NumpyRandomState(2^32)
    @test_throws ArgumentError NumpyRandomState(Int[])
    @test_throws ArgumentError NumpyRandomState(zeros(Int, 623), 0)
    @test_throws ArgumentError NumpyRandomState(zeros(Int, 624), 625)
    @test_throws ArgumentError legacy_randint(NumpyRandomState(0), 3, 3)
    @test_throws ArgumentError ParamMalfunctionGen(0.1, 3, 2)
    @test_throws ArgumentError ParamMalfunctionGen(0.1, 1, 2, 1.5, BitVector(), 0)
    @test_throws ArgumentError ParamMalfunctionGen(0.1, 1, 2, 0.1, trues(3), 0)
    @test_throws ArgumentError ParamMalfunctionGen(0.1, 1, 2, 0.1, BitVector(), NUM_CACHED_DRAWS)
end

function allocations_of_draws(rs, gen)
    random_sample(rs)
    legacy_randint(rs, 0, 10)
    generate_malfunction!(gen, rs)
    return @allocated begin
        random_sample(rs)
        legacy_randint(rs, 0, 10)
        generate_malfunction!(gen, rs)
    end
end

@testset "No allocations, type stable" begin
    rs = NumpyRandomState(3)
    gen = ParamMalfunctionGen(0.5, 1, 4)
    @test allocations_of_draws(rs, gen) == 0
    JET.@test_opt random_sample(rs)
    JET.@test_opt legacy_randint(rs, 0, 10)
    JET.@test_opt generate_malfunction!(gen, rs)
    JET.@test_call generate_malfunction!(gen, rs)
    JET.@test_opt next_uint32!(rs)
end
