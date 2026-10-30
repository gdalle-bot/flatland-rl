using Flatland
using Flatland: cap_speed, distance_after_crossing, distance_without_crossing, is_cell_exit
using JET: JET
using JSON: JSON
using Test

# exported from the Python package by tests/conformance/export_candidates.py
const CANDIDATES = JSON.parsefile(joinpath(@__DIR__, "data", "candidates.json"))

const TABLE_1_ROWS = [
    "Done", "Target reached", "Malfunction", "Map entry", "Stay off map", "On-map cell transition",
    "Invalid action at cell exit", "Stopped", "Acceleration or start moving", "Braking", "Default",
]

python_rational(x) = isnothing(x) ? nothing : x[1] // x[2]

function python_case(config, record)
    r = Dict(zip(CANDIDATES["fields"], record))
    profile = SpeedProfile(
        python_rational(r["max_speed"]),
        python_rational(config["acceleration_delta"]),
        python_rational(config["braking_delta"]),
    )
    agent = PreStepAgent(;
        current_entry_point = r["current_entry_point"],
        next_entry_point = r["next_entry_point"],
        initial_entry_point = r["initial_entry_point"],
        done = r["done"],
        in_malfunction = r["in_malfunction"],
        # None (off the map) becomes 0, which is ignored off the map
        speed = something(to_ticks(profile, python_rational(r["speed"])), 0),
        distance = something(to_ticks(profile, python_rational(r["distance"])), 0),
        earliest_departure = r["earliest_departure"],
    )
    kwargs = (;
        independent = r["independent"],
        next_is_target = r["next_is_target"],
        elapsed_steps = r["elapsed_steps"],
        remove_agents_at_target = config["remove_agents_at_target"],
    )
    expected = (
        r["candidate_entry_point"],
        r["candidate_next_entry_point"],
        python_rational(r["candidate_speed"]),
        python_rational(r["candidate_distance"]),
    )
    return (; row = r["row"], profile, agent, action = RailEnvAction(r["action"]), kwargs, expected)
end

const CASES = [python_case(config, record) for config in CANDIDATES["configs"] for record in config["records"]]

@testset "SpeedProfile" begin
    p = SpeedProfile(1 // 3, 1 // 2, -1 // 4)
    @test (p.ticks_per_cell, p.max_speed, p.acceleration, p.braking) == (12, 4, 6, -3)
    @test to_ticks(p, 3 // 4) == 9
    @test from_ticks(p, 9) == 3 // 4
    @test isnothing(to_ticks(p, nothing)) && isnothing(from_ticks(p, nothing))
    @test_throws InexactError to_ticks(p, 1 // 5)
    @test_throws ArgumentError SpeedProfile(3 // 2, 1 // 1, -1 // 1)
    @test cap_speed(p, 7) == 4 && cap_speed(p, -1) == 0
    @test distance_after_crossing(p, 10, 4) == 2
    @test distance_without_crossing(p, 10, 4) == 12
    @test is_cell_exit(p, 4, 8) && !is_cell_exit(p, 4, 7) && !is_cell_exit(p, 0, 12)
end

@testset "Candidates match Python's collect phase" begin
    @test Set(c.row for c in CASES) == Set(TABLE_1_ROWS)
    @testset "$row" for row in TABLE_1_ROWS
        cases = filter(c -> c.row == row, CASES)
        mismatches = filter(cases) do c
            result = candidate(c.agent, c.profile, c.action; c.kwargs...)
            actual = (result.entry_point, result.next_entry_point, from_ticks(c.profile, result.speed), from_ticks(c.profile, result.distance))
            actual != c.expected
        end
        @test isempty(mismatches)
        @test length(cases) >= 3
    end
end

function allocations(c)
    candidate(c.agent, c.profile, c.action; c.kwargs...)
    return @allocated candidate(c.agent, c.profile, c.action; c.kwargs...)
end

@testset "No allocations, type stable" begin
    @test all(c -> allocations(c) == 0, CASES)
    c = first(CASES)
    JET.@test_opt candidate(c.agent, c.profile, c.action; c.kwargs...)
    JET.@test_call candidate(c.agent, c.profile, c.action; c.kwargs...)
end
