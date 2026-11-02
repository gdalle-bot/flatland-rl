using Flatland: MotionCheck, add_agent!, check_resource, find_conflicts!, reset!
using JET: JET
using Test

@isdefined(julia_data) || include("python.jl")

const MOTION_CHECK_CASES = python_module("tests.conformance.motion_check_cases")

# Python numbers agents from 0, Flatland.jl from 1
function resolve!(mc::MotionCheck, inputs)
    reset!(mc)
    for (i, (held, wanted)) in enumerate(inputs)
        add_agent!(mc, i, something(held, 0), something(wanted, 0))
    end
    find_conflicts!(mc)
    n = length(inputs)
    return Dict{String, Any}(
        "inputs" => inputs,
        "check_resource" => [check_resource(mc, i) for i in 1:n],
        "stopped" => [i - 1 for i in 1:n if mc.stopped[i]],
        "deadlocked" => [i - 1 for i in 1:n if mc.deadlocked[i]],
    )
end

function num_resources_of(records)
    return maximum(something(r, 0) for record in records for pair in record["inputs"] for r in pair; init = 1)
end

@testset "Random configurations match Python (up to $max_agents agents on $max_resources resources)" for (n, max_agents, max_resources) in
    [(3000, 12, 8), (1000, 40, 20), (200, 200, 40)]
    records = julia_data(MOTION_CHECK_CASES.motion_check_data(n, 0, max_agents, max_resources))
    @test length(records) == n
    # one fresh MotionCheck per configuration, and one reused for all of them
    fresh = [resolve!(MotionCheck(num_resources_of([r]), length(r["inputs"])), r["inputs"]) for r in records]
    reused_mc = MotionCheck(num_resources_of(records), max_agents)
    reused = [resolve!(reused_mc, r["inputs"]) for r in records]
    @test first(filter(((r, j),) -> r != j, collect(zip(records, fresh))), 1) == []
    @test reused == fresh
    # the configurations cover swaps, stopped agents, and agents that move
    @test count(r -> !isempty(r["deadlocked"]), records) >= n ÷ 50
    @test count(r -> length(r["stopped"]) >= 3, records) >= n ÷ 10
    @test count(r -> any(r["check_resource"]), records) >= n ÷ 2
end

@testset "Chains, swaps and contested cells" begin
    # the scenario of `test_agent_chains_new` in tests/test_agent_chains.py: cell (r, c) is resource 20r + c + 1
    cells = [
        # stopped chain
        ((1, 0), (1, 1)), ((1, 1), (1, 2)), ((1, 2), (1, 3)), ((1, 3), (1, 3)),
        # running chain
        ((2, 0), (2, 1)), ((2, 1), (2, 2)), ((2, 2), (2, 3)), ((2, 3), (2, 4)),
        # stopped short chain
        ((3, 0), (3, 1)), ((3, 1), (3, 1)),
        # swap
        ((4, 1), (4, 2)), ((4, 2), (4, 1)),
        # mid-chain stop
        ((5, 1), (5, 2)), ((5, 2), (5, 3)), ((5, 3), (5, 2)), ((6, 1), (6, 2)), ((6, 2), (6, 3)), ((6, 3), (6, 4)),
        ((6, 4), (6, 4)), ((6, 5), (6, 6)), ((6, 6), (6, 7)),
        # mid-chain swap
        ((7, 1), (7, 2)), ((7, 2), (7, 3)), ((7, 3), (7, 4)), ((7, 4), (7, 3)), ((7, 5), (7, 4)), ((7, 6), (7, 5)),
        # land on same
        ((8, 1), (8, 2)), ((8, 3), (8, 2)),
        # chains onto same
        ((9, 1), (9, 2)), ((9, 2), (9, 3)), ((9, 3), (9, 4)), ((9, 5), (9, 4)), ((9, 6), (9, 5)), ((9, 7), (9, 6)),
        # 3-way same
        ((10, 1), (10, 2)), ((10, 3), (10, 2)), ((11, 2), (10, 2)),
        # tee
        ((12, 1), (12, 2)), ((12, 2), (12, 3)), ((12, 3), (12, 4)), ((13, 3), (12, 3)),
        # tree
        ((14, 1), (14, 2)), ((14, 2), (14, 3)), ((14, 3), (14, 4)), ((15, 3), (14, 3)), ((15, 2), (15, 3)),
        ((16, 2), (15, 3)), ((18, 3), (17, 3)), ((18, 2), (18, 3)), ((19, 2), (18, 3)), ((17, 1), (17, 2)),
        ((17, 2), (17, 3)), ((17, 3), (17, 4)),
    ]
    not_stopped = Bool[
        0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0,
        1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1,
    ]
    resource((r, c)) = 20 * r + c + 1
    mc = MotionCheck(20 * 20, length(cells))
    for (i, (held, wanted)) in enumerate(cells)
        add_agent!(mc, i, resource(held), resource(wanted))
    end
    find_conflicts!(mc)
    @test .!mc.stopped == not_stopped
    @test findall(mc.hold) == [4, 10, 19]
    @test !any(mc.no_resource)
end

@testset "Many agents asking for one free cell" begin
    mc = MotionCheck(201, 200)
    for i in 1:200
        add_agent!(mc, i, i + 1, 1)
    end
    find_conflicts!(mc)
    @test findall(i -> check_resource(mc, i), 1:200) == [1]
    @test !any(mc.deadlocked)
end

@testset "Off-map agents and agents asking for nothing" begin
    # agents 1 and 2 enter the same cell from off the map: the smaller one wins; agent 3 stays off the map; agent 4
    # leaves the map (asks for nothing)
    mc = MotionCheck(4, 4)
    add_agent!(mc, 1, 0, 2)
    add_agent!(mc, 2, 0, 2)
    add_agent!(mc, 3, 0, 0)
    add_agent!(mc, 4, 3, 0)
    find_conflicts!(mc)
    @test [check_resource(mc, i) for i in 1:4] == [true, false, true, true]
    @test mc.hold == [false, false, true, false]
    @test mc.no_resource == [false, false, true, true]
end

@testset "Invalid inputs" begin
    mc = MotionCheck(3, 2)
    @test_throws ArgumentError add_agent!(mc, 3, 1, 1)
    @test_throws ArgumentError add_agent!(mc, 1, 4, 1)
    @test_throws ArgumentError add_agent!(mc, 1, 1, -1)
    add_agent!(mc, 1, 1, 2)
    @test_throws ArgumentError add_agent!(mc, 1, 2, 2)
    # two agents holding the same resource
    @test_throws ArgumentError add_agent!(mc, 2, 1, 3)
    reset!(mc)
    add_agent!(mc, 2, 1, 3)
    @test_throws ArgumentError MotionCheck(-1, 2)
end

function allocations_of_resolution(mc, inputs)
    resolve!(mc, inputs)
    return @allocated begin
        reset!(mc)
        for (i, (held, wanted)) in enumerate(inputs)
            add_agent!(mc, i, held, wanted)
        end
        find_conflicts!(mc)
    end
end

@testset "No allocations, type stable" begin
    inputs = [(2, 3), (3, 2), (4, 3), (5, 5), (0, 5), (6, 7), (7, 8), (8, 6)]
    mc = MotionCheck(8, length(inputs))
    @test allocations_of_resolution(mc, inputs) == 0
    JET.@test_opt find_conflicts!(mc)
    JET.@test_call find_conflicts!(mc)
    JET.@test_opt add_agent!(mc, 1, 2, 3)
end
