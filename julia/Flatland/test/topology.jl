using Flatland
using Flatland: ALL_ACTIONS, MOVE_FORWARD, num_entry_points
using JET: JET
using JSON: JSON
using Test

# exported from the Python package by tests/conformance/export_topology.py
const TOPOLOGY = JSON.parsefile(joinpath(@__DIR__, "data", "topology.json"))

function python_resource(net, resource)
    r, c = resource[1], resource[2]
    cell = r * net.width + c
    return length(resource) == 2 ? cell + 1 : net.height * net.width + 2 * cell + resource[3] + 1
end

@testset "check_action matches Python for every cell type, heading and action" begin
    mismatches = filter(TOPOLOGY["check_action"]) do (code, orientation, action, new_direction, transition_valid, preprocessed, action_valid)
        expected = (new_direction, transition_valid, RailEnvAction(preprocessed), action_valid)
        check_action(RailEnvAction(action), UInt16(code), orientation) != expected
    end
    @test isempty(mismatches)
    @test length(TOPOLOGY["check_action"]) > 100
end

function allocations_of_next_entry_point(net, i)
    next_entry_point(net, i, MOVE_FORWARD)
    return @allocated next_entry_point(net, i, MOVE_FORWARD)
end

@testset "Entry point network matches Python (rail $(rail["seed"]), p_level_free=$(rail["p_level_free"]))" for rail in TOPOLOGY["rails"]
    grid = Int.(permutedims(reduce(hcat, rail["grid"])))
    net = EntryPointNetwork(GridRail(grid, Tuple{Int, Int}[(p[1], p[2]) for p in rail["level_free"]]))
    @test (net.height, net.width) == size(grid)
    @test num_entry_points(net) == 4 * length(grid)

    expected_valid = Set(entry_point_index(net, e["ep"]...) for e in rail["entry_points"])
    @test Set(findall(net.valid)) == expected_valid
    @test all(iszero, net.next[:, .!net.valid])

    next_mismatches, successor_mismatches, resource_mismatches = 0, 0, 0
    for e in rail["entry_points"]
        i = entry_point_index(net, e["ep"]...)
        @test entry_point(net, i) == Tuple(e["ep"])
        for (action, target) in zip(ALL_ACTIONS, e["next"])
            expected = isnothing(target) ? 0 : entry_point_index(net, target...)
            next_mismatches += next_entry_point(net, i, action) != expected
        end
        successor_mismatches += successors(net, i) != [entry_point_index(net, s...) for s in e["successors"]]
        resource_mismatches += net.resource[i] != python_resource(net, e["resource"])
    end
    @test next_mismatches == 0
    @test successor_mismatches == 0
    @test resource_mismatches == 0
    if !isempty(rail["level_free"])
        # level-free crossings: the two axes of the cell are distinct resources
        r, c = rail["level_free"][1]
        @test net.resource[entry_point_index(net, r, c, 0)] != net.resource[entry_point_index(net, r, c, 1)]
        @test net.resource[entry_point_index(net, r, c, 0)] == net.resource[entry_point_index(net, r, c, 2)]
    end

    @test allocations_of_next_entry_point(net, findfirst(net.valid)) == 0
end

@testset "Type stability" begin
    code = UInt16(0b1000_0000_0010_0000)  # straight north-south rail
    JET.@test_opt check_action(MOVE_FORWARD, code, 0)
    rail = GridRail(fill(code, 3, 1))
    JET.@test_opt apply_action(rail, MOVE_FORWARD, 1, 0, 0)
    JET.@test_opt EntryPointNetwork(rail)
    @test apply_action(rail, MOVE_FORWARD, 1, 0, 0) == (0, 0, 0)
    @test apply_action(rail, MOVE_FORWARD, 0, 0, 0) === nothing
end
