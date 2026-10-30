using Flatland
using Flatland: ALL_ACTIONS
using JET: JET
using JSON: JSON
using Test

# exported from the Python package by tests/conformance/export_topology.py
const TOPOLOGY = JSON.parsefile(joinpath(@__DIR__, "data", "topology.json"))

@test TOPOLOGY["entry_point_fields"] == ["r", "c", "d", "valid", "next", "successors", "resource"]

function python_resource(net, resource)
    r, c = resource[1], resource[2]
    cell = r * net.width + c
    return length(resource) == 2 ? cell + 1 : net.height * net.width + 2 * cell + resource[3] + 1
end

python_entry_point(net, ep) = isnothing(ep) ? 0 : entry_point_index(net, ep...)

function rail_and_network(record)
    grid = Int.(permutedims(reduce(hcat, record["grid"])))
    rail = GridRail(grid, Tuple{Int, Int}[(p[1], p[2]) for p in record["level_free"]])
    return grid, rail, EntryPointNetwork(rail)
end

"""
    topology_mismatches(rail, net, entry_points)

Every difference between the Julia functions and tables and the Python entry point records, as `(entry point, what,
Python, Julia)` tuples.
"""
function topology_mismatches(rail, net, entry_points)
    mismatches = []
    check(ep, what, expected, actual) = expected == actual || push!(mismatches, (ep, what, expected, actual))
    for (r, c, d, valid, nexts, python_successors, resource) in entry_points
        ep = (r, c, d)
        i = entry_point_index(net, r, c, d)
        check(ep, "entry_point", ep, entry_point(net, i))
        check(ep, "is_valid_entry_point", valid, is_valid_entry_point(rail, r, c, d))
        check(ep, "valid", valid, net.valid[i])
        for (action, target) in zip(ALL_ACTIONS, nexts)
            check(ep, "apply_action $action", isnothing(target) ? nothing : Tuple(target), apply_action(rail, action, r, c, d))
            check(ep, "next $action", valid ? python_entry_point(net, target) : 0, next_entry_point(net, i, action))
        end
        check(ep, "successors", [python_entry_point(net, s) for s in python_successors], successors(net, i))
        check(ep, "resource", python_resource(net, resource), net.resource[i])
    end
    return mismatches
end

function allocations_of_next_entry_point(net, i)
    next_entry_point(net, i, MOVE_FORWARD)
    return @allocated next_entry_point(net, i, MOVE_FORWARD)
end

@testset "check_action matches Python for every cell type, heading and action" begin
    mismatches = filter(TOPOLOGY["check_action"]) do (code, orientation, action, new_direction, transition_valid, preprocessed, action_valid)
        expected = (new_direction, transition_valid, RailEnvAction(preprocessed), action_valid)
        check_action(RailEnvAction(action), UInt16(code), orientation) != expected
    end
    @test first(mismatches, 5) == []
    @test length(TOPOLOGY["check_action"]) > 100
end

@testset "Entry point network matches Python (rail $(record["seed"]), p_level_free=$(record["p_level_free"]))" for record in TOPOLOGY["rails"]
    grid, rail, net = rail_and_network(record)
    @test (net.height, net.width) == size(grid)
    @test num_entry_points(net) == 4 * length(grid)
    # the valid entry points of these rails, and only them
    @test Set(findall(net.valid)) == Set(entry_point_index(net, e[1], e[2], e[3]) for e in record["entry_points"])
    @test all(iszero, net.next[:, .!net.valid])
    @test first(topology_mismatches(rail, net, record["entry_points"]), 5) == []
    if !isempty(record["level_free"])
        # level-free crossings: the two axes of the cell are distinct resources
        r, c = record["level_free"][1]
        @test net.resource[entry_point_index(net, r, c, 0)] != net.resource[entry_point_index(net, r, c, 1)]
        @test net.resource[entry_point_index(net, r, c, 0)] == net.resource[entry_point_index(net, r, c, 2)]
    end
    @test allocations_of_next_entry_point(net, findfirst(net.valid)) == 0
end

@testset "Random non-square grids match Python at every entry point" begin
    records = TOPOLOGY["random_rails"]
    @test length(records) >= 20
    @test all(record -> length(record["grid"]) != length(record["grid"][1]), records)
    mismatches = []
    for record in records
        _, rail, net = rail_and_network(record)
        @test num_entry_points(net) == length(record["entry_points"])
        append!(mismatches, topology_mismatches(rail, net, record["entry_points"]))
        @test all(r -> 1 <= r <= num_resources(net), net.resource)
    end
    @test first(mismatches, 5) == []
    entry_points = [e for record in records for e in record["entry_points"]]
    # the data contains invalid entry points, and valid ones with and without a target for some actions
    @test any(e -> !e[4], entry_points)
    @test any(e -> e[4] && any(isnothing, e[5]), entry_points)
    @test any(e -> e[4] && all(!isnothing, e[5]), entry_points)
end

@testset "Indices out of range throw" begin
    code = UInt16(0b1000_0000_0010_0000)  # straight north-south rail
    rail = GridRail(fill(code, 3, 2))
    net = EntryPointNetwork(rail)
    @test [entry_point(net, i) for i in 1:num_entry_points(net)] == [(r, c, d) for r in 0:2 for c in 0:1 for d in 0:3]
    @test_throws BoundsError entry_point(net, 0)
    @test_throws BoundsError entry_point(net, num_entry_points(net) + 1)
    @test_throws BoundsError next_entry_point(net, 0, MOVE_FORWARD)
    @test_throws BoundsError next_entry_point(net, num_entry_points(net) + 1, MOVE_FORWARD)
    @test_throws BoundsError apply_action(rail, MOVE_FORWARD, 3, 0, 0)
    @test_throws BoundsError apply_action(rail, MOVE_FORWARD, 0, -1, 0)
    @test_throws ArgumentError check_action(MOVE_FORWARD, code, 4)
    @test_throws ArgumentError check_action(MOVE_FORWARD, code, -1)
end

@testset "GridRail copies its input" begin
    transitions = fill(UInt16(0b1000_0000_0010_0000), 2, 3)
    rail = GridRail(transitions)
    transitions[1, 1] = 0
    @test rail.transitions[1, 1] != 0
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
