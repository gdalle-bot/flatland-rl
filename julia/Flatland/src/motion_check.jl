"""
    MotionCheck(num_resources, num_agents)

Resolution of the conflicts between the moves that the agents ask for in a step, as
`flatland.envs.agent_chains.MotionCheck` in Python. Each agent holds a resource (a cell, or an axis of a level-free
crossing) and asks for one; an agent off the map holds a resource of its own, and so does an agent that asks for none.

Agents are numbered `1:num_agents` and resources `1:num_resources` (the [`num_resources`](@ref) of an
[`EntryPointNetwork`](@ref)). All buffers are allocated by the constructor, so a step allocates nothing once the
queue of conflicts has reached its largest size.

In each step: [`reset!`](@ref), [`add_agent!`](@ref) for every agent in increasing order, [`find_conflicts!`](@ref),
then [`check_resource`](@ref) for each agent. The resolution, as in Python:

- two agents asking for each other's resource (a swap) are both stopped, and are deadlocked;
- of two agents asking for the same resource, the one with the larger number is stopped;
- an agent asking for a resource whose holder is stopped, or keeps it, is stopped;
- conflicts are processed in first-in, first-out order: those found by scanning the agents in order, then those
  created by stopping an agent.

The deadlocked agents are stopped in increasing order, where Python iterates over a `set`: the agents that may move
do not depend on that order.

Each resource may be held by one agent at most, as in the env. [`add_agent!`](@ref) throws an `ArgumentError`
otherwise, where Python's resolution may not terminate.
"""
struct MotionCheck
    num_resources::Int
    # per agent: the resource held, the resource asked for (the held one once stopped), and the flags of Python
    position::Vector{Int32}
    target::Vector{Int32}
    added::Vector{Bool}
    hold::Vector{Bool}
    no_resource::Vector{Bool}
    stopped::Vector{Bool}
    deadlocked::Vector{Bool}
    # agents in the order they were added
    order::Vector{Int32}
    # per resource: the agent holding it, 0 if none
    holder::Vector{Int32}
    # for each resource, the agents asking for it in the order they asked, as a doubly linked list
    head::Vector{Int32}
    tail::Vector{Int32}
    next::Vector{Int32}
    prev::Vector{Int32}
    # conflicts between two agents asking for the same resource, smaller agent first
    queue::Vector{Tuple{Int32, Int32}}
end

function MotionCheck(num_resources::Integer, num_agents::Integer)
    num_resources >= 0 || throw(ArgumentError("num_resources must be non-negative, got $num_resources"))
    num_agents >= 0 || throw(ArgumentError("num_agents must be non-negative, got $num_agents"))
    agent_ints() = zeros(Int32, num_agents)
    agent_flags() = zeros(Bool, num_agents)
    queue = Tuple{Int32, Int32}[]
    sizehint!(queue, 4 * num_agents)
    return MotionCheck(
        num_resources, agent_ints(), agent_ints(), agent_flags(), agent_flags(), agent_flags(), agent_flags(),
        agent_flags(), sizehint!(Int32[], num_agents), zeros(Int32, num_resources), zeros(Int32, num_resources + num_agents),
        zeros(Int32, num_resources + num_agents), agent_ints(), agent_ints(), queue,
    )
end

num_agents(mc::MotionCheck) = length(mc.position)

"""
    reset!(mc::MotionCheck)

Forget the agents of the previous step.
"""
function reset!(mc::MotionCheck)
    for i in mc.order
        # every agent is in the list of the resource it asks for, so this empties all lists
        mc.head[mc.target[i]] = 0
        mc.tail[mc.target[i]] = 0
        mc.position[i] <= mc.num_resources && (mc.holder[mc.position[i]] = 0)
        mc.added[i] = mc.hold[i] = mc.no_resource[i] = mc.stopped[i] = mc.deadlocked[i] = false
    end
    empty!(mc.order)
    empty!(mc.queue)
    return mc
end

"""
    add_agent!(mc::MotionCheck, i, held, wanted)

Register agent `i`, holding resource `held` and asking for resource `wanted`; `0` stands for none (off the map, or
nothing to ask for). Asking for the held resource means staying. Throws an `ArgumentError` if `i` was already added
since the last [`reset!`](@ref), if another agent holds `held`, or if a number is out of range.
"""
function add_agent!(mc::MotionCheck, i::Integer, held::Integer, wanted::Integer)
    1 <= i <= num_agents(mc) || throw(ArgumentError("agent $i is not in 1:$(num_agents(mc))"))
    0 <= held <= mc.num_resources || throw(ArgumentError("resource $held is not in 0:$(mc.num_resources)"))
    0 <= wanted <= mc.num_resources || throw(ArgumentError("resource $wanted is not in 0:$(mc.num_resources)"))
    mc.added[i] && throw(ArgumentError("agent $i was already added"))
    held != 0 && mc.holder[held] != 0 && throw(ArgumentError("resource $held is held by agents $(mc.holder[held]) and $i"))
    own = Int32(mc.num_resources + i)
    position = held == 0 ? own : Int32(held)
    target = wanted == 0 ? own : Int32(wanted)
    mc.added[i] = true
    held != 0 && (mc.holder[held] = i)
    mc.no_resource[i] = wanted == 0
    mc.hold[i] = position == target
    mc.position[i] = position
    mc.target[i] = target
    push!(mc.order, i)
    append_to_list!(mc, target, Int32(i))
    return mc
end

function append_to_list!(mc::MotionCheck, resource::Int32, i::Int32)
    last = mc.tail[resource]
    mc.prev[i] = last
    mc.next[i] = 0
    if last == 0
        mc.head[resource] = i
    else
        mc.next[last] = i
    end
    mc.tail[resource] = i
    return nothing
end

function remove_from_list!(mc::MotionCheck, resource::Int32, i::Int32)
    before, after = mc.prev[i], mc.next[i]
    if before == 0
        mc.head[resource] = after
    else
        mc.next[before] = after
    end
    if after == 0
        mc.tail[resource] = before
    else
        mc.prev[after] = before
    end
    return nothing
end

"""
    find_conflicts!(mc::MotionCheck)

Resolve the conflicts between the agents added since the last [`reset!`](@ref).
"""
function find_conflicts!(mc::MotionCheck)
    for i in mc.order
        position, target = mc.position[i], mc.target[i]
        # swaps: a later agent asks for this agent's resource, and this agent for the later agent's
        j = mc.head[position]
        while j != 0
            if i < j && mc.position[j] == target
                mc.stopped[i] = mc.stopped[j] = true
                mc.deadlocked[i] = mc.deadlocked[j] = true
            end
            j = mc.next[j]
        end
        # later agents asking for the same resource
        j = mc.head[target]
        while j != 0
            i < j && push!(mc.queue, (i, j))
            j = mc.next[j]
        end
        position == target && (mc.stopped[i] = true)
    end
    for i in eachindex(mc.deadlocked)
        mc.deadlocked[i] && stop!(mc, Int32(i))
    end
    k = 1
    while k <= length(mc.queue)
        u, v = mc.queue[k]
        k += 1
        if mc.position[v] == mc.target[v]
            # v stays: u cannot enter its resource
            stop!(mc, u)
        elseif mc.target[u] == mc.target[v]
            # both move to the same resource: the smaller agent wins
            stop!(mc, v)
        end
    end
    return mc
end

# Stop agent `v` in its resource, and queue its conflicts with the agents asking for that resource.
function stop!(mc::MotionCheck, v::Int32)
    position, target = mc.position[v], mc.target[v]
    mc.stopped[v] = true
    j = mc.head[position]
    while j != 0
        j != v && push!(mc.queue, v < j ? (v, j) : (j, v))
        j = mc.next[j]
    end
    remove_from_list!(mc, target, v)
    append_to_list!(mc, position, v)
    mc.target[v] = position
    return nothing
end

"""
    check_resource(mc::MotionCheck, i)

Whether agent `i` may take the resource it asked for: always if it asked to stay or for nothing, and otherwise unless
[`find_conflicts!`](@ref) stopped it.
"""
check_resource(mc::MotionCheck, i::Integer) = mc.hold[i] || mc.no_resource[i] || !mc.stopped[i]
