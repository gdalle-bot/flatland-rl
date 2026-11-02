# The tests compare Flatland.jl with the Python package `flatland-rl` of the same repository checkout, run in-process
# with PythonCall. The Python interpreter needs the dependencies of `requirements.txt`: point PythonCall to it with the
# environment variables `JULIA_CONDAPKG_BACKEND=Null` and `JULIA_PYTHONCALL_EXE=/path/to/python`.

using PythonCall: Py, pyconvert, pyimport, pyisinstance, pybuiltins, pyis

# the root of the repository, whose `flatland` package is the one under test
const REPOSITORY = normpath(joinpath(@__DIR__, "..", "..", ".."))

"""
    python_module(name)

Import the Python module `name` (e.g. `"flatland.envs.rail_env"` or `"tests.conformance.topology_cases"`) from the
repository checkout, ahead of any installed `flatland-rl`.
"""
function python_module(name::AbstractString)
    sys = pyimport("sys")
    if !pyconvert(Bool, sys.path.__contains__(REPOSITORY))
        sys.path.insert(0, REPOSITORY)
    end
    pyimport("os").environ.setdefault("MPLBACKEND", "Agg")
    return pyimport(name)
end

"""
    julia_data(x::Py)

Convert plain Python data (nested dicts with string keys, lists, tuples, ints, floats, bools, strings and `None`)
to `Dict{String, Any}`, `Vector{Any}`, `Int`, `Float64`, `Bool`, `String` and `nothing`.
"""
function julia_data(x::Py)
    b = pybuiltins
    if pyis(x, b.None)
        return nothing
    elseif pyisinstance(x, b.bool)  # before int: Python's bools are ints
        return pyconvert(Bool, x)
    elseif pyisinstance(x, b.int)
        return pyconvert(Int, x)
    elseif pyisinstance(x, b.float)
        return pyconvert(Float64, x)
    elseif pyisinstance(x, b.str)
        return pyconvert(String, x)
    elseif pyisinstance(x, b.dict)
        return Dict{String, Any}(pyconvert(String, k) => julia_data(v) for (k, v) in x.items())
    elseif pyisinstance(x, b.list) || pyisinstance(x, b.tuple)
        return Any[julia_data(v) for v in x]
    else
        throw(ArgumentError("cannot convert Python data of type $(pyconvert(String, b.type(x).__name__))"))
    end
end
