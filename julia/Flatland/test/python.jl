# The tests compare Flatland.jl with the Python package `flatland-rl` of the same repository checkout, run in-process
# with PythonCall. The Python interpreter needs the dependencies of `requirements.txt`: point PythonCall to it with the
# environment variables `JULIA_CONDAPKG_BACKEND=Null` and `JULIA_PYTHONCALL_EXE=/path/to/python`.

using PythonCall: Py, PyArray, pyconvert, pyhasattr, pyimport, pyisinstance, pybuiltins, pyis

# the root of the repository, whose `flatland` package is the one under test
const REPOSITORY = dirname(dirname(dirname(@__DIR__)))

const PYTHON_SETUP = """
the tests of Flatland.jl run the Python package `flatland` of the repository checkout containing them, with \
PythonCall and a Python interpreter that has the packages of requirements.txt installed: set the environment \
variables JULIA_CONDAPKG_BACKEND=Null and JULIA_PYTHONCALL_EXE=/path/to/python"""

"""
    python_module(name)

Import the Python module `name` (e.g. `"flatland.envs.rail_env"` or `"tests.conformance.topology_cases"`) from the
repository checkout, ahead of any installed `flatland-rl`. Throws an error explaining the setup if the checkout or the
Python dependencies are missing, or if `flatland` was already imported from elsewhere.
"""
function python_module(name::AbstractString)
    isdir(joinpath(REPOSITORY, "flatland")) || error("$REPOSITORY is not a checkout of flatland-rl: $PYTHON_SETUP")
    sys = pyimport("sys")
    if !pyconvert(Bool, sys.path.__contains__(REPOSITORY))
        sys.path.insert(0, REPOSITORY)
    end
    pyimport("os").environ.setdefault("MPLBACKEND", "Agg")
    # the generators warn about each env they cannot build as asked, which the tests do on purpose
    pyimport("warnings").simplefilter("ignore", pybuiltins.UserWarning)
    flatland = try
        pyimport("flatland")
    catch err
        error("cannot import the Python package flatland ($err): $PYTHON_SETUP")
    end
    location = pyconvert(String, flatland.__file__)
    startswith(location, joinpath(REPOSITORY, "flatland")) ||
        error("the Python package flatland was imported from $location, not from the checkout $REPOSITORY")
    return pyimport(name)
end

"""
    julia_data(x::Py)

Convert plain Python data (nested dicts with string keys, lists, tuples, ints, floats, bools, strings, `None` and
numpy arrays) to `Dict{String, Any}`, `Vector{Any}`, `Int`, `Float64`, `Bool`, `String`, `nothing` and `Array`s.
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
    elseif pyhasattr(x, "__array_interface__")
        return copy(PyArray(x))
    else
        throw(ArgumentError("cannot convert Python data of type $(pyconvert(String, b.type(x).__name__))"))
    end
end
