using Documenter
using Flatland

makedocs(;
    modules = [Flatland],
    sitename = "Flatland.jl",
    pages = ["Home" => "index.md", "API reference" => "api.md"],
)

deploydocs(;
    repo = "github.com/flatland-association/flatland-rl",
    dirname = "julia",
    devbranch = "main",
    # the Julia package is tagged Flatland-vX.Y.Z, next to the Python package's vX.Y.Z
    tag_prefix = "Flatland-",
)
