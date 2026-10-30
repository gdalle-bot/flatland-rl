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
)
