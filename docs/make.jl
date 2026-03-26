using Documenter
using CuQuantum

makedocs(;
    sitename = "CuQuantum.jl",
    modules = [CuQuantum],
    authors = "Harmoniqs",
    format = Documenter.HTML(;
        canonical = "https://harmoniqs.github.io/CuQuantum.jl",
        edit_link = "main",
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting-started.md",
        "Concepts" => [
            "Overview" => "concepts/overview.md",
            "Operators" => "concepts/operators.md",
            "Callbacks" => "concepts/callbacks.md",
            "States" => "concepts/states.md",
        ],
        "API Reference" => [
            "CuDensityMat" => "api/cudensitymat.md",
        ],
        "Benchmarks" => "benchmarks.md",
    ],
)

deploydocs(;
    repo = "github.com/harmoniqs/CuQuantum.jl",
    devbranch = "main",
    push_preview = true,
)
