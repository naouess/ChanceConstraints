push!(LOAD_PATH,"../src/")
using Documenter, ChanceConstraints

makedocs(
         sitename = "ChanceConstraints.jl",
         modules  = [ChanceConstraints],
         pages=[
                "Home" => "index.md",
                "Individual Chance Contraints" => "icc.md",
                "Joint Chance Contraints" => "jcc.md",
                "Examples" => "examples.md"
               ])
               
deploydocs(;
    repo="github.com/naouess/ChanceConstraints.jl",
)