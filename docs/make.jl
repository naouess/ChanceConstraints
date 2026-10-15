push!(LOAD_PATH,"../src/")
using Documenter, ChanceConstraints

makedocs(
         sitename = "ChanceConstraints.jl",
         modules  = [ChanceConstraints],
         pages=[
                "Home" => "index.md",
                "Functions" => "jcc.md",
                "Examples" => "examples.md",
                "Practical tips" => "tips.md"
               ])
               
deploydocs(;
    repo="github.com/naouess/ChanceConstraints.jl",
)