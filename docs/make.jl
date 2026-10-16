push!(LOAD_PATH,"../src/")
using Documenter, ChanceConstraints

makedocs(
         sitename = "ChanceConstraints.jl",
         format = Documenter.HTML(
             toc_depth = 3,
         ),
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