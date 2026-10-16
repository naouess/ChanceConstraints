using Ipopt, LinearAlgebra, Distributions, Statistics, Random, Printf, Plots
using Main.ChanceConstraints          

include("data.jl")
include("models.jl")
include("simulation.jl")
include("plots.jl")

d       = default_data()
p       = 0.9
T       = n_years(d)
results = joinpath(@__DIR__, "results")

# --- 1. solve ----------------------------------------------------------
w       = 3                                  # rolling window length for the last variant
starts  = collect(1:T-w+1)

solutions = [
    ("deterministic",     solve_deterministic(d)),
    ("ICC",               solve_icc(d, p)),
    ("JCC (full horizon)", solve_jcc(d, p; method = :SRD)),
    ("JCC (rolling $w-yr)", solve_jcc(d, p; method = :SRD, window_starts = starts, κ = w - 1)),
]

# --- 2. evaluate on common scenarios ---------------------------------
ξ = sample_liabilities(d, 100_000; rng = MersenneTwister(1234))

@printf("%-22s %10s %14s %14s\n", "formulation", "cost", "P(no shortfall)", "worst $w-yr window")
for (name, sol) in solutions
    paths = cash_paths(d, sol.x, ξ)
    rel   = empirical_reliability(paths)
    wrel  = minimum(window_reliabilities(paths, starts, w - 1))
    @printf("%-22s %10.2f %14.3f %14.3f\n", name, sol.cost, rel, wrel)

    tag = replace(lowercase(name), r"[^a-z0-9]+" => "_")
    plot_cash_paths(paths[1:100, :]; title = name, file = joinpath(results, "$(tag)_all.pdf"))
    plot_cash_paths(paths[1:100, :]; title = name, only_violations = true,
                    file = joinpath(results, "$(tag)_violations.pdf"))
end