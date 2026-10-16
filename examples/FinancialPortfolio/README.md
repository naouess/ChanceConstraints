# Bond portfolio with joint chance constraints

Buy bonds `x ≥ 0` at minimum cost so that cash covers uncertain liabilities in every year.
With cumulative liabilities `ξ_t = β_1 + … + β_t` (multivariate normal) and
`cash_t = K + Σ_i (Σ_{k≤t} yields_ik) x_i`, we compare:

| formulation | requirement |
|---|---|
| deterministic | `cash_t ≥ E[ξ_t]` |
| ICC | `P(cash_t ≥ ξ_t) ≥ p` for each year separately |
| JCC | `P(cash_t ≥ ξ_t, t = s..s+κ) ≥ p` for each window start `s` (`add_JCC_QMC` / `add_JCC_SRD`) |

Individual constraints do not control the probability of a shortfall in *any* year;
the joint constraint does. `run.jl` solves all variants, evaluates them on 100 000 common
scenarios and prints the empirical joint reliability next to the cost.

## Run

```julia
using Pkg
Pkg.activate(".")
Pkg.develop(path = "../..")       # ChanceConstraints.jl
Pkg.add(["JuMP", "Ipopt", "Distributions", "Plots"])
include("run.jl")
```

## Files

- `data.jl` – `PortfolioData`, default instance, distribution of ξ
- `models.jl` – the formulations; `solve_jcc` is where the library is used
- `simulation.jl` – scenarios, cash paths, empirical reliabilities
- `plots.jl` – cash paths, shortfall scenarios in red
- `run.jl` – entry point

## Notes

- `idx` / `window_starts` are the starting years of reliability windows, `κ + 1` is the window length.
- Ipopt's `ma27` (HSL) is optional: pass `linear_solver = "ma27"` to `solve_jcc` if you have it.
