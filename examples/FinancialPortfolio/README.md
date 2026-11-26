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

## Mapping to the library

The joint constraint has the form `P(ζ ≤ h(x)) ≥ p` with `ζ = Aξ`. Here `A = I`, so `ζ = ξ`, and the bounds are the affine expressions `h(x) = cash(x)`.

- `μ`, `Σ`: mean and covariance of the cumulative liabilities `ξ` (from `liability_distribution`).
- `h`: the vector `cash` of affine expressions in `x`, one per year, passed as the first argument after the model.
- Window starts and `κ`: `idx` and `κ = w - 1`, where `w` is the window length.

```julia
add_JCC_QMC!(model, cash, [1], T - 1, Σ, μ, p)        # one window, years 1..T
add_JCC_QMC!(model, cash, 1:T-w+1, w - 1, Σ, μ, p)    # rolling windows of length w
```

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
- Window length: QMC requires `κ ≥ 2` (at least 3 years), SRD requires `κ ≥ 1` (at least 2 years).
- SRD (`method = :SRD`) requires `0.5 < p < 1`, because it needs the mean to be strictly inside the constraints, `cash(x) > μ`. QMC works for any `p ∈ (0, 1)`.
- Ipopt's `ma27` (HSL) is optional: pass `linear_solver = "ma27"` to `solve_jcc` if you have it.