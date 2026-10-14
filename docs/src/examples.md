# Examples

This page shows `ChanceConstraints.jl` in use, from minimal examples to a full comparison against alternative formulations using deterministic and individual chance constraints models.

| Example | What it shows | Files |
|---|---|---|
| [SimpleExample](#Simple-example) | Minimal setup and solve | [`SimpleExample.jl`](https://github.com/naouess/ChanceConstraints/blob/main/examples/SimpleExample.jl) |
| [Reservoir problem](#Reservoir-problem) | Simple reservoir problem formulation and solve| [`ReservoirExample.jl`](https://github.com/naouess/ChanceConstraints/blob/main/examples/ReservoirExample.jl) |
| [Financial portfolio optimization](#Comparison) | Full comparison between deterministic vs. individual vs. joint chance constraints on a realistic problem | [`FinancialPortfolio/`](https://github.com/naouess/ChanceConstraints/blob/main/examples/FinancialPortfolio) |

## Simple example

A joint chance constraint requires that *all* components of a random vector are satisfied simultaneously with probability at least `p`. This example illustrates the minimal workflow for that case.

**Problem.** Find the cheapest `x` with unit costs `c` such that

```math
\mathbb{P}(\xi \le x) \ge p, \qquad \xi \sim \mathcal{N}(\mu, \Sigma), \quad p = 0.9,
```

The covariance `Σ` has non-zero off-diagonal entries, so the components are correlated and the joint probability cannot be split into independent per-component constraints.

**What the example shows.**

1. Building a JuMP model and adding the joint chance constraint with 
   `add_JCC_QMC`, which uses quasi-Monte Carlo to approximate the probability.
2. Solving the same problem with `add_JCC_SRD`, which uses spherical-radial
   decomposition.
3. Verifying the optimal solution with an independent estimate of
   `P(ξ ≤ x*)`, which should be close to `p`.

Both formulations should give the same optimal cost.

```@eval
using Markdown
code = read(joinpath(@__DIR__, "..", "..", "examples", "SimpleExample.jl"), String)
Markdown.parse("```julia\n" * code * "\n```")
```

## Reservoir problem example

### Joint-chance-constrained formulation
A hydropower operator chooses how much water to release from a reservoir in each of `T` periods. Releases generate revenue, but the water level must stay within safe limits in **every** period. Because inflows are uncertain, the requirement is stated as a joint chance constraint: the level must stay within its bounds in all periods *simultaneously* with probability at least `p`.

#### Model

Let `x[t]` be the release in period `t` and `ξ[t]` the cumulative inflow up to period `t`. The water level is

```math
\ell_t = \ell_0 + \xi_t - \sum_{s=1}^{t} x_s .
```

The cumulative inflow is `ξ ~ N(μ, Σ)`. It is built from independent yearly inflows with means `β_t` and variances `σ²_t`, so

```math
\mu_t = \sum_{s=1}^{t} \beta_s, \qquad
\Sigma_{ij} = \sum_{s=1}^{\min(i,j)} \sigma^2_s .
```

The cumulative sums make the levels strongly correlated across periods, which is why the constraints must be treated jointly. 

The optimization problem is

```math
\max_{x} \; \sum_{t=1}^{T} x_t
\quad \text{s.t.} \quad
\mathbb{P}\big(\ell_{\min} \le \ell_t \le \ell_{\max},\; t = 1,\dots,T\big) \ge p,
\qquad x_{min} \le x_t \le x_{max} .
```

#### Writing it as a joint chance constraint

`add_JCC_QMC` and `add_JCC_SRD` handle constraints of the form `P(ζ ≤ x) ≥ p` with `ζ ~ N(μ, Σ)`, so the level constraints are rewritten in that form.

**Single-sided (lower bound only).** The constraint `ℓ_t ≥ ℓ_min` is equivalent to

```math
-\xi_t \le \ell_0 - \ell_{\min} - \sum_{s \le t} x_s ,
```

and `-ξ ~ N(-μ, Σ)`. Pass the right-hand side (the slack) as the bound and
`-μ` as the mean:

```julia
cumx  = [sum(x[1:t]) for t in 1:T]
slack = [l_0 - l_min - cumx[t] for t in 1:T]
add_JCC_QMC(model, slack, [1], T - 1, Σ, -μ, p)
```

The window starting at time step `[1]` with `κ = T - 1` covers all `T` periods in a single joint constraint.

**Two-sided (lower and upper bound).** The constraint
`ℓ_min ≤ ℓ_t ≤ ℓ_max` is equivalent to `lower_t ≤ ξ_t ≤ upper_t` with

```math
\text{lower}_t = \ell_{\min} - \ell_0 + \sum_{s \le t} x_s, \qquad
\text{upper}_t = \ell_{\max} - \ell_0 + \sum_{s \le t} x_s .
```

Stack the bounds as `[lower; upper]` (length `2T`) and set `two_sided = true`.
The random vector still has `T` components, so `κ` stays `T - 1`:

```julia
add_JCC_QMC(model, vcat(lower, upper), [1], T - 1, Σ, μ, p; two_sided = true)
```

#### What the example shows

1. **Single-sided, QMC and SRD.** Both approximations give the same optimal total release.
2. **Two-sided, QMC.** Adding the upper bound makes the problem slightly more
   restrictive. Two-sided implementation is currently not possible using SRD.
3. **Monte Carlo check.** The probability that the level stays within bounds
   is estimated by simulating 200,000 inflow scenarios. It should be close to
   `p = 0.9` for both solutions.

```@eval
using Markdown
code = read(joinpath(pwd(), "..", "..", "examples", "ReservoirExample.jl"), String)
Markdown.parse("```julia\n" * code * "\n```")
```

### Individual-chance-constrained formulation

Instead of requiring the level to stay in bounds in all periods jointly, each period is constrained on its own:

```math
\mathbb{P}(\ell_t \ge \ell_{\min}) \ge p, \qquad t = 1,\dots,T .
```

This ignores the correlation between periods. Because each `ξ_t` is a scalar Gaussian, `ξ_t ~ N(μ_t, Σ_tt)`, every constraint has a closed form. With
`z_p = Φ⁻¹(p)` and `σ_t = √Σ_tt`:

```math
\mathbb{P}(-\xi_t \le \ell_0 - \ell_{\min} - \textstyle\sum_{s \le t} x_s) \ge p
\iff
\sum_{s \le t} x_s \le \ell_0 - \ell_{\min} + \mu_t - z_p\,\sigma_t .
```

These are linear in `x`, so no package call is needed:

```julia
using Distributions, LinearAlgebra

z = quantile(Normal(), p)
σ = sqrt.(diag(Σ))

model = Model(Ipopt.Optimizer)
set_silent(model)

@variable(model, 0 <= x[1:T] <= 30)
@objective(model, Max, sum(x))

cumx = [sum(x[1:t]) for t in 1:T]
@constraint(model, [t = 1:T], cumx[t] <= l_0 - l_min + μ[t] - z * σ[t])

optimize!(model)
x_ind = value.(x)
objective_value(model)
```

**Two-sided version.** The constraint `P(lower_t ≤ ξ_t ≤ upper_t) ≥ p` becomes

```math
\Phi\!\left(\tfrac{\text{upper}_t - \mu_t}{\sigma_t}\right)
- \Phi\!\left(\tfrac{\text{lower}_t - \mu_t}{\sigma_t}\right) \ge p ,
```

which is nonlinear but smooth. Using `Φ(u) = ½(1 + erf(u/√2))`:

```julia
using SpecialFunctions

Φ(u) = 0.5 * (1 + erf(u / sqrt(2)))

lower = [l_min - l_0 + cumx[t] for t in 1:T]
upper = [l_max - l_0 + cumx[t] for t in 1:T]

@constraint(model, [t = 1:T],
    Φ((upper[t] - μ[t]) / σ[t]) - Φ((lower[t] - μ[t]) / σ[t]) >= p)
```

**Checking the joint probability.** Reuse the Monte Carlo check from the joint example on `x_ind`:

```julia
mean(all(level(x_ind) .>= l_min; dims = 1))
```

Each period satisfies its own constraint with probability `p`, but the probability that *all* periods are satisfied together is generally lower than `p`. That gap is what the joint formulation closes.

## Financial portfolio example

