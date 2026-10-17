# Examples

This page shows `ChanceConstraints.jl` in use, from minimal examples to a full comparison against alternative formulations like deterministic and individual chance constraints models.

| Example | What it shows | Files |
|---|---|---|
| [SimpleExample](#Simple-example) | Minimal setup and solve | [`SimpleExample.jl`](https://github.com/naouess/ChanceConstraints/blob/main/examples/SimpleExample.jl) |
| [Reservoir problem](#Reservoir-problem-example) | Solve a simple reservoir problem formulation (both with joint and individual chance constraints) | [`ReservoirExample.jl`](https://github.com/naouess/ChanceConstraints/blob/main/examples/ReservoirExample.jl) and [`ReservoirExampleICC.jl`](https://github.com/naouess/ChanceConstraints/blob/main/examples/ReservoirExampleICC.jl) |
| [Financial portfolio optimization](#Financial-potfolio-example) | Full comparison between deterministic vs. individual vs. joint chance constraints on a realistic problem | [`FinancialPortfolio/`](https://github.com/naouess/ChanceConstraints/blob/main/examples/FinancialPortfolio) |

The following sections are structured as follows:
```@contents
Pages = ["examples.md"]
Depth = 3
```

## Simple example

A joint chance constraint requires that several components of a random vector are satisfied simultaneously with probability at least `p`. This example illustrates the minimal workflow for that case.

**Problem.** Find the cheapest $x$ with unit costs $c$ such that

```math
\mathbb{P}(\xi \le x) \ge p, \qquad \xi \sim \mathcal{N}(\mu, \Sigma), \quad p = 0.9,
```

The covariance $Σ$ has non-zero off-diagonal entries, so the components are correlated and the joint probability cannot be split into independent per-component constraints.

#### What the example shows

1. Building a JuMP model and adding the joint chance constraint with `add_JCC_QMC`, which uses quasi-Monte Carlo to approximate the probability.
2. Solving the same problem with `add_JCC_SRD`, which uses spherical-radial decomposition.
3. Verifying the optimal solution with an independent estimate of $P(ξ ≤ x^*)$, which should be close to $p$.

Both formulations should give the same optimal cost.

#### Full implementation

```@eval
using Markdown
code = read(joinpath(@__DIR__, "..", "..", "examples", "SimpleExample.jl"), String)
Markdown.parse("```julia\n" * code * "\n```")
```

## Reservoir problem example

### Joint-chance-constrained formulation
A hydropower operator chooses how much water to release from a reservoir in each of $T$ periods. Releases generate revenue, but the water level must stay within safe limits in **every** period. Because inflows are uncertain, the requirement is stated as a joint chance constraint: the level must stay within its bounds in all periods *simultaneously* with probability at least $p$.

#### Model

Let $x_t$ be the release in period $t$ and $ξ_t$ the cumulative inflow up to period $t$. The water level is

```math
\ell_t = \ell_0 + \xi_t - \sum_{s=1}^{t} x_s .
```

The cumulative inflow is $ξ \backsim N(μ, Σ)$. It is built from independent yearly inflows with means $β_t$ and variances $σ²_t$, so

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

`add_JCC_QMC` and `add_JCC_SRD` handle constraints of the form $P(ζ ≤ x) ≥ p$ with $ζ \backsim N(μ, Σ)$, so the level constraints are rewritten in that form.

**Single-sided (lower bound only).** The constraint $ℓ_t ≥ ℓ_min$ is equivalent to

```math
-\xi_t \le \ell_0 - \ell_{\min} - \sum_{s \le t} x_s ,
```

and $-ξ \backsim N(-μ, Σ)$. Pass the right-hand side (the slack) as the bound and
$-μ$ as the mean:

```julia
cumx  = [sum(x[1:t]) for t in 1:T]
slack = [l_0 - l_min - cumx[t] for t in 1:T]
add_JCC_QMC(model, slack, [1], T - 1, Σ, -μ, p)
```

The window starting at time step `[1]` with duration $κ = T - 1$ covers all $T$ periods in a single joint constraint.

**Two-sided (lower and upper bound).** The constraint
$ℓ_\text{min} ≤ ℓ_t ≤ ℓ_\text{max}$ is equivalent to $\text{lower}_t ≤ ξ_t ≤ \text{upper}_t$ with

```math
\text{lower}_t = \ell_{\min} - \ell_0 + \sum_{s \le t} x_s, \qquad
\text{upper}_t = \ell_{\max} - \ell_0 + \sum_{s \le t} x_s .
```

Stack the bounds as `[lower; upper]` (length $2T$) and set `two_sided = true`.
The random vector still has $T$ components, and $κ$ stays $T - 1$:

```julia
add_JCC_QMC(model, vcat(lower, upper), [1], T - 1, Σ, μ, p; two_sided = true)
```

#### What the example shows

1. **Single-sided, QMC and SRD.** Both approximations give the same optimal solution.
2. **Two-sided, QMC.** Adding the upper bound makes the problem slightly more    restrictive. *Two-sided implementation is currently not possible using SRD.*
3. Verifying the optimal solution with an independent estimate of $P(ξ ≤ x*)$, which should be close to $p$.

#### Full implementation

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

This ignores the correlation between periods. Because each $ξ_t$ is a scalar Gaussian, $ξ_t \backsim N(μ_t, Σ_{tt})$, every constraint has a closed form. With $z_p = Φ⁻¹(p)$ and $σ_t = √Σ_{tt}$:

```math
\mathbb{P}(-\xi_t \le \ell_0 - \ell_{\min} - \textstyle\sum_{s \le t} x_s) \ge p
\iff
\sum_{s \le t} x_s \le \ell_0 - \ell_{\min} + \mu_t - z_p\,\sigma_t .
```

These are linear in $x$, so no package call is needed:

```julia
using Distributions, LinearAlgebra
using JuMP, Ipopt

z = quantile(Normal(), p)
σ = sqrt.(diag(Σ))

model = Model(Ipopt.Optimizer)
set_silent(model)

@variable(model, 0 <= x[1:T] <= 30)
@objective(model, Max, sum(price[t] * x[t] for t in 1:T))

cumx = [sum(x[1:t]) for t in 1:T]
@constraint(model, [t = 1:T], cumx[t] <= l_0 - l_min + μ[t] - z * σ[t])

optimize!(model)
x_ind = value.(x)
objective_value(model)
```

**Checking the joint probability.** Reuse the independent Monte Carlo check on the individual-chance constraints solution $x_\text{ind}$. Each period satisfies its own constraint with probability $p = 0.9$, but the probability that *all* periods are satisfied *jointly* is lower than $p$ and equal to $p_\text{ind} ≈ 0.8$. That gap is what the joint formulation closes.

The full script for the implementation of the reservoir problem with individual chance constraints, both in the one-sided and two-sided case can be found in [ReservoirExampleICC](https://github.com/naouess/ChanceConstraints/blob/main/examples/ReservoirExampleICC.jl).

## Financial portfolio example

This example aims to find the optimal way to fund a stream of uncertain yearly liabilities with a portfolio of bonds. It compares three ways of handling the uncertainty (a deterministic model, individual chance constraints, and joint chance constraints with `ChanceConstraints.jl`).

### Problem description

A company has to pay uncertain liabilities $\beta_t$ in the years $t = 1,\dots,T$ with $T = 8$. They are independent and normally distributed, $\beta_t \sim \mathcal{N}(m_t, s_t^2)$. To fund them, the company buys $n = 10$ bonds. Bond $i$ costs $c_i$ per unit and pays $y_{it}$ in year $t$ (coupons, plus the principal at maturity). The decision variable $x_i \ge 0$ is the number of units of bond $i$. The initial cash is $K$, and cash that is received is carried forward without interest.

The cash available up to year $t$ is

```math
C_t(x) = K + \sum_{i=1}^{n} \Big(\sum_{k \le t} y_{ik}\Big) x_i ,
```

and it has to cover the cumulative liability

```math
\xi_t = \sum_{k=1}^{t} \beta_k .
```

Since the $\beta_k$ are independent normals, $\xi = (\xi_1,\dots,\xi_T)$ is a
multivariate normal vector with

```math
\mu_t = \sum_{k=1}^{t} m_k , \qquad
\Sigma_{ij} = \sum_{k=1}^{\min(i,j)} s_k^2 .
```

The goal is to buy the cheapest portfolio that covers the liabilities in every year:

```math
\min_{x \ge 0} \; c^\top x
\quad \text{subject to} \quad C_t(x) \ge \xi_t \ \text{ for all } t = 1,\dots,T .
```

Because $\xi$ is random, the formulation of the constraint depends on the adopted model. The example considers the following formulations:

| formulation | requirement | implemented by |
|---|---|---|
| deterministic | $C_t(x) \ge \mathbb{E}[\xi_t]$ | plain linear JuMP constraints |
| individual chance constraints (ICC) | $\mathbb{P}(C_t(x) \ge \xi_t) \ge p$ for each $t$ separately | closed-form quantile |
| joint chance constraint (JCC) | $\mathbb{P}(C_t(x) \ge \xi_t, \ t = 1,\dots,T) \ge p$ | `add_JCC_QMC` and `add_JCC_SRD` |
| JCC with rolling windows | the same, but for every window of $w = 3$ consecutive years (years 1--3, 2--4, ..., 6--8) | `add_JCC_QMC` and `add_JCC_SRD` with several window starts |

The joint constraint is imposed in one call. The vector `cash` contains the affine
expressions $C_t(x)$:

```julia
add_JCC_QMC(model, cash, [1], T - 1, Σ, μ, p)        # one window, years 1..T
add_JCC_QMC(model, cash, 1:T-w+1, w - 1, Σ, μ, p)    # rolling windows of length w
```

The second argument after `cash` is the set of window starts, and `κ = w - 1` is the window length minus one.

### What to expect

Qualitatively, we expect that:

- the deterministic portfolio is the cheapest, but it fails in a large share of scenarios of realized uncertainty;
- the ICC portfolio is more expensive than the deterministic case because it guarantees a proability $p$ each year separately, but its joint reliability is still clearly below $p$;
- the JCC portfolio reaches the joint reliability $p$ and is the most expensive one;
- the rolling-window JCC lies between ICC and the full-horizon JCC.

### File structure

You can access the files here: \
[Financial portfolio optimization](https://github.com/naouess/ChanceConstraints/blob/main/examples/FinancialPortfolio)

The files are structured as follows:

```
examples/FinancialPortfolio/
├── README.md
├── data.jl          # problem data and the distribution of ξ
├── models.jl        # the formulations (deterministic, ICC, JCC)
├── simulation.jl    # scenario generation and ex-post evaluation
├── plots.jl         # plot of cash paths with shortfall scenarios highlighted
├── run.jl           # main file to run the example
└── results/         # folder where results figures are saved
```

| file | content |
|---|---|
| `data.jl` | The `PortfolioData` struct (initial cash, bond costs, yields, distribution of the yearly liabilities), `default_data()` with the data of this example, and `liability_distribution(d)`, which returns $\mu$ and $\Sigma$ of the cumulative liabilities. |
| `models.jl` | `base_model` (variables, cash expressions, cost objective) and one function per formulation: `solve_deterministic`, `solve_icc` and `solve_jcc`. In `solve_jcc`, the function chooses between `add_JCC_QMC` and `add_JCC_SRD` and takes the window starts and `κ` as keyword arguments. Each solve function returns the portfolio `x`, its cost and the solver status. |
| `simulation.jl` | `sample_liabilities` draws correlated scenarios of $\xi$. `cash_paths` computes cash minus liabilities for each scenario and year. `empirical_reliability` and `window_reliabilities` give the fraction of scenarios without a shortfall. |
| `plots.jl` | `plot_cash_paths` draws scenario paths, in red when there is a shortfall. |
| `run.jl` | Solves all formulations, evaluates them on common scenarios, prints a table and saves the figures. |

### How to run

From the command line:

```
julia --project run.jl
```

`run.jl` prints a table with one row per formulation: the cost of the portfolio, the empirical probability of no shortfall in any year, and the worst reliability over the rolling windows.

### Adapting the example

- **Probability level.** Change `p` in `run.jl`.
- **Method.** Also in `run.jl`, pass `method = :QMC` or `method = :SRD` to `solve_jcc`. SRD requires $p > 1/2$.
- **Window length.** Change `w` in `run.jl` (the windows must lie within the `T` years).
- **Your own data.** Construct a `PortfolioData` with your costs, yields and liability distributions instead of calling `default_data()`.
- **Solver.** The model is solved with Ipopt. You can use any installed linear solver you have by passing it to `solve_jcc`, for example `linear_solver = "ma27"`.

### Results
 
Results for the default data with $p = 0.9$. Every portfolio is evaluated out of sample on the same 100 000 simulated scenarios.
 
| formulation | cost | extra cost vs. deterministic | $\mathbb{P}(\text{no shortfall})$ | worst 3-yr window |
|:--|--:|--:|--:|--:|
| deterministic | 101.62 | – | 0.106 | 0.290 |
| ICC | 165.30 | +62.7 % | 0.656 | 0.805 |
| JCC (rolling 3-yr) | 172.97 | +70.2 % | 0.767 | 0.900 |
| JCC (full horizon) | 181.77 | +78.9 % | 0.900 | 0.910 |
 
- **$\mathbb{P}(\text{no shortfall})$** is the fraction of scenarios in which the cash covers the   liabilities in *all* eight years.
- **worst 3-yr window** is the lowest, over the six windows, of the fraction of scenarios without a shortfall inside that window.

What the table shows:
 
- **Deterministic.** Covering the *expected* liabilities is the cheapest option, but only
  about 11 % of the scenarios are free of shortfalls.
- **Individual chance constraints.** Each year is covered with probability 90%, yet the  probability of no shortfall in any year is only $\approx$ 66%: in roughly a third of the  scenarios the cash falls short at least once.
- **Rolling windows.** Every 3-year window is covered with probability 90%, at a cost of 70% more than in the deterministic case, which represents additional 7.5 percentage points compared to the individual constraints. This protects each  3-year stretch, but not the whole horizon: the probability of no shortfall in any year is $\approx$ 77%.
- **Full-horizon joint chance constraint.** The joint reliability is 90%, which matches the predefined target value. Guaranteeing it costs around 79% compared to the deterministic case, which represents 16.2 percentage points more than the individual constraints and 8.7 percentage points more than the rolling windows.