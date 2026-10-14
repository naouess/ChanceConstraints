# ChanceConstraints.jl

_A package for computing and optimizing with Joint Chance Constraints in Julia._

```@contents
Pages = ["index.md"]
```

## Install 
```julia repl
# pkg> add ChanceConstraints
```

## Resources for getting started

Here some math stuff \
Some JuMP and Julia stuff \
Refer to our paper in Annals of OR

## The math behind it all

Here we explain Chance constraints 


`ChanceConstraints.jl` adds **joint chance constraints** with Gaussian uncertainty to [JuMP](https://jump.dev) models. It computes the joint probability and its gradient directly, instead of relying on conservative approximations.

## Getting started

This page covers:

1. [Why chance constraints](#Why-chance-constraints)
2. [Individual vs. joint chance constraints](#Individual-vs.-joint-chance-constraints)
3. [Bonferroni vs. exact computation](#Bonferroni-vs.-exact-computation)
4. [The math](#The-math)
5. [Julia and JuMP basics](#Julia-and-JuMP-basics)
6. [A first model](#A-first-model)

The [Examples](examples.md) page then shows complete models.

## Why chance constraints

Many optimization problems depend on quantities that are not known when the
decision is made: inflows to a reservoir, demand, solar and wind generation.
Two common ways of handling this both have drawbacks:

- **Deterministic models** replace the uncertain quantity by its expected value. The
  resulting solution often fails when the realization differs from the mean.
- **Worst-case (robust) models** protect against every realization. This is
  safe but often far too expensive.

A **chance constraint** sits in between. It asks that a constraint holds with
at least a chosen probability `p` (for example 90% or 95%), and accepts a
small violation risk `1 - p` in exchange for a cheaper solution. The risk level
is an explicit modelling choice, and the cost of reliability can be adjusted by
changing `p`.

## Individual vs. joint chance constraints

Suppose a decision `x` must satisfy `m` uncertain constraints, for example one
per time period.

An **individual chance constraint** treats each constraint separately:

```math
\mathbb{P}\big(g_i(x,\xi) \le 0\big) \ge p, \qquad i = 1,\dots,m .
```

Each constraint is satisfied with probability `p`, but this says nothing about
all of them holding *at the same time*.

A **joint chance constraint** requires all constraints to hold simultaneously:

```math
\mathbb{P}\big(g_i(x,\xi) \le 0,\; i = 1,\dots,m\big) \ge p .
```

This is what is usually meant by "the system is reliable". When the constraints are correlated, for instance because they depend on cumulative inflows or on the same weather conditions, individual constraints can be misleading: the probability that *everything* holds is generally lower than `p`,
so the system is less reliable than the model suggests.

The possibly striking difference between individual and joint chance constraints has been widely studied (e.g., [Van Ackooij et al., 2010, p. 547](https://doi.org/10.1007/s00186-010-0316-3), [Berthold et al., 2022, p. 34](https://doi.org/10.1080/02331934.2019.1576670)).


## Bonferroni vs. exact computation

A classical way to avoid the joint probability is the **Bonferroni
approximation** [(Nemirovski and Shapiro, 2006)](https://doi.org/10.1137/050622328). Since

```math
\mathbb{P}\Big(\bigcup_{i} \{g_i > 0\}\Big) \le \sum_{i=1}^m \mathbb{P}(g_i > 0),
```

requiring each individual constraint to hold with probability `1 - (1-p)/m` guarantees the joint constraint at level `p`. This gives a model with only individual constraints, each with a closed form for Gaussian data.

The price is conservatism. The bound ignores correlation and assumes the violation events are disjoint, so the resulting solution is typically more expensive than necessary, and the gap grows with `m`.

This package instead evaluates the joint probability itself. For Gaussian uncertainty, it is a multivariate normal CDF, which can be computed accurately with the methods described below.

## The math

**Setting.** Let `ξ ~ N(μ, Σ)` be a Gaussian random vector. The constraints
treated by this package have the form

```math
\mathbb{P}\big(\xi \le h(x)\big) \ge p , 
```
<!-- TODO here: adjust h(x) vs x -->

where `h(x)` is a vector of bounds depending on the decision variables `x`.

**Individual case.** For a single component `ξ_i ~ N(μ_i, Σ_ii)`, with
`σ_i = √Σ_ii` and `Φ` the standard normal CDF,

```math
\mathbb{P}(\xi_i \le h_i(x)) \ge p 
\iff
h_i(x) \ge \mu_i + \sigma_i\,\Phi^{-1}(p) ,
```

a closed-form deterministic constraint.

**Joint case.** The probability

```math
F(x) = \mathbb{P}(\xi \le h(x)) = \Phi_\Sigma\big(h(x)-\mu\big)
```

is a multivariate normal CDF. It has no closed form for `m > 2`, but it is
smooth, and it is log-concave, so for `p ≥ 1/2` and `h` affine in `x`, the
feasible set `{x : F(x) ≥ p}` is convex.

**Computation.** Two approaches are available.

- **Quasi-Monte Carlo (QMC)** (`compute_with_QMC` and `add_JCC_QMC`). The CDF value and its gradient are evaluated numerically with a low-discrepancy sequence, following the algorithm of Genz (1992) implemented in Julia using the library [MvNormalCDF](https://github.com/PharmCat/MvNormalCDF.jl). The gradient is passed to the solver.
- **Spherical-radial decomposition (SRD)** (`compute_with_SRD` and`add_JCC_SRD`). Writing `ξ = μ + R·L·u`, with `Σ = LLᵀ`, `u` uniform on the unit sphere and `R` 
  chi-distributed, the probability becomes an expectation over directions `u` of
  a one-dimensional chi CDF, computed along the ray from the mean to the boundary. SRD requires the mean to be strictly feasible, `h(x) > μ`.

**Reliability windows.** For time series, a constraint need not cover the whole
horizon. The arguments `[starts]` and `κ` define windows of `κ + 1` consecutive
periods, each required to hold jointly with probability `p`. A single window
covering all periods gives one joint constraint over the horizon.

## Julia and JuMP basics

**Julia.** Install Julia from [julialang.org](https://julialang.org/downloads/)
<!-- (1.9 or later is recommended; check `Project.toml` for the tested range). -->
To add packages, press `]` in the Julia REPL to enter the package manager:

```julia
pkg> add JuMP Ipopt
pkg> add https://github.com/you/ChanceConstraints.jl 
```
<!-- adjust here once package is published -->

Useful starting points are the
[Julia documentation](https://docs.julialang.org) and
[Julia Data Science](https://juliadatascience.io).

**JuMP.** JuMP is a modelling language for optimization. A model has variables,
an objective, constraints and a solver:

```julia
using JuMP, Ipopt

model = Model(Ipopt.Optimizer)       # choose a solver
@variable(model, x[1:3] >= 0)        # decision variables
@constraint(model, sum(x) <= 10)     # constraints
@objective(model, Max, 2x[1] + x[2] + x[3])
optimize!(model)

objective_value(model)
value.(x)
```

Because the joint probability is a nonlinear function of `x`, this package needs a **nonlinear solver** such as [Ipopt](https://github.com/jump-dev/Ipopt.jl), which is the solver used in the examples. 

See the [JuMP tutorials](https://jump.dev/JuMP.jl/stable/tutorials/getting_started/getting_started_with_JuMP/) for more.

## A first simple model

Find the cheapest `x` such that `P(ξ ≤ x) ≥ 0.9`:

```julia
using ChanceConstraints, JuMP, Ipopt

μ = [5.0, 2.0, 4.0, 3.0]    # Mean
Σ = [ 2.0  0.5  0.3 -0.1;   # Covariance
      0.5  2.2  0.7  0.4;
      0.3  0.7  1.6 -0.2;
     -0.1  0.4 -0.2  1.8]
p = 0.9                     # Probability level
κ = 3                       # Window length (e.g., outage duration in hours)

model = Model(Ipopt.Optimizer)
@variable(model, x[1:4] >= 0)
@objective(model, Min, sum(x))
add_JCC_QMC(model, x, [1], κ, Σ, μ, p)
optimize!(model)
```

More complete models, including a comparison of deterministic, individual and
joint formulations, are on the [Examples](examples.md) page.

## References

This package grew out of research on the economic dispatch of weakly connected mini-grids under uncertainty around solar and demand forecasts, as well as main grid outage events. The research is presented in

- Ouanes, N., *et al.* (2025). Optimizing the economic dispatch of weakly-connected mini-grids under uncertainty using joint chance constraints. *Annals of Operations Research*, 344, 499–531 (2025). https://doi.org/10.1007/s10479-024-06287-9

The methods implemented in this package build on the following theoretical
work:
- Genz, A. (1992). Numerical computation of multivariate normal probabilities. *Journal of Computational and Graphical Statistics*, 1(2), 141-149. https://doi.org/10.2307/1390838
- Van Ackooij, W., & Henrion, R. (2014). Gradient formulae for nonlinear probabilistic constraints with gaussian and gaussian-like distributions. *SIAM Journal on Optimization*, 24(4), 1864–1889. https://doi.org/10.1137/130922689
- Prékopa, A. (1995). Stochastic programming. Dordrecht: Kluwer Academic Publishers.
- Prékopa, A., & Sántai, T. (1978). Flood control reservoir system design using stochastic programming. In M. L. Balinski & C. Lemarechal (Eds.), *Mathematical programming in use* (pp. 138–151). Berlin: Springer.

The implementation relies on the following software:

- [MvNormalCDF.jl](https://github.com/PharmCat/MvNormalCDF.jl), a Julia implementation of the Genz algorithm for multivariate normal probabilities, used to evaluate the joint probability.