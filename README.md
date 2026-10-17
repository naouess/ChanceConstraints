# Chance Constraints

[![Build Status](https://github.com/naouess/ChanceConstraints/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/naouess/ChanceConstraints/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/naouess/ChanceConstraints.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/naouess/ChanceConstraints.jl)

This library computes multivariate probabilistic functions using two methods: Quasi-Monte Carlo (QMC), based on the A. Genz algorithm and implemented in the MvNormalCDF.jl package, and Spherical-Radial Decomposition (SRD). Both methods support efficient evaluation of multivariate probability functions and their gradients.

Additionally, the library integrates with JuMP, enabling the formulation of stochastic optimization problems with linearly separable joint chance constraints. It is compatible with nonlinear solvers such as IPOPT.

## Documentation

The documentation is available at
[ChanceConstraints.jl – Documentation](https://naouess.github.io/ChanceConstraints.jl/dev/).

Here is a first model to get started

Find the cheapest $x$ such that $P(ξ ≤ x) ≥ 0.9$:

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
More complete examples can be found in the docs page.

## Reference

The methods implemented in this package, and a comparison of their strengths and weaknesses, are described in:

> Ouanes, N., Grandón, T. (2026) ChanceConstraints.jl: A Julia Package for Chance-Constrained Programming. **LINK TO BE ADDED SOON**

This package grew out of research on the economic dispatch of weakly connected mini-grids under uncertainty around solar and demand forecasts, as well as main grid outage events. The research is presented in:

> Ouanes, N., *et al.* (2025). Optimizing the economic dispatch of weakly-connected mini-grids under uncertainty using joint chance constraints. *Annals of Operations Research*, 344, 499–531 (2025). https://doi.org/10.1007/s10479-024-06287-9

**If you use `ChanceConstraints.jl` in your research, please cite these papers.**

## License

ChanceConstraints.jl is released under the [<MIT NAME> license](LICENSE).
