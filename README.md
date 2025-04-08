# JointChance

This library computes multivariate probabilistic functions and their gradients using two methods: Quasi-Monte Carlo (QMC), based on the A. Genz algorithm and implemented in the MvNormalCDF.jl package, and Spherical-Radial Decomposition (SRD). Both methods support efficient evaluation of multivariate probability functions and their gradients.

Additionally, the library integrates with JuMP, enabling the formulation of separable joint chance constraints in optimization problems under uncertainty. It is compatible with nonlinear solvers such as IPOPT.

[![Build Status](https://github.com/naouess/JointChance/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/naouess/JointChance/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/naouess/JointChance.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/naouess/JointChance.jl)
