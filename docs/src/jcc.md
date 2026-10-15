# Functions

## Computing probability functions

### Using Quasi-Monte-Carlo

```@docs
compute_with_QMC
```

### Using Spherical Radial Decomposition

**Restriction for SRD.** The spherical-radial decomposition method requires
`p > 1/2`, and the mean `μ` must be a strictly feasible point (guaranteed through an implementation of the lower bound `x ≥ μ + δ` within the SRD functions). 

For smaller probability levels – which are rarely of practical relevance, use the QMC method.

```@docs
compute_with_SRD
```

## Adding chance constraints to a JuMP model

### Using Quasi-Monte-Carlo

```@docs
add_JCC_QMC
```

### Using Spherical Radial Decomposition
```@docs
add_JCC_SRD
```