# Common mistakes or issues

This section lists possible implementation mistakes or bugs that can occur when using `ChanceConstraints.jl` and offers practical tips to deal with them.

## Notation
Throughout, ξ is the random vector, A the matrix multiplying it, and ζ = Aξ ∼ N(μ, Σ). The mean `μ` and covariance `Σ` passed to `add_JCC_QMC!` / `add_JCC_SRD!` are those of ζ = Aξ, not of ξ. The bounds are affine expressions h(x) = Bx + c (and z(x) in the two-sided case), and the constraint is $P(ζ ≤ h(x)) ≥ p$.


## Covariance matrix not positive definite
Σ must be symmetric and positive definite. Typical causes are rounding asymmetries, a covariance estimated from too few samples, or duplicated / perfectly correlated components.

You can use these tricks to check positive-definiteness and adjust your covariance matrix:

```julia
isposdef(Σ)                          # check
Σ = (Σ + Σ') / 2 + 1e-10 * I         # symmetrise and add a small jitter
```

## SRD with probabilities of at most 0.5
SRD requires the mean of ζ to be strictly inside every constraint, $h_i(x) > μ_i$. For affine constraints this holds at every point with joint probability above 1/2, so SRD requires $0.5 < p < 1$.
For $p ≤ 0.5$, which is rarely of practical relevance, use QMC instead.

`add_JCC_SRD!` adds the constraints $h_i(x) ≥ μ_i + 10^{-8}$ for all components covered by the windows. For $p > 0.5$ they practically remove no feasible point. They only keep the solver away from trial points where the SRD probability cannot be evaluated; there `ProbFunction` throws the error "h - mu must be strictly positive". If you see this error, start the solver at a point with $h(x) > μ$ (see "Starting point" below).

SRD is available only for one-sided constraints. For two-sided constraints use QMC.


## Window length: minimum κ
The window length is $κ + 1$. The smallest admissible $κ$ depends on the method:

| method | minimum κ | window length |
|---|---|---|
| `add_JCC_QMC!` | 2 | at least 3 components |
| `add_JCC_SRD!` | 1 | at least 2 components |

**$κ = 0$** (one component): the constraint is an individual chance constraint. It has a closed-form quantile, which is cheaper and exact, so use it directly (see `solve_icc` in the portfolio example). Neither method accepts it:

```julia
  @constraint(model, cash[t] >= quantile(Normal(μ[t], sqrt(Σ[t, t])), p))
```

**$κ = 1$** (two components), QMC: after the Prékopa reduction of the problem, the  covariance matrix has dimension smaller than 2 and the method, called through `MvNormalCDF` cannot be applied. The reduced matrix is large enough only from $κ = 2$ on. Use SRD for windows of two components.

## Starting point
With Ipopt, it is highly encouraged to set a starting point for the optimization.

Ipopt may fail to converge when started at a point where the chance constraint is badly violated. Start at, or near, a point where the constraint is satisfied, for example the solution of the deterministic problem (all random variables replaced by their mean). For SRD the starting point must also satisfy $h(x) > μ$, which holds at the deterministic solution whenever the constraint is strictly satisfied.
You can get a report about feasibility with `primal_feasibility_report` before calling `optimize!`. Read more on this feature here: [Checking feasibility of solutions](https://jump.dev/JuMP.jl/stable/manual/solutions/#Checking-feasibility-of-solutions).

## Bad scaling
Some solvers, like Ipopt, are sensitive to very large or very small numbers in the objective and the constraints. If the solver stalls or reports poor convergence, rescale, e.g., divide the objective by a constant and recompute the true cost afterwards.

## Slow end-phase
For some solvers, like Ipopt, dual infeasibility stalls at a certain threshhold with sampled constraints. Loosening tol (for example to 1e-5), setting acceptable_tol and acceptable_iter, and accepting `ALMOST_LOCALLY_SOLVED` as solver status might fix this issue.


## Inverted sign of the mean
The constraint is of the form $P(ζ ≤ h(x)) ≥ p$ (single-sided). If your requirement points the other way, $P(ζ ≥ h(x)) ≥ p$, rewrite it for $-ζ \sim N(-μ, Σ)$: pass $-μ$, not $μ$, together with the same $Σ$ (see the single-sided [reservoir example](examples.md#reservoir-problem-example)).

```julia
slack = [l_0 - l_min - cumx[t] for t in 1:T]    # level_t ≥ l_min  ⇔  -ξ_t ≤ slack_t
add_JCC_QMC!(model, slack, [1], T - 1, Σ, -μ, p)
```

## Input values check
Before calling `add_JCC_QMC!` / `add_JCC_SRD!`, check that:

- 0 < p < 1 (for SRD: 0.5 < p < 1);
- Σ is square, positive definite and has a matching dimension for $μ$;
- `h` has the same length as `μ`, i.e. one affine expression per component of ζ;
- every window `s:s+κ` for `s` in `idx` lies inside `1:length(μ)`;
- κ is the dimension of the window minus 1, not the number of constraints. With `two_sided = true`, the lower bounds are passed separately as the keyword argument `z`, a vector of affine expressions of the same length as `h` and `μ`. There is no stacking, and $κ$ is unchanged.


## Two-sided window too narrow
For a two-sided constraint $P(z(x) ≤ ζ ≤ h(x)) ≥ p$, the probability cannot exceed that of a window of the same width centred on the mean. If the width is too small compared to the spread of ξ, no choice of $x$ can reach $p$, and the solver reports infeasibility or returns a meaningless point (for example a joint probability of 0).
In the reservoir example, a width of $l_\text{max} - l_\text{min} = 5$ with $σ ≈ 1.73$ in the last period gives at most 0.85, so $p = 0.9$ is infeasible.

If the width is fixed (it does not depend on $x$), check it before building the model:

```julia
using MvNormalCDF

w = l_max - l_min                                             # upper - lower
d = length(μ)
best = mvnormcdf(Σ, fill(-w/2, d), fill(w/2, d))[1]           # window centred on μ
best >= p || @warn "Infeasible: the best possible probability is $best < p"
```

If the check fails, widen the window, lower $p$, or shorten the horizon.

## Crossing bounds in the two-sided case
The two-sided probability requires $z_i(x) < h_i(x)$ for all $i$. If the solver tries a point where this is violated, the probability is exactly 0. The QMC operator then returns 0 with a zero gradient and issues a warning (once per window). This is not an error: the constraint is simply violated at that point and the solver should backtrack. If the warning appears repeatedly or the solver does not recover, start from a point with $z(x) < h(x)$, or add explicit constraints such as `@constraint(model, z .+ δ .<= h)` with a small margin `δ`.

## Wrong affine expression of $x$
The vector `h` passed to `add_JCC_QMC!` / `add_JCC_SRD` is not the decision variable but the affine expression $h(x) = Bx + c$ in $P(ζ ≤ h(x)) ≥ p$. Typical mistakes are a wrong sign or direction (see "Inverted sign of the mean"), a forgotten constant, a missing cumulative sum, etc. If $A$ is not the identity, remember that $μ$ and $Σ$ must be $Aμ_ξ$ and $AΣ_ξA^\top$, not the mean and covariance of ξ. 

# Other practical tips

- **Check the result independently.** Estimate the probability at the solution with `mvnormcdf` or a large Monte Carlo sample on scenarios that were not used in the optimisation, and compare it with $p$.
- **Accuracy vs. speed.** The number of samples $s$ controls the accuracy of the probability approximation. Increase it if the empirical reliability differs noticeably from $p$.
- **Accuracy vs. speed.** The number of samples $s$ controls the accuracy of the probability approximation. Increase it if the empirical reliability differs noticeably from $p$. Both the probability and its gradient are sample estimates, and the fixed seed (`rng` for QMC, `seed` for SRD) keeps them repeatable across solver iterations.
- **Compare against the deterministic and individual-constraint models.** They show what the joint constraint buys you: the deterministic model violates the requirement often, and individual constraints do not control the joint probability.
