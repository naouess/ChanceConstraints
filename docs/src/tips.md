# Common mistakes or issues

This section lists possible implementation mistakes or bugs that can occur when using `ChanceConstraints.jl` and offers practical tips to deal with them.

## Covariance matrix not positive definite
Σ must be symmetric and positive definite. Typical causes are rounding asymmetries, a covariance estimated from too few samples, or duplicated / perfectly correlated components.

You can use these tricks to check positive-definiteness and adjust your covariance matrix:

```julia
isposdef(Σ)                          # check
Σ = (Σ + Σ') / 2 + 1e-10 * I         # symmetrise and add a small jitter
```

## SRD with probabilities below 0.5
SRD requires the mean to be strictly feasible, which in general only holds for $p > 0.5$. For smaller $p$, which is rarely of practical relevance, use QMC instead.

## Window length: minimum κ
The window length is $κ + 1$. The smallest admissible $κ$ depends on the method:

| method | minimum κ | window length |
|---|---|---|
| `add_JCC_QMC` | 2 | at least 3 components |
| `add_JCC_SRD` | 1 | at least 2 components |

**$κ = 0$** (one component): the constraint is an individual chance constraint. It has a closed-form quantile, which is cheaper and exact, so use it directly (see `solve_icc` in the portfolio example). Neither method accepts it:

```julia
  @constraint(model, cash[t] >= quantile(Normal(μ[t], sqrt(Σ[t, t])), p))
```

**$κ = 1$** (two components), QMC: after the Prékopa reduction of the problem, the  covariance matrix has dimension smaller than 2 and the method, called through `MvNormalCDF` cannot be applied. The reduced matrix is large enough only from $κ = 2$ on. Use SRD for windows of two components.

## Starting point
With Ipopt, it is highly encouraged to set a starting point for the optimization.

Ipopt may fail to converge when started at a point where the chance constraint is badly violated. Start at, or near, a point where the constraint is satisfied, for example the solution of the deterministic problem (all random variables replaced by their mean). You can get a report about feasibility with `primal_feasibility_report` before calling `optimize!`. Read more on this feature here: [Checking feasibility of solutions](https://jump.dev/JuMP.jl/stable/manual/solutions/#Checking-feasibility-of-solutions).

## Bad scaling
Some solvers, like Ipopt, are sensitive to very large or very small numbers in the objective and the constraints. If the solver stalls or reports poor convergence, rescale, e.g., divide the objective by a constant and recompute the true cost afterwards.

## Slow end-phase
For some solvers, like Ipopt, dual infeasibility stalls at a certain threshhold with sampled constraints. Loosening tol (for example to 1e-5), setting acceptable_tol and acceptable_iter, and accepting `ALMOST_LOCALLY_SOLVED` as solver status might fix this issue.


## Inverted sign of the mean
The constraint is of the form $P(ξ ≤ l(x)) ≥ p$ (single-sided). If your requirement points the other way, $P(ξ ≥ l(x)) ≥ p$, rewrite it for $-ξ \sim N(-μ, Σ)$: pass $-μ$, not $μ$, together with the same $Σ$ (see the single-sided [reservoir example](examples.md#reservoir-problem-example)).

```julia
slack = [l_0 - l_min - cumx[t] for t in 1:T]    # level_t ≥ l_min  ⇔  -ξ_t ≤ slack_t
add_JCC_QMC(model, slack, [1], T - 1, Σ, -μ, p)
```

## Input values check
Before calling `add_JCC_QMC` / `add_JCC_SRD`, check that:

- $0 < p < 1$;
- $Σ$ is square, positive definite and has a matching dimension for $μ$;
- every window `s:s+κ` for `s` in `idx` lies inside `1:length(μ)`;
- $κ$ is the dimension of the window minus 1, not the number of constraints. With `two_sided = true` the vector of affine expressions $x$ which is passed to the joint contraint, has length `2 × length(μ)`, stacked as `[lower; upper]`, but $κ$ is unchanged.

## Two-sided window too narrow
For a two-sided constraint $P(\text{lower} ≤ ξ ≤ \text{upper}) ≥ p$, the probability cannot exceed that of a window of the same width centred on the mean. If the width is too small compared to the spread of ξ, no choice of $x$ can reach $p$, and the solver reports infeasibility or returns a meaningless point (for example a joint probability of 0).
In the reservoir example, a width of $l_\text{max} - l_\text{min} = 5$ with $σ ≈ 1.73$ in the last period gives at most 0.85, so $p = 0.9$ is infeasible.

If the width is fixed (it does not depend on $x$), check it before building the model:

```julia
using MvNormalCDF

w = l_max - l_min                                              # upper - lower
d = length(μ)
best = mvnormcdf(Σ, fill(-w/2, d), fill(w/2, d))[1]           # window centred on μ
best >= p || @warn "Infeasible: the best possible probability is $best < p"
```

If the check fails, widen the window, lower $p$, or shorten the horizon.

## Wrong affine expression of $x$
The vector passed to `add_JCC_QMC` / `add_JCC_SRD` is not the decision variable but the affine expression $x$ in $P(ξ ≤ x) ≥ p$. Typical mistakes are a wrong sign or direction (see "Inverted sign of the mean"), a forgotten constant, a missing cumulative sum, etc.

# Other practical tips

- **Check the result independently.** Estimate the probability at the solution with `mvnormcdf` or a large Monte Carlo sample on scenarios that were not used in the optimisation, and compare it with $p$.
- **Accuracy vs. speed.** The number of samples $s$ controls the accuracy of the probability approximation. Increase it if the empirical reliability differs noticeably from $p$.
- **Compare against the deterministic and individual-constraint models.** They show what the joint constraint buys you: the deterministic model violates the requirement often, and individual constraints do not control the joint probability.
