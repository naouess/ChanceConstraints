"""
```julia
    compute_with_QMC(j, κ, Σ, μ; two_sided=false, s=10_000, rng=MersenneTwister(1234))
```

Compute a quasi-Monte Carlo (QMC) approximation of a joint probability and
its gradient.

Let ``\\xi`` be a random vector with mean ``\\mu`` and covariance matrix
``\\Sigma``. For the indices ``i = j, \\ldots, j+\\kappa``, the one-sided
probability is

```math
f(x) = \\mathbb{P}\\bigl(x_i \\geq \\xi_i,\\; i = j, \\ldots, j+\\kappa\\bigr).
```

When `two_sided=true`, the probability is computed using the corresponding
two-sided inequalities.

Returns `(f, ∇f)`, where `f` is the estimated probability and `∇f` is its
gradient with respect to ``x``. The gradient is computed using the Prékopa
theorem, as described in Section X.

# Arguments
- `j::Integer`: Starting index of the components in ``\\Sigma`` and ``\\mu``.
- `κ::Integer`: Number of additional components. The joint event includes
  indices ``j, \\ldots, j+\\kappa``.
- `Σ::AbstractMatrix`: Covariance matrix of ``\\xi``.
- `μ::AbstractVector`: Mean vector of ``\\xi``.

# Keyword Arguments
- `two_sided::Bool=false`: Whether to use two-sided inequalities.
- `s::Integer=10_000`: Number of QMC samples.
- `rng=MersenneTwister(1234)`: Random number generator for sampling. The
  fixed default seed makes estimates repeatable across solver iterations,
  which can help optimization converge. For standalone probability
  evaluation with a non-fixed random source, pass `RandomDevice()`.

"""
function compute_with_QMC(
    j::Integer,
    κ::Integer,
    Σ::AbstractMatrix,
    μ::AbstractVector;
    two_sided::Bool = false,
    s::Integer = 10_000,
    rng = MersenneTwister(1234),
)

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------

    @assert j >= 1 "j must be positive."
    @assert κ >= 0 "κ must be nonnegative."
    @assert s > 0 "s must be positive."

    n = length(μ)

    @assert size(Σ) == (n, n) "Σ must be square and match μ."

    idx = j:(j + κ)
    @assert first(idx) >= 1 && last(idx) <= n "j + κ exceeds dimension."

    q = length(idx)

    Σj = Matrix(Σ[idx, idx])
    μj = collect(μ[idx])

    # Check that the covariance block is usable
    @assert isfinite.(Σj) |> all "Σ contains non-finite values."
    @assert isfinite.(μj) |> all "μ contains non-finite values."

    # This will throw an error if Σj is not positive definite
    F = cholesky(Symmetric(Σj))

    # ------------------------------------------------------------------
    # Use a deterministic RNG stream for each function evaluation.
    #
    # This is important for nonlinear optimization: the function should
    # not change randomly when evaluated at the same x.
    # ------------------------------------------------------------------

    function local_rng()
        return deepcopy(rng)
    end

    # ------------------------------------------------------------------
    # Standard normal density
    # ------------------------------------------------------------------

    normal_pdf(z) = exp(-0.5 * z^2) / sqrt(2π)

    # ------------------------------------------------------------------
    # Extract the relevant components of x
    # ------------------------------------------------------------------

    function get_bounds(x)
        xvec = collect(x)

        if two_sided
            @assert length(xvec) == 2q

            lower = collect(view(xvec, 1:q))
            upper = collect(view(xvec, q + 1:2q))

            return lower, upper
        else
            @assert length(xvec) >= last(idx)

            upper = collect(view(xvec, idx))
            lower = fill(-Inf, q)

            return lower, upper
        end
    end

    # ------------------------------------------------------------------
    # Probability function
    # ------------------------------------------------------------------

    function f(x...)
        lower, upper = get_bounds(x)

        # Convert bounds to deviations from μ
        lower_dev = lower .- μj
        upper_dev = upper .- μj

        result = mvnormcdf(
            Σj,
            lower_dev,
            upper_dev;
            m = s,
            rng = local_rng(),
        )

        return result[1]
    end

    # ------------------------------------------------------------------
    # Conditional multivariate normal CDF
    #
    # Computes the probability of all remaining variables being within
    # [lower, upper], conditional on variable r equaling x_r.
    # ------------------------------------------------------------------

    function conditional_cdf(
        lower,
        upper,
        r::Int,
        xr,
    )
        keep = [k for k in 1:q if k != r]

        # If q == 1, there are no remaining variables
        if isempty(keep)
            return 1.0
        end

        σrr = Σj[r, r]

        @assert σrr > 0 "Σ[$r,$r] must be positive."

        σr = Σj[keep, r]
        Σcond = Σj[keep, keep] - (σr * transpose(σr)) / σrr

        μcond = μj[keep] .+ (σr / σrr) .* (xr - μj[r])

        lower_cond = lower[keep] .- μcond
        upper_cond = upper[keep] .- μcond

        return mvnormcdf(
            Σcond,
            lower_cond,
            upper_cond;
            m = s,
            rng = local_rng(),
        )[1]
    end

    # ------------------------------------------------------------------
    # Gradient
    # ------------------------------------------------------------------

    function ∇f(g::AbstractVector, x...)
        lower, upper = get_bounds(x)

        # Ensure the gradient is initialized everywhere
        fill!(g, zero(eltype(g)))

        for r in 1:q

            σr = sqrt(Σj[r, r])

            # Derivative with respect to the upper bound
            upper_conditional = conditional_cdf(
                lower,
                upper,
                r,
                upper[r],
            )

            upper_derivative =
                normal_pdf((upper[r] - μj[r]) / σr) / σr *
                upper_conditional

            if two_sided
                # First q variables are lower bounds
                # Last q variables are upper bounds

                g[q + r] += upper_derivative
            else
                # The first q variables are upper bounds at positions idx[r]
                g[idx[r]] += upper_derivative
            end

            if two_sided

                # Derivative with respect to the lower bound.
                # It has a negative sign because increasing the lower
                # bound reduces the probability.
                lower_conditional = conditional_cdf(
                    lower,
                    upper,
                    r,
                    lower[r],
                )

                lower_derivative =
                    normal_pdf((lower[r] - μj[r]) / σr) / σr *
                    lower_conditional

                g[r] -= lower_derivative
            end
        end

        return
    end

    return f, ∇f
end

#=
	TODO
    [ ] Multiple windows. If idx has more than one entry, 
        every window currently receives the same full x. 
        In two-sided mode that only makes sense for a single window. 
        For several windows, each needs its own 2q slice.
=# 

"""
```julia
    add_JCC_QMC(m, x, idx, κ, Σ, μ, p; two_sided=false, s=10_000, rng=MersenneTwister(1234))
```
Add a system of joint chance constraints to a JuMP model.

For each starting index ``j \\in \\texttt{idx}``, the added constraint is

```math
\\mathbb{P}\\bigl(x_i \\geq \\xi_i,\\; i = j, \\ldots, j+\\kappa\\bigr) \\geq p,
```

where ``\\xi`` is a random vector with mean ``\\mu`` and covariance matrix
``\\Sigma``. When `two_sided=true`, the probability uses the corresponding
two-sided inequalities.

The probability is approximated using quasi-Monte Carlo (QMC). The function
registers it as a user-defined operator in the JuMP model and supplies the
gradient computed by `compute_with_QMC`.

# Arguments
- `m::JuMP.Model`: JuMP model to which the constraints are added.
- `x::AbstractVector`: Decision variables used in the constraints.
- `idx::AbstractArray`: Starting indices of the reliability windows.
- `κ::Integer`: Number of additional components in each window; each
  probability covers indices ``j, \\ldots, j+\\kappa``.
- `Σ::AbstractMatrix`: Covariance matrix of ``\\xi``.
- `μ::AbstractArray`: Mean vector of ``\\xi``.
- `p::Float64`: Minimum required probability level.

# Keyword Arguments
- `two_sided::Bool=false`: Whether to use two-sided inequalities.
- `s::Integer=10_000`: Number of QMC samples.
- `rng=MersenneTwister(1234)`: Random number generator for QMC sampling.
  The fixed default seed makes estimates repeatable across solver iterations,
  which can help optimization converge. For standalone probability
  evaluation with a non-fixed random source, pass `RandomDevice()`.

"""
function add_JCC_QMC(m::JuMP.Model, x::AbstractVector, idx::AbstractArray, κ::Integer, 
                     Σ::AbstractMatrix, μ::AbstractArray, p::Float64; 
	                 two_sided::Bool = false, s::Integer = 10_000, rng = MersenneTwister(1234))

    @assert s > 0 "ERROR: s=$s must be positive (check keyword passing)"

    for j in idx
        f, ∇f = compute_with_QMC(j, κ, Σ, μ; two_sided = two_sided, s = s, rng = rng)

        op = JuMP.add_nonlinear_operator(
            m, length(x), f, ∇f;
            name = Symbol("mvncdf_$j"),
        )

        @constraint(m, op(x...) >= p)
    end
    return
end