"""
```julia
    compute_with_QMC(j, κ, Σ, μ; two_sided=false, s=10_000, rng=MersenneTwister(1234))
```

Compute a quasi-Monte Carlo (QMC) approximation of a joint Gaussian probability and
its gradient with respect to the deterministic bounds.

Let ``\\zeta = A\\xi \\sim \\mathcal{N}(\\mu, \\Sigma)`` and let
``I = \\{j, \\ldots, j+\\kappa\\}`` be a window of ``q = \\kappa + 1`` consecutive
components. The returned function `f` evaluates, for bounds ``h \\in \\mathbb{R}^q``,

```math
F(h) = \\mathbb{P}\\bigl(\\zeta_i \\leq h_i,\\; i \\in I\\bigr).
```

When `two_sided=true`, for bounds ``(z, h) \\in \\mathbb{R}^{2q}``,

```math
F(z, h) = \\mathbb{P}\\bigl(z_i \\leq \\zeta_i \\leq h_i,\\; i \\in I\\bigr).
```

Returns `(f, ∇f)`. The arguments of `f` are `h...` (one-sided) or
`z..., h...` (two-sided), each of length ``q``. `∇f` is the gradient of ``F``
with respect to these arguments. The ``r``-th partial derivative with respect
to ``h_r`` is the density of ``\\zeta_r`` at ``h_r`` times the probability of the
remaining components conditional on ``\\zeta_r = h_r``. The derivative with
respect to ``z_r`` has the same form with a negative sign. Both the probability
and the conditional probabilities are QMC estimates, so `∇f` is an approximation.

In the two-sided case the bounds should satisfy ``z_i < h_i`` for all ``i``.
If this is violated, `f` returns 0 and `∇f` returns the zero vector, and a
warning is issued.

# Arguments
- `j::Integer`: Starting index of the window in ``\\Sigma`` and ``\\mu``.
- `κ::Integer`: Number of additional components; the window is ``j, \\ldots, j+\\kappa``.
- `Σ::AbstractMatrix`: Covariance matrix of ``\\zeta = A\\xi``.
- `μ::AbstractVector`: Mean vector of ``\\zeta = A\\xi``.

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
    @assert size(Σ) == (length(μ), length(μ)) "Σ must be square and match μ." 

    n = length(μ)

    idx = j:(j + κ)
    @assert last(idx) <= n "j + κ exceeds dimension."

    q = length(idx)

    Σj = Matrix(Σ[idx, idx])
    μj = collect(μ[idx])

    # Check that the covariance block is usable
    @assert all(isfinite, Σj) "Σ contains non-finite values."
    @assert all(isfinite, μj) "μ contains non-finite values."

    # This will throw an error if Σj is not positive definite
    @assert isposdef(Σj) "Σj is not positive definite"

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
    # Extract the lower and upper bounds from the operator arguments
    #   one-sided: arguments are h_1, ..., h_q
    #   two-sided: arguments are z_1, ..., z_q, h_1, ..., h_q
    # ------------------------------------------------------------------

    function get_bounds(args)
        v = collect(args)

        if two_sided
            @assert length(v) == 2q "Expected 2q = $(2q) arguments."

            lower = v[1:q]
            upper = v[q + 1:2q]

            return lower, upper
        else
            @assert length(v) == q "Expected q = $q arguments."

            upper = v 
            lower = fill(-Inf, q)

            return lower, upper
        end
    end

    # ------------------------------------------------------------------
    # Probability function
    # ------------------------------------------------------------------

    function f(args...)
        lower, upper = get_bounds(args)

        if two_sided && !all(lower .< upper)
                @warn "Bounds violate z_i < h_i for some i (window starting at j = $j); " *
                      "returning probability 0." maxlog = 1  
            return 0.0
        end 

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

    function ∇f(g::AbstractVector, args...)
        lower, upper = get_bounds(args)


        # Ensure the gradient is initialized everywhere
        fill!(g, zero(eltype(g)))

        if two_sided && !all(lower .< upper)
                @warn "Bounds violate z_i < h_i for some i (window starting at j = $j); " *
                      "returning probability 0." maxlog = 1  
            return
        end 

        for r in 1:q

            σr = sqrt(Σj[r, r])

            # Derivative with respect to the upper bound h
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
                # First q variables are lower bounds z
                # Last q variables are upper bounds h

                g[q + r] += upper_derivative
            else
                # The first q variables are upper bounds h_1, ... h_q
                g[r] += upper_derivative
            end

            if two_sided
                # Derivative with respect to the lower bound z
                # It has a negative sign.
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


"""
```julia
    add_JCC_QMC!(m, h, idx, κ, Σ, μ, p; z=nothing, two_sided=false, s=10_000, rng=MersenneTwister(1234))
```
Add a system of joint chance constraints to a JuMP model.

Let ``\\zeta = A\\xi \\sim \\mathcal{N}(\\mu, \\Sigma)``, and let ``h`` (and, in the 
two-sided case, ``z``) be vectors of affine JuMP expressions in the decision 
variables, e.g. `h = B*x .+ c`. For each starting index ``j \\in \\texttt{idx}``,
with window ``I_j = \\{j, \\ldots, j+\\kappa\\}``, the added constraint is 

```math
\\mathbb{P}\\bigl(\\zeta_i \\leq h_i(x),\\; i \\in I_j\\bigr) \\geq p,
```
or, when `two_sided=true`, 

```math
\\mathbb{P}\\bigl(z_i(x) \\leq \\zeta_i \\leq h_i(x),\\; i \\in I_j\\bigr) \\geq p.
```

where ``\\xi`` is a random vector with mean ``\\mu`` and covariance matrix
``\\Sigma``. When `two_sided=true`, the probability uses the corresponding
two-sided inequalities.

The probability is approximated by quasi-Monte Carlo (QMC). For each window it
is registered as a user-defined operator of the bounds, with the gradient
supplied by `compute_with_QMC`. Because the bounds are affine, JuMP obtains the
gradient with respect to the decision variables by the chain rule.

# Arguments
- `m::JuMP.Model`: JuMP model to which the constraints are added.
- `h::AbstractVector`: Upper bounds ``h(x)``, one affine expression per component of ``\\zeta``.
- `idx::AbstractArray`: Starting indices of the windows.
- `κ::Integer`: Number of additional components in each window.
- `Σ::AbstractMatrix`: Covariance matrix of ``\\zeta = A\\xi``.
- `μ::AbstractVector`: Mean vector of ``\\zeta = A\\xi``.
- `p::Float64`: Minimum required probability level, ``p \\in (0,1)``.


# Keyword Arguments
- `z=nothing`: Lower bounds ``z(x)`` (same length as `h`); required if `two_sided=true`. 
The bounds must satisfy ``z_i(x) < h_i(x)`` at every point where the constraint is evaluated; 
otherwise the probability is set to 0 and a warning is issued.
- `s::Integer=10_000`: Number of QMC samples.
- `rng=MersenneTwister(1234)`: Random number generator for QMC sampling.
  The fixed default seed makes estimates repeatable across solver iterations,
  which can help optimization converge. For standalone probability
  evaluation with a non-fixed random source, pass `RandomDevice()`.

"""
function add_JCC_QMC!(m::JuMP.Model, h::AbstractVector, idx::AbstractArray, κ::Integer,    
                     Σ::AbstractMatrix, μ::AbstractVector, p::Float64;                     
                     z = nothing,
                     two_sided::Bool = false, s::Integer = 10_000, rng = MersenneTwister(1234))

    # ------------------------------------------
    # Validate inputs 
    # ------------------------------------------
    @assert κ >= 2 "κ must be at least 2 for QMC (window of κ+1 ≥ 3 components)."
    @assert s > 0 "The number of sample s must be positive."
    @assert size(Σ) == (length(μ), length(μ)) "Σ must be square and match μ."
    @assert length(h) == length(μ) "h must have the same length as μ."                  
    @assert 0.0 < p < 1.0 "The probability p must be in (0, 1)."                           
    @assert first(idx) >= 1 && last(idx) + κ <= length(μ) "idx .+ κ exceeds dimension."
    
    if two_sided
        @assert z !== nothing "Lower bounds z must be provided when two_sided = true."      
        @assert length(z) == length(μ) "z must have the same length as μ."
        # Each lower bound z[i] must be strictly less than h[i]: TODO add the following constraint?
        # @constraint(m, [i in eachindex(h)], z[i] <= h[i] - 1e-6)
    end                                                                                     

    # Check that the covariance is usable
    @assert all(isfinite, Σ) "Σ contains non-finite values."                                
    @assert all(isfinite, μ) "μ contains non-finite values."                                
    @assert isposdef(Σ) "Σ is not positive definite"

    for j in idx
        f, ∇f = compute_with_QMC(j, κ, Σ, μ; two_sided = two_sided, s = s, rng = rng)
        
        window = j:(j + κ)
        args = two_sided ? vcat(z[window], h[window]) : h[window]

        op = JuMP.add_nonlinear_operator(
            m, length(args), f, ∇f;
            name = Symbol("mvncdf_$j"),
        )

        @constraint(m, op(args...) >= p)
    end
    return
end