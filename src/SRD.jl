"""
```julia
    ProbFunction(h, SampleOnSphere, mu)
```

Estimate the probability that each component of the Gaussian vector
``\\zeta = A\\xi \\sim \\mathcal{N}(\\mu, \\Sigma)`` is smaller than or equal to
the corresponding bound ``h_i``:

```math
\\mathbb{P}(\\zeta_i \\leq h_i \\text{ for all } i).
```

The estimate uses the spherical-radial decomposition. Each row of
`SampleOnSphere` is a direction ``L v``, where ``v`` is a point on the unit sphere 
and ``L`` is the Cholesky factor of ``\\Sigma``. For each direction, the radial
distance to the boundary of the region ``\\{\\zeta \\leq h\\}`` is computed, and the
probability is the average of the chi CDF at these distances. This requires
``h_i > \\mu_i`` for all ``i``.


# Arguments
- `h`: Vector of upper bounds, ``h_i > \\mu_i``.
- `SampleOnSphere`: Transformed samples ``L v`` used in the probability estimate.
- `mu`: Mean vector of ``\\zeta``.
"""
function ProbFunction(h, SampleOnSphere, mu)

    SampleSize, M = size(SampleOnSphere)

    @assert length(h) == M
    @assert length(mu) == M
    @assert SampleSize > 0

    y = h .- mu

    if any(y .<= 1e-10) 
        error(
            "ProbFunction(): h - mu must be strictly positive " *
            "for the current radial formulation."
        )
    end

    # Gamma-related constants used by the chi CDF/PDF functions
    gm = [
        loggamma(M / 2),
        loggamma(M / 2 + 1),
        gamma(M / 2)
    ]

    # Probability Phi(h) and its gradient
    phi = zero(eltype(h))
    grad_phi = zeros(eltype(h), M)

    # Tolerance for zero components of v
    vtol = sqrt(eps(Float64))

    for sample in 1:SampleSize

        # Always use a one-dimensional vector, not a 1 × M matrix
        v = vec(SampleOnSphere[sample, :])

        rb = Inf
        active = 0

        # Compute the radial upper bound:
        #
        # rb = min_k (h[k] - mu[k]) / v[k],
        # considering only positive radial upper bounds.
        for k in 1:M

            if abs(v[k]) <= vtol
                continue
            end

            rstep = y[k] / v[k]

            if rstep > 0 && rstep < rb
                rb = rstep
                active = k
            end
        end

        if isfinite(rb)

            # Probability contribution for this direction
            chi = chi_cdf_gamma(rb, M, gm)
            phi += chi

            if active != 0

                # r_b = (h[active] - mu[active]) / v[active]
                # ∂r_b / ∂h[active] = 1 / v[active]
                pdf = chi_pdf_gamma(rb, M, gm)

                grad_phi[active] += pdf / v[active]
            end

        else
            # No finite radial upper bound: the whole radial ray is feasible
            phi += one(eltype(h))
        end
    end

    # Average over sampled sphere directions
    phi /= SampleSize
    grad_phi ./= SampleSize

    return phi, grad_phi
end

function transform_srd_samples(
    SampleOnSphere::AbstractMatrix,
    κ::Integer,
    Σ::AbstractMatrix
    )
    d = size(Σ, 1)
    dim_cons = κ + 1

    @assert size(Σ) == (d, d)
    @assert size(SampleOnSphere, 2) >= dim_cons
    # @assert size(SampleOnSphere, 2) >=  dim_cons * (d - dim_cons + 1)


    # number of windows present in the matrix (replaces the Cholesky of the full Σ)
    nwin = min(size(SampleOnSphere, 2) ÷ dim_cons, d - dim_cons + 1)

    transformed = copy(SampleOnSphere)

    for j in 1:nwin
        first_col = dim_cons * (j - 1) + 1
        last_col = first_col + dim_cons - 1

        # Cholesky factor of the window's own covariance
        Lblock = cholesky(Symmetric(Σ[j:j+dim_cons-1, j:j+dim_cons-1])).L

        for n in axes(transformed, 1)
            v = copy(transformed[n, first_col:last_col])

            norm_v = norm(v)
            @assert norm_v > 0

            v ./= norm_v

            transformed[n, first_col:last_col] = (Lblock * v)'
        end
    end

    return transformed
end

"""
```julia
    compute_with_SRD(w, κ, transformed_samples, μ)
```
Build a spherical radial decomposition (SRD) approximation of a joint Gaussia
probability and its gradient with respect to the deterministic bounds.      

For the indices ``i = w, \\ldots, w+\\kappa``, the probability is

```math
F(h) = \\mathbb{P}\\bigl(\\zeta_i \\leq h_i,\\; i \\in I\\bigr),
```

where the samples used in the SRD calculation are provided
by `transformed_samples`, and ``\\mu`` is the mean vector of ``\\zeta``. 

Returns `(f, ∇f)`. The arguments of `f` are `h...`, of length ``q``. `∇f` is the
gradient of ``F`` with respect to these arguments. For each sampled direction, the
derivative is the chi density at the radial distance to the boundary, times the
derivative of that distance with respect to the active bound. The bounds must satisfy
``h_i > \\mu_i``.

# Arguments
- `w::Integer`: Starting index of the window.
- `κ::Integer`: Number of additional components in the window; the window 
  covers indices ``w, \\ldots, w+\\kappa``.
- `transformed_samples::AbstractMatrix`: Transformed samples used by the
  SRD calculation.
- `μ::AbstractVector`: Mean vector of ``\\zeta = A\\xi``.
"""
function compute_with_SRD(
        w::Integer,
        κ::Integer,
        transformed_samples::AbstractMatrix,
        μ::AbstractVector
    )

    d = length(μ)
    dim_cons = κ + 1

    @assert 1 <= w <= d - κ
    # the matrix only has to contain the blocks up to window w
    @assert size(transformed_samples, 2) >= dim_cons * w

    first_col = dim_cons * (w - 1) + 1
    last_col = first_col + dim_cons - 1

    sample_window = transformed_samples[:, first_col:last_col]
    μ_window = μ[w:w+κ]

    function f(h...)
        h_window = collect(h)

        return ProbFunction(
            h_window,
            sample_window,
            μ_window
        )[1]
    end

    function ∇f(g::AbstractVector{T}, h::T...) where {T}

        @assert length(h) == dim_cons "Expected q = $dim_cons arguments."
        h_window = collect(h)

        gradient = ProbFunction(
            h_window,
            sample_window,
            μ_window
        )[2]

        fill!(g, zero(T))

        for r in 1:dim_cons
            g[r] = gradient[r]
        end

        return
    end

    return f, ∇f
end


"""
```julia
    add_JCC_SRD!(model, h, idx, κ, Σ, μ, p; s=100_000, seed=1234)
```

Add a system of joint chance constraints to a JuMP model using spherical
radial decomposition (SRD).

Let ``\\zeta = A\\xi \\sim \\mathcal{N}(\\mu, \\Sigma)``, and let ``h`` be a vector of 
affine JuMP expressions in the decision variables, e.g. `h = B*x .+ c`. For each
starting index ``j \\in \\texttt{idx}``, with window 
``I_j = \\{j, \\ldots, j+\\kappa\\}``, the added constraint is

```math
\\mathbb{P}\\bigl(\\zeta_i \\leq h_i(x),\\; i \\in I_j\\bigr) \\geq p.
```

Only the one-sided formulation is available. The SRD requires ``h_i(x) > \\mu_i``,
so the constraints ``h_i(x) \\geq \\mu_i + 10^{-8}`` are added for all components 
covered by the windows. 

The probability is computed using `compute_with_SRD`. For each window it is 
registered as a user-defined operator of the bounds, with the gradient supplied by
`compute_with_SRD`. Because the bounds are affine, JuMP obtains the gradient with 
respect to the decision variables by the chain rule. 


# Arguments
- `model::JuMP.Model`: JuMP model to which the constraints are added.
- `h::AbstractVector`: Upper bounds ``h(x)``, one affine expression per component
  of ``\\zeta``. 
- `idx::AbstractVector`: Starting indices of the windows.
- `κ::Integer`: Number of additional components in each window.
- `Σ::AbstractMatrix`: Covariance matrix of ``\\zeta = A\\xi``.
- `μ::AbstractVector`: Mean vector of ``\\zeta = A\\xi``. 
- `p::Real`: Minimum required probability level, ``p \\in (0.5, 1)``. 

# Keyword Arguments
- `s::Integer=100_000`: Number of samples used by the SRD computation.
- `seed::Integer=1234`: Seed used for the random-number generation. The
  default makes the estimates repeatable across solver iterations, which
  can help optimization converge.

"""
function add_JCC_SRD!(
        model::JuMP.Model,
        h::AbstractVector,
        idx::AbstractVector,
        κ::Integer,
        Σ::AbstractMatrix,
        μ::AbstractVector,
        p::Real;
        s::Integer = 100_000,
        seed::Integer = 1234
    )

    # ------------------------------------------
    # Validate inputs 
    # ------------------------------------------
    @assert κ >= 1 "κ must be at least 1 (window of κ+1 components)." # TODO check here
    @assert s > 0 "The number of sample s must be positive."
    @assert size(Σ) == (length(μ), length(μ)) "Σ must be square and match μ."
    @assert 0.5 < p < 1.0 "The probability p must be between 0.5 and 1"
    @assert minimum(idx) >= 1 && maximum(idx) + κ <= length(μ) "idx .+ κ exceeds dimension."

    # Check that the covariance is usable
    @assert isfinite.(Σ) |> all "Σ contains non-finite values."
    @assert isfinite.(μ) |> all "μ contains non-finite values."
    @assert isposdef(Σ) "Σ is not positive definite"

    # lower bound only for the components covered by the windows
    covered = falses(length(μ))
    for j in idx
        covered[j:j+κ] .= true
    end
    @constraint(model, [i in findall(covered)], h[i] >= μ[i] + 1e-8)

    d = length(μ)
    dim_cons = κ + 1

    rng = MersenneTwister(seed)

    # Generate raw samples exactly once
    # only for the blocks up to the last requested window
    raw_samples = randn(
        rng,
        s,
        dim_cons * maximum(idx)
    )

    # Transform raw samples exactly once
    transformed_samples = transform_srd_samples(
        raw_samples,
        κ,
        Σ
    )

    for j in idx
        f, ∇f = compute_with_SRD(
            j,
            κ,
            transformed_samples,
            μ
        )

        window = j:(j + κ)
        args = h[window]

        op = JuMP.add_nonlinear_operator(
            model, length(args), f, ∇f; 
            name = Symbol("srd_prob_$j"),
        )

        @constraint(model, op(args...) >= p)

    end

    return # transformed_samples
end
