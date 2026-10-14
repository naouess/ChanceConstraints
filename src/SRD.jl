#=
```julia
    ProbFunction(x, SampleOnSphere, mu)
```

Estimate the probability that each component of ``x`` is greater than or equal
to the corresponding component of a normally distributed random vector
``\\xi``:

```math
\\mathbb{P}(x_i \\geq \\xi_i \\text{ for all } i).
```

The estimate is computed using `SampleOnSphere`, a sample of points on the
unit sphere, and a normal random vector with mean ``\\mu`` and the covariance
structure used to construct the samples.

# Arguments
- `x`: Vector of threshold values.
- `SampleOnSphere`: Sample of points on the unit sphere used in the
  probability estimate.
- `mu`: Mean vector of the normal random vector.

=#

function ProbFunction(x, SampleOnSphere, mu)

    SampleSize, M = size(SampleOnSphere)

    @assert length(x) == M
    @assert length(mu) == M
    @assert SampleSize > 0

    y = x .- mu

    if any(y .<= 1e-10) 
        error(
            "ProbFunction(): x - mu must be strictly positive " *
            "for the current radial formulation."
        )
    end

    # Gamma-related constants used by the chi CDF/PDF functions
    gm = [
        loggamma(M / 2),
        loggamma(M / 2 + 1),
        gamma(M / 2)
    ]

    # Probability Phi(x) and its gradient
    phi = zero(eltype(x))
    grad_phi = zeros(eltype(x), M)

    # Tolerance for zero components of v
    vtol = sqrt(eps(Float64))

    for sample in 1:SampleSize

        # Always use a one-dimensional vector, not a 1 × M matrix
        v = vec(SampleOnSphere[sample, :])

        rb = Inf
        active = 0

        # Compute the radial upper bound:
        #
        # rb = min_k (x[k] - mu[k]) / v[k],
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

                # r_b = (x[active] - mu[active]) / v[active]
                # ∂r_b / ∂x[active] = 1 / v[active]
                pdf = chi_pdf_gamma(rb, M, gm)

                grad_phi[active] += pdf / v[active]
            end

        else
            # No finite radial upper bound: the whole radial ray is feasible
            phi += one(eltype(x))
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
    @assert size(SampleOnSphere, 2) >=
            dim_cons * (d - dim_cons + 1)

    L = cholesky(Symmetric(Σ), check = true).L

    transformed = copy(SampleOnSphere)

    for j in 1:(d - dim_cons + 1)
        first_col = dim_cons * (j - 1) + 1
        last_col = first_col + dim_cons - 1

        Lblock = L[j:j+dim_cons-1, j:j+dim_cons-1]

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
Compute a spherical radial decomposition (SRD) approximation of a joint
probability and its gradient.

For the indices ``i = w, \\ldots, w+\\kappa``, the probability is

```math
f(x) = \\mathbb{P}\\bigl(x_i \\geq \\xi_i,\\; i = w, \\ldots, w+\\kappa\\bigr),
```

where the random-variable samples used in the SRD calculation are provided
by `transformed_samples`, and ``\\mu`` is the mean vector.

Returns `(f, ∇f)`, where `f` is the estimated probability and `∇f` is its
gradient with respect to ``x``. The gradient is computed using the Prékopa
theorem, as described in Section X.

# Arguments
- `w::Integer`: Starting index of the components in the joint event.
- `κ::Integer`: Number of additional components in the event; the event
  covers indices ``w, \\ldots, w+\\kappa``.
- `transformed_samples::AbstractMatrix`: Transformed samples used by the
  SRD calculation.
- `μ::AbstractVector`: Mean vector used in the probability calculation.

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
    @assert size(transformed_samples, 2) >=
            dim_cons * (d - dim_cons + 1)

    first_col = dim_cons * (w - 1) + 1
    last_col = first_col + dim_cons - 1

    sample_window = transformed_samples[:, first_col:last_col]
    μ_window = μ[w:w+κ]

    function f(x...)
        x_window = collect(x[w:w+κ])

        return ProbFunction(
            x_window,
            sample_window,
            μ_window
        )[1]
    end

    function ∇f(g::AbstractVector{T}, x::T...) where {T}
        x_window = collect(x[w:w+κ])

        gradient = ProbFunction(
            x_window,
            sample_window,
            μ_window
        )[2]

        fill!(g, zero(T))

        for i in w:(w + κ)
            g[i] = gradient[i - w + 1]
        end

        return
    end

    return f, ∇f
end


"""
```julia
    add_JCC_SRD(model, x, idx, κ, Σ, μ, p; s=100_000, seed=1234)
```

Add a system of joint chance constraints to a JuMP model using spherical
radial decomposition (SRD).

For each starting index ``j \\in \\texttt{idx}``, the added constraint is

```math
\\mathbb{P}\\bigl(x_i \\geq \\xi_i,\\; i = j, \\ldots, j+\\kappa\\bigr) \\geq p,
```

where ``\\xi`` is a random vector with mean ``\\mu`` and covariance matrix
``\\Sigma``.

The probability is computed using `compute_with_SRD`. The function registers
it as a user-defined operator in the JuMP model and supplies its gradient,
also computed using `compute_with_SRD`.

# Arguments
- `model::JuMP.Model`: JuMP model to which the constraints are added.
- `x::AbstractVector`: Decision variables used in the constraints.
- `idx::AbstractVector`: Starting indices of the reliability windows.
- `κ::Integer`: Number of additional components in each window; each
  probability covers indices ``j, \\ldots, j+\\kappa``.
- `Σ::AbstractMatrix`: Covariance matrix of ``\\xi``.
- `μ::AbstractVector`: Mean vector of ``\\xi``.
- `p::Real`: Minimum required probability level.

# Keyword Arguments
- `s::Integer=100_000`: Number of samples used by the SRD computation.
- `seed::Integer=1234`: Seed used for the random-number generation. The
  default makes the estimates repeatable across solver iterations, which
  can help optimization converge.

"""
function add_JCC_SRD(
        model::JuMP.Model,
        x::AbstractVector,
        idx::AbstractVector,
        κ::Integer,
        Σ::AbstractMatrix,
        μ::AbstractVector,
        p::Real;
        s::Integer = 100_000,
        seed::Integer = 1234
    )

    @constraint(model, [i in eachindex(x)], x[i] >= μ[i] + 1e-8)

    d = length(μ)
    dim_cons = κ + 1

    rng = MersenneTwister(seed)

    # Generate raw samples exactly once
    raw_samples = randn(
        rng,
        s,
        dim_cons * (d - dim_cons + 1)
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

        op = JuMP.add_nonlinear_operator(
            model, length(x), f, ∇f;
            name = Symbol("srd_prob_$j"),
        )

        @constraint(model, op(x...) >= p)

    end

    return transformed_samples
end
