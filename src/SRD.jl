#=
    ProbFunction(x, SampleOnSphere, mu)

    This function computes the probability of a given vector `x` 
    being less than or equal to a normal random variable `xi` 
    with mean `mu` and a given covariance matrix, based on a sample
    of points distributed on the unit sphere.

    TODO 

=#

function ProbFunction(x, SampleOnSphere, mu)
      
    SampleSize, M = size(SampleOnSphere)
    PROB = 0
    
    # Required gamma values for Chi-distibution
    gm = [loggamma(M/2.0), loggamma(M/2.0 + 1.0), gamma(M/2.0)]
    
    # setup gradient
    grad = zeros(size(x, 1), 1)
    
    for sample = 1:SampleSize
        
        # Select sample on sphere
        v = SampleOnSphere[sample, :]'
        
        # Compute radius r (Inner problem)
        
        # Pre-Initialization
        r = 1e99
        active = 0
        
        y = x - mu
        
        for k = 1:M
            
            # k-th component
            rstep = y[k] / v[k]
            
            if rstep > 0
                if rstep < r
                    r = rstep
                    active = k
                end
            end
            
        end
        
        # Compute chi-distribution of r
        
        # PROB
        if r > 0
            chi = chi_cdf_gamma(r, M, gm)
            PROB += chi
        end
        
        # Gradient of ProbFunction
          
        if r > 0
            
            if active > 0
                
                # active inequality for r
                gradz = zeros(M, 1)
                gradz[active] = 1
                factor = dot(gradz', v)
                factor = chi_pdf_gamma(r, M, gm) / factor
                
                # update grad_vector components
                gradx = zeros(M, 1)
                gradx[active] = -1
                grad -= factor .* gradx
                
            end
            
        end
              
    end
    
    PROB = PROB / SampleSize
    
    # Value of constraint:  probLevel - Phi(u) <= 0
    prob = PROB
    
    # Gradient of constraint: - Grad(Phi(u))
    grad = grad / SampleSize
        
    return (prob, grad)
end

#=
	TODO
	[ ] relax type definitions?
	[ ] add more error and debugging hints
=# 
"""	
```julia
compute_with_SRD(w::Integer, κ::Integer, Σ::AbstractMatrix, μ::AbstractVector, s::Integer, rng)
```

This function computes a separable joint probability function and its gradient using 
Spherical Radial Decomposition. It returns an array consisting of the following function in its first element:

```math
f(x) = \\mathbb{P}(g_i (x, ξ) ≥ 0 \\quad ∀i = j, ..., j+κ),
```

As a second element, it returns the gradient of the above function, which is computed using the Prékopa theorem as explained in Section X.

The function takes 6 arguments:
* `w` - index in Σ and μ from which the joint probability applies
* `κ` - dimension of the multivariate distribution minus 1
* `Σ` - covariance matrix
* `μ` - mean vector
* `s` - sample
* `rng` - random number generator

If `s` and `rng` are not specified, they take the default values `s = 5000` and `rng = MersenneTwister(1234)`.
With the latter, we fix the random number generator `rng` to guarantee the stability of the gradients in the solver iterations.
This tempers with the precision of the computation, yet it is necessary for solver convergence, in case the probability 
function is used in an optimization problem. 
If the goal is only the numerical computation of the probability function without further integration into an optimization
problem, the argument `rng` can be unfixed by passing the value `RandomDevice()`

"""
function compute_with_SRD(w::Integer, κ::Integer, Σ::AbstractMatrix, μ::AbstractVector, 
    s::Integer = 5000, rng = MersenneTwister(1234))

    #  Validate inputs
	@assert all(v -> v > 0, (w, κ, s)) "The parameters w, κ, s should be strictly positive."
	@assert size(Σ) == (length(μ), length(μ)) "Σ must be square and match the dimension of μ."

    # Set dimension of each probability constraint
    dim_x = length(μ)
    dim_cons = κ + 1
    SampleSize = s

    # Compute cholesky of covariance
    L = cholesky(Σ, check=true).L

    # Compute Sample on the sphere
    # SampleOnSphere = randn(rng, SampleSize, dim_cons * (dim_x - dim_cons + 1))
    SampleOnSphere = randn(SampleSize, dim_cons * (dim_x - dim_cons + 1))

    for j in 1:dim_x-dim_cons+1
        i = dim_cons * (j-1) + 1
        for n = 1:SampleSize
            # Normalize sample (projection to sphere)
            v = SampleOnSphere[n, i:i+dim_cons-1]'
            v = v / norm(v)

            # Transformation with cholesky of covariance
            v = L[j:j+dim_cons-1, j:j+dim_cons-1] * v'

            # save result
            SampleOnSphere[n, i:i+dim_cons-1] = v'
        end 
    end

    k = (1+κ) * (w-1) + 1

    f(x...) = ProbFunction(vec([x[r] for r in w:w+κ]), SampleOnSphere[:, k:k+κ], μ[w:w+κ])[1]

    function ∇f(g::AbstractVector{T}, x::T...) where {T}
        for i in eachindex(x)
            if i in w:w+κ
                g[i] = ProbFunction(vec([x[r] for r in w:w+κ]), SampleOnSphere[:, k:k+κ], μ[w:w+κ])[2][i-w+1]
            else 
                g[i] = 0.00000
            end 
        end
        return
    end
    return f, ∇f
end 

#=
	TODO
	[ ] debug non-legacy formulation (see below)
=# 
"""
```julia
add_JCC_SRD(m::JuMP.Model, x::AbstractVector, idx::AbstractArray, κ::Integer, Σ::AbstractMatrix, μ::AbstractArray, p::Float64)
```

This function allows to add a system of joint chance constraints into a JuMP model as follows:

```math

∀ \\quad j \\text{ in} \\quad idx:

f(x) = \\mathbb{P}(g_i (x, ξ) ≥ 0 \\quad ∀i = j, ..., j+κ) ≥ p
```

To add this system of nonlinear constraints to the model, the function defines the probability function 
computed using the spherical radial decomposition method in `compute_with_SRD` as a user-defined operator 
and provides it also with the gradient, also computed using `compute_with_SRD.

The function takes 6 arguments:
    * `m` - a JuMP model
    * `x` - a vector of decision variables
    * `idx` - set of indices that constitue the starting time of a reliability window
    * `κ` - dimension of the multivariate distribution minus 1
    * `Σ` - covariance matrix
    * `μ` - mean vector
    * `p` - probability level that has to be met
	* `s` - sample size
	* `rng` - random number generator

The arguments `s` and `rng` are optional. If not specified, they take the default values `s = 5000` and `rng = MersenneTwister(1234)`.
With the latter, we fix the random number generator `rng` to guarantee the stability of the gradients in the solver iterations.
This tempers with the precision of the computation, yet it is necessary for solver convergence, in case the probability 
function is used in an optimization problem. 
If the goal is only the numerical computation of the probability function without further integration into an optimization
problem, the argument `rng` can be unfixed by passing the value `RandomDevice()`.

"""
function add_JCC_SRD(m::JuMP.Model, x::AbstractVector, idx::AbstractArray, κ::Integer, Σ::AbstractMatrix, μ::AbstractArray, p::Float64, 
    s::Integer = 5000, rng = MersenneTwister(1234))
    for j in idx
		JuMP.register(m, Symbol("srd_prob_$(x)_$j"), length(x), compute_with_SRD(j, κ, Σ, μ, s, rng)[1], 
                    compute_with_SRD(j, κ, Σ, μ, s, rng)[2])
        JuMP.add_nonlinear_constraint(m, :($(Symbol("srd_prob_$(x)_$j"))($(x...)) >= $(p)))

		# Non-legacy reformulation:
		# sym = add_nonlinear_operator(m, length(x), compute_with_SRD(j, κ, Σ, μ, s, rng)[1], compute_with_SRD(j, κ, Σ, μ, s, rng)[2]; name = Symbol("srd_prob_$(x)_$j"))
        # @constraint(m, sym(x...) >= p)
    end
end