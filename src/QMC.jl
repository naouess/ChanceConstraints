#=
	TODO
	[ ] relax type definitions?
	[ ] add more error and debugging hints
=# 
"""	
```julia
compute_with_QMC(j::Integer, κ::Integer, Σ::AbstractMatrix, μ::AbstractVector, s::Integer, rng)
```

This function computes a separable joint probability function and its gradient using 
Quasi-Monte Carlo. It returns the following function at a first position:

```math
f(x) = \\mathbb{P}(g_i (x, ξ) ≥ 0 \\quad ∀i = j, ..., j+κ),
```

In a second position, it returns the gradient of the above function, which is computed using the Prékopa theorem as explained in Section X.

The function takes 6 arguments:
* `j` - index in Σ and μ from which the joint probability applies
* `κ` - dimension of the multivariate distribution minus 1
* `Σ` - covariance matrix
* `μ` - mean vector
* `two_sided` - boolean to describe if the probability function at hand had a two-sided inequality
* `s` - sample size
* `rng` - random number generator

The argument `two_sided` has the default value `false`.

If `s` and `rng` are not specified, they take the default values `s = 5000` and `rng = MersenneTwister(1234)`.
With the latter, we fix the random number generator `rng` to guarantee the stability of the gradients in the solver iterations.
This tempers with the precision of the computation, yet it is necessary for solver convergence, in case the probability 
function is used in an optimization problem. 
If the goal is only the numerical computation of the probability function without further integration into an optimization
problem, the argument `rng` can be unfixed by passing the value `RandomDevice()`.

"""
function compute_with_QMC(j::Integer, κ::Integer, Σ::AbstractMatrix, μ::AbstractVector, 
	two_sided::Bool = false, s::Integer = 5000, rng = MersenneTwister(1234))
	
	# Validate inputs
	@assert all(v -> v > 0, (j, κ, s)) "The parameters j, κ, s should be strictly positive."
	@assert size(Σ) == (length(μ), length(μ)) "Σ must be square with the same dimension as μ."

	idx = j:j+κ
	Σ_j = Σ[idx, idx]
	μ_j = μ[idx]
	a_j = fill(-Inf, κ+1)
  
	if two_sided
	    f1(x...) = mvnormcdf(Σ_j, vec([x[i] for i in idx]) - μ_j, # collect(x[idx])
                    vec([x[i + length(x) ÷ 2] for i in idx]) - μ_j, 
                    m = s, rng = rng)[1]
		f = f1
		e1(x...) = mvnormcdf(Σ_j, vec([x[i] for i in idx]) - μ_j,
			vec([x[i + length(x) ÷ 2] for i in idx]) - μ_j, 
			m = s, rng = rng)[2]
	    e = e1
	else
		f2(x...) = mvnormcdf(Σ_j, a_j - μ_j,
			vec([x[i] for i in idx]) - μ_j, 
			m = s, rng = rng)[1]
		f = f2
		e2(x...) = mvnormcdf(Σ_j, a_j - μ_j, 
			vec([x[i] for i in idx]) - μ_j, 
			m = s, rng = rng)[2]
		e = e2
	end 

	@assert e != 0 "Numerical issues detected! Results could be unstable or wrong. One of the reasons can be an ill-conditioned Σ matrix"

	norm_pdf(k) = exp(-(k^2) / 2) / (2 * pi)^0.5

    function ∇f(g::AbstractVector{T}, x::T...) where {T}
		idx = j:j+κ
		Σ_j = Σ[idx, idx]
		μ_j = μ[idx]
		a_j = fill(-Inf, κ+1)
		σ_j = sqrt.(diag(Σ_j, 0))

		for o in eachindex(x)
			# i = mod1(o, κ+1) # Adjust index for both sides if two_sided
			i = (o <= length(x) ÷ 2) ? o : o - length(x) ÷ 2

			if i in idx
				Σ_inv = inv(Σ_j[i-j+1, i-j+1])
				Σ_new = Σ_j - Σ_inv * Σ_j[i-j+1, :] * Σ_j[i-j+1, :]'
				μ_new = μ_j + Σ_inv * (x[o] - μ_j[i-j+1]) * Σ_j[i-j+1, :]

				sign_factor = ifelse(two_sided && o <= length(x) ÷ 2, -1, 1)

				g[o] = sign_factor * norm_pdf((x[o] - μ_j[i-j+1]) / σ_j[i-j+1]) / σ_j[i-j+1] *
									 mvnormcdf(Σ_new[1:end .!= i-j+1, 1:end .!= i-j+1], 
											   two_sided ? vec([x[k] for k in idx if k != i])-μ_new[1:end .!=i-j+1] : a_j[1:end .!=i-j+1]-μ_new[1:end .!=i-j+1],
											   two_sided ? vec([x[k + length(x) ÷ 2] for k in idx if k != i]) - μ_new[1:end .!= i-j+1] : vec([x[k] for k in idx if k != i])-μ_new[1:end .!=i-j+1], # TODO indexing here is not generalized  
											   m = s, rng = rng)[1]
			else
				g[o] = 1e-10
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
add_JCC_QMC(m::JuMP.Model, x::AbstractVector, idx::AbstractArray, κ::Integer, Σ::AbstractMatrix, μ::AbstractArray, p::Float64)
```

This function allows to add a system of joint chance constraints into a JuMP model as follows:

```math
∀ \\quad j \\text{ in} \\quad idx:

f(x) = \\mathbb{P}(g_i (x, ξ) ≥ 0 \\quad ∀i = j, ..., j+κ) ≥ p
```
To add this system of nonlinear constraints to the model, the function defines the probability function 
computed using the quasi-monte-carlo method in `compute_with_QMC` as a user-defined operator 
and provides it also with the gradient, also computed using `compute_with_QMC.

The function takes 6 arguments:
	* `m` - a JuMP model
	* `x` - a vector of decision variables
	* `idx` - set of indices that constitue the starting time of a reliability window
	* `κ` - dimension of the multivariate distribution minus 1
	* `Σ` - covariance matrix
	* `μ` - mean vector
	* `p` - probability level that has to be met
	* `two_sided` - boolean to describe if the probability function at hand had a two-sided inequality
	* `s` - sample size
	* `rng` - random number generator

The argument `two_sided` is optional and has the default value `false`.

The arguments `s` and `rng` are optional. If not specified, they take the default values `s = 5000` and `rng = MersenneTwister(1234)`.
With the latter, we fix the random number generator `rng` to guarantee the stability of the gradients in the solver iterations.
This tempers with the precision of the computation, yet it is necessary for solver convergence, in case the probability 
function is used in an optimization problem. 
If the goal is only the numerical computation of the probability function without further integration into an optimization
problem, the argument `rng` can be unfixed by passing the value `RandomDevice()`.

"""
function add_JCC_QMC(m::JuMP.Model, x::AbstractVector, idx::AbstractArray, κ::Integer, Σ::AbstractMatrix, μ::AbstractArray, p::Float64, 
	two_sided::Bool = false, s::Integer = 5000, rng = MersenneTwister(1234))
    for j in idx
		JuMP.register(m, Symbol("mvncdf_$j"), length(x), compute_with_QMC(j, κ, Σ, μ, two_sided, s, rng)[1], 
					compute_with_QMC(j, κ, Σ, μ, two_sided, s, rng)[2])
		JuMP.add_nonlinear_constraint(m, :($(Symbol("mvncdf_$j"))($(x...)) >= $(p)))

		# Non-legacy reformulation:
		# sym = add_nonlinear_operator(m, length(x), compute_with_QMC(j, κ, Σ, μ, two_sided, s, rng)[1], compute_with_QMC(j, κ, Σ, μ, two_sided, s, rng)[2]; name = Symbol("mvncdf_$j"))
        # @constraint(m, sym(x...) >= p)
	end
end