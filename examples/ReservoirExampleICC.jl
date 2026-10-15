using Distributions, LinearAlgebra
using JuMP, Ipopt

T = 3
l_0, l_min, l_max = 5.0, 5.0, 12.0
p = 0.9
price = [3.0, 2.0, 1.0]     # decreasing: releasing early pays more


β_mean = fill(10.0, T)      # independent yearly inflows with these means …
β_var  = fill(0.9, T)         # … and variances
μ = cumsum(β_mean)
Σ = [sum(β_var[1:min(i, j)]) for i in 1:T, j in 1:T]

z = quantile(Normal(), p)
σ = sqrt.(diag(Σ))

model = Model(Ipopt.Optimizer)
set_silent(model)

@variable(model, 0 <= x[1:T] <= 30)
@objective(model, Max, sum(price[t] * x[t] for t in 1:T))

cumx = [sum(x[1:t]) for t in 1:T]
@constraint(model, [t = 1:T], cumx[t] <= l_0 - l_min + μ[t] - z * σ[t])

optimize!(model)
x_ind = value.(x)               # [8.78, 9.50, 9.61]
objective_value(model)          # ≈ 54.96

#-------------------------------------------------------
# Two-sided
#-------------------------------------------------------

w = l_max - l_min
σ = sqrt.(cumsum(β_var))

# largest centre offset m with P(window of width w, centred m away from the mean) ≥ p
function half_width(σ, w, p)
    prob(m) = cdf(Normal(0, σ), m + w/2) - cdf(Normal(0, σ), m - w/2)
    prob(0) < p && return NaN        # window too narrow for this p
    lo, hi = 0.0, w/2 + 10σ
    for _ in 1:100
        mid = (lo + hi) / 2
        prob(mid) >= p ? (lo = mid) : (hi = mid)
    end
    return lo
end

δ = [half_width(σ[t], w, p) for t in 1:T]
any(isnan, δ) && error("two-sided ICC infeasible: widen l_max - l_min or lower p")

model = Model(Ipopt.Optimizer)             # fresh model
@variable(model, 0 <= x[1:T] <= 30)
@objective(model, Max, sum(price[t] * x[t] for t in 1:T))
cumx = [sum(x[1:t]) for t in 1:T]
@constraint(model, [t = 1:T],
    -δ[t] <= (l_min + l_max)/2 - l_0 + cumx[t] - μ[t] <= δ[t])
optimize!(model)
termination_status(model)

x_two = value.(x)               # [8.78, 9.50, 9.60]
objective_value(model)          # ≈ 54.94

#-------------------------------------------------------
# Checking the joint probability
#-------------------------------------------------------
using Random, Statistics
ξ = rand(MersenneTwister(1234), MvNormal(μ, Σ), 200_000) 
level(x) = l_0 .+ ξ .- cumsum(x)
mean(all(level(x_ind) .>= l_min; dims = 1))                                     # ≈ 0.81
mean(all((level(x_two) .>= l_min) .& (level(x_two) .<= l_max); dims = 1))       # ≈ 0.81

