using Main.ChanceConstraints, Ipopt

# Hydropower reservoir over T periods. With cumulative inflow ξ ~ N(μ, Σ) and
# release x_t, the water level is   level_t = l_0 + ξ_t - (x_1 + … + x_t).

# Maximise the total release while keeping the level inside the allowed range
# in all periods jointly with probability p.

T = 3
l_0, l_min, l_max = 5.0, 5.0, 10.0
p = 0.9

β_mean = fill(10.0, T)   # independent yearly inflows with these means …
β_var  = fill(0.5, T)    # … and variances
μ = cumsum(β_mean)
Σ = [sum(β_var[1:min(i, j)]) for i in 1:T, j in 1:T]


# ---------------------------------------------------------------
# Single-sided QMC: only the lower bound  level_t ≥ l_min  matters
# ---------------------------------------------------------------

model = Model(Ipopt.Optimizer)
set_silent(model)

@variable(model, 0 <= x[1:T] <= 30)
set_start_value.(x, 9.0)

@objective(model, Max, sum(x))

cumx = [sum(x[1:t]) for t in 1:T]

# level_t ≥ l_min  ⇔  -ξ_t ≤ l_0 - l_min - cumx_t,  and  -ξ ~ N(-μ, Σ)
slack = [l_0 - l_min - cumx[t] for t in 1:T]
μ_negative = -μ

# One reliability window starting at period 1 and spanning all T periods (κ = T - 1)
add_JCC_QMC(model, slack, [1], T - 1, Σ, μ_negative, p)

optimize!(model)

x_one = value.(x)
objective_value(model) # ≈ 28.43  (only the total is pinned down; the split over periods is not)

# ---------------------------------------------------------------
# Single-sided SRD: only the lower bound  level_t ≥ l_min  matters
# ---------------------------------------------------------------

model = Model(Ipopt.Optimizer)
set_silent(model)

@variable(model, 0 <= x[1:T] <= 30)
set_start_value.(x, 9.0)

@objective(model, Max, sum(x))

cumx = [sum(x[1:t]) for t in 1:T]

# level_t ≥ l_min  ⇔  -ξ_t ≤ l_0 - l_min - cumx_t,  and  -ξ ~ N(-μ, Σ)
slack = [l_0 - l_min - cumx[t] for t in 1:T]
μ_negative = -μ

# One reliability window starting at period 1 and spanning all T periods (κ = T - 1)
add_JCC_SRD(model, slack, [1], T - 1, Σ,  μ_negative, p)

optimize!(model)

x_one = value.(x)
objective_value(model) # ≈ 28.43  (only the total is pinned down; the split over periods is not)


# ---------------------------------------------------------------
# Two-sided QMC: lower AND upper bound, l_min ≤ level_t ≤ l_max
# ---------------------------------------------------------------

model = Model(Ipopt.Optimizer)
set_silent(model)

@variable(model, 0 <= x[1:T] <= 30)
set_start_value.(x, [9.0, 10.0, 12.0])   # fill(9.0, T) does not converge here

@objective(model, Max, sum(x))

cumx = [sum(x[1:t]) for t in 1:T]

# l_min ≤ level_t ≤ l_max  ⇔  lower_t ≤ ξ_t ≤ upper_t
lower = [l_min - l_0 + cumx[t] for t in 1:T]
upper = [l_max - l_0 + cumx[t] for t in 1:T]

# Stack the bounds as [lower; upper] (length 2T). The random vector still has
# T components, so κ stays T - 1.
add_JCC_QMC(model, vcat(lower, upper), [1], T - 1, Σ, μ, p; two_sided = true)

optimize!(model)

x_two = value.(x)
objective_value(model) # ≈ 28.39, slightly below the single-sided value


# ---------------------------------------------------------------
# Monte Carlo check of the probability that the level stays in bounds
# ---------------------------------------------------------------

using Distributions, Random, Statistics

ξ = rand(MersenneTwister(1234), MvNormal(μ, Σ), 200_000)   # T × N scenarios

level(x) = l_0 .+ ξ .- cumsum(x)

mean(all(level(x_one) .>= l_min; dims = 1))                 # ≈ 0.9
mean(all((level(x_two) .>= l_min) .& (level(x_two) .<= l_max); dims = 1))  # ≈ 0.9
