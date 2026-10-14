using Main.ChanceConstraints, Ipopt

# Find the cheapest x such that P(ξ ≤ x) ≥ p, where ξ ~ N(μ, Σ)
μ = [5.0, 2.0, 4.0, 3.0]
Σ = [ 2.0  0.5  0.3 -0.1;
      0.5  2.2  0.7  0.4;
      0.3  0.7  1.6 -0.2;
     -0.1  0.4 -0.2  1.8]
p = 0.9
c = [1.0, 1.0, 1.0, 1.0]


model = Model(Ipopt.Optimizer)
set_silent(model)

@variable(model, x[1:4] >= 0.0)
set_start_value.(x, [9.0, 11.45, 9.0, 9.0])

@objective(model, Min, c'x)

# One reliability window starting at index 1 and spanning all 4 time steps (κ = 3)
add_JCC_QMC(model, x, [1], 3, Σ, μ, p)

optimize!(model)

objective_value(model) # ≈ 24.56
value.(x)              # ≈ [7.71, 4.74, 6.47, 5.63]

# ---------------------------------------------------------------
# The same problem with spherical-radial decomposition (SRD).
# SRD needs the mean to be strictly feasible: x ≥ μ + δ
# ---------------------------------------------------------------

model = Model(Ipopt.Optimizer)
set_silent(model)

@variable(model, x[i = 1:4] >= 0.0)
set_start_value.(x, [9.0, 11.45, 9.0, 9.0])

@objective(model, Min, c'x)

add_JCC_SRD(model, x, [1], 3, Σ, μ, p)

optimize!(model)

objective_value(model) # ≈ 24.56
value.(x)              # ≈ [7.71, 4.74, 6.47, 5.63]


# ---------------------------------------------------------------
# Independent check of the joint probability P(ξ ≤ x*)
# ---------------------------------------------------------------

using MvNormalCDF, Random

prob, err = mvnormcdf(Σ, fill(-Inf, 4), value.(x) .- μ;
                      m = 1_000_000, rng = MersenneTwister(1234))
prob # ≈ 0.9
