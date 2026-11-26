using ChanceConstraints
using Test
using Random
using Distributions
using LinearAlgebra

function finite_difference_gradient(f, x; h=1e-5)
    gradient = zeros(length(x))

    for i in eachindex(x)
        x_plus = copy(x)
        x_minus = copy(x)
        x_plus[i] += h
        x_minus[i] -= h

        gradient[i] = (f(x_plus...) - f(x_minus...)) / (2h)
    end

    return gradient
end

# ---------------------------------------------------------------------
# Closed-form references for independent components (diagonal Σ)
# ---------------------------------------------------------------------

# P(ζ ≤ h) = ∏ Φ((hᵢ - μᵢ)/σᵢ)
function exact_one_sided(h, μ, σ)
    return prod(cdf.(Normal(), (h .- μ) ./ σ))
end

# ∂/∂hᵢ = φ(zᵢ)/σᵢ ∏_{k≠i} Φ(z_k)
function exact_one_sided_gradient(h, μ, σ)
    z = (h .- μ) ./ σ
    return [pdf(Normal(), z[i]) / σ[i] *
            prod(cdf(Normal(), z[k]) for k in eachindex(z) if k != i)
            for i in eachindex(z)]
end

# P(z ≤ ζ ≤ h) = ∏ [Φ(uᵢ) - Φ(lᵢ)]
function exact_two_sided(z, h, μ, σ)
    l = (z .- μ) ./ σ
    u = (h .- μ) ./ σ
    return prod(cdf.(Normal(), u) .- cdf.(Normal(), l))
end

# Returns the gradient with respect to (z..., h...), the order used by the operator.
function exact_two_sided_gradient(z, h, μ, σ)
    l = (z .- μ) ./ σ
    u = (h .- μ) ./ σ
    q = length(z)
    mass(k) = cdf(Normal(), u[k]) - cdf(Normal(), l[k])
    others(i) = prod(mass(k) for k in 1:q if k != i)
    grad_h = [pdf(Normal(), u[i]) / σ[i] * others(i) for i in 1:q]
    grad_z = [-pdf(Normal(), l[i]) / σ[i] * others(i) for i in 1:q]
    return vcat(grad_z, grad_h)
end


# ---------------------------------------------------------------------
# compute_with_QMC
# ---------------------------------------------------------------------

# @testset "ChanceConstraints.jl" begin
@testset "compute_with_QMC: three-dimensional benchmark" begin
    normal = Normal()

    μ = [0.5, -1.0, 2.0]
    variances = [1.0, 4.0, 0.25]
    Σ = Matrix(Diagonal(variances))
    x = [1.0, 0.0, 2.5]

    probability, gradient = compute_with_QMC(
        1, 2, Σ, μ;
        s=100_000,
        rng=MersenneTwister(1234),
    )

    σ = sqrt.(variances)
    z = (x .- μ) ./ σ

    # For diagonal Σ, the components are independent.
    exact_probability = prod(cdf(normal, zi) for zi in z)

    # Analytical gradient for independent components:
    # ∂f/∂xᵢ = φ(zᵢ)/σᵢ × ∏ⱼ≠ᵢ Φ(zⱼ),
    # where zᵢ = (xᵢ - μᵢ)/σᵢ.
    exact_gradient = [
        pdf(normal, z[i]) / σ[i] *
        prod(cdf(normal, z[k]) for k in eachindex(z) if k != i)
        for i in eachindex(z)
    ]

    @test probability(x...) ≈ exact_probability atol=2e-3

    # The gradient callback writes into its first argument.
    gradient_value = zeros(length(x))
    gradient(gradient_value, x...)

    @test all(isfinite, gradient_value)
    @test gradient_value ≈ exact_gradient atol=3e-3

    # Also check the gradient against finite differences of the
    # probability function.
    numeric_gradient = finite_difference_gradient(probability, x)
    @test gradient_value ≈ numeric_gradient atol=1e-6
end

@testset "compute_with_QMC: two-sided benchmark" begin
    μ = [0.5, -1.0, 2.0]
    variances = [1.0, 4.0, 0.25]
    Σ = Matrix(Diagonal(variances))
    σ = sqrt.(variances)

    lower = [-0.5, -3.0, 1.0]
    upper = [1.5, 2.0, 3.0]
    args = vcat(lower, upper)       # operator arguments are (z..., h...)

    probability, gradient = compute_with_QMC(
        1, 2, Σ, μ;
        two_sided=true,
        s=100_000,
        rng=MersenneTwister(1234),
    )

    @test probability(args...) ≈ exact_two_sided(lower, upper, μ, σ) atol=2e-3

    gradient_value = zeros(length(args))
    gradient(gradient_value, args...)

    @test all(isfinite, gradient_value)
    @test gradient_value ≈ exact_two_sided_gradient(lower, upper, μ, σ) atol=3e-3

    numeric_gradient = finite_difference_gradient(probability, args)
    @test gradient_value ≈ numeric_gradient atol=3e-3
end

@testset "compute_with_QMC: window inside a larger vector" begin
    # The window j = 2, κ = 2 covers components 2:4 of a five-dimensional ζ.
    # The operator then takes only the q = 3 bounds of the window.
    μ = [0.0, 0.5, -1.0, 2.0, 1.0]
    variances = [1.0, 1.0, 4.0, 0.25, 2.0]
    Σ = Matrix(Diagonal(variances))
    idx = 2:4
    h = [1.0, 0.0, 2.5]

    probability, gradient = compute_with_QMC(
        2, 2, Σ, μ;
        s=100_000,
        rng=MersenneTwister(1234),
    )

    σ = sqrt.(variances[idx])
    @test probability(h...) ≈ exact_one_sided(h, μ[idx], σ) atol=2e-3

    gradient_value = zeros(length(h))
    gradient(gradient_value, h...)
    @test gradient_value ≈ exact_one_sided_gradient(h, μ[idx], σ) atol=3e-3
end

@testset "compute_with_QMC: crossing bounds" begin 
    # Depends on the guard that returns 0 and a zero gradient when z_i ≥ h_i.
    μ = zeros(3)
    Σ = Matrix{Float64}(I, 3, 3)
    probability, gradient = compute_with_QMC(
        1, 2, Σ, μ;
        two_sided=true,
        s=10_000,
        rng=MersenneTwister(1234),
    )

    args = [0.0, 1.0, -1.0, 1.0, 0.5, 2.0]   # z = (0, 1, -1), h = (1, 0.5, 2): z₂ > h₂

    # The warning uses maxlog = 1, so it is silenced instead of being tested.
    # with_logger(NullLogger()) do
    @test probability(args...) == 0.0

    gradient_value = ones(length(args))
    gradient(gradient_value, args...)
    @test all(iszero, gradient_value)

end

@testset "compute_with_QMC: input validation" begin
    μ = zeros(3)
    Σ = Matrix{Float64}(I, 3, 3)
    @test_throws AssertionError compute_with_QMC(0, 2, Σ, μ)            # j must be positive
    @test_throws AssertionError compute_with_QMC(2, 2, Σ, μ)            # window exceeds dimension
    @test_throws AssertionError compute_with_QMC(1, 2, Σ[:, 1:2], μ)    # Σ not square
    @test_throws AssertionError compute_with_QMC(1, 2, -Σ, μ)           # Σ not positive definite
end

# ---------------------------------------------------------------------
# compute_with_SRD
# ---------------------------------------------------------------------

@testset "compute_with_SRD: probability and gradient" begin
    μ = [0.5, -1.0, 2.0]
    x = [1.0, 0.0, 2.5]

    # Generate fixed, unit-length samples for the SRD calculation.
    rng = MersenneTwister(1234)
    transformed_samples = randn(rng, 3, 10_000)
    transformed_samples ./= sqrt.(sum(abs2, transformed_samples; dims=1))

    probability, gradient = compute_with_SRD(
        1, 2, transformed_samples, μ,
    )

    probability_value = probability(x...)
    @test isfinite(probability_value)
    @test 0.0 <= probability_value <= 1.0

    gradient_value = zeros(length(x))
    gradient(gradient_value, x...)

    @test length(gradient_value) == length(x)
    @test all(isfinite, gradient_value)

    numeric_gradient = finite_difference_gradient(probability, x)
    @test gradient_value ≈ numeric_gradient atol=3e-3
end

@testset "compute_with_SRD: probability and gradient" begin
    μ = [0.5, -1.0, 2.0]
    x = [1.0, 0.0, 2.5]

    # Generate fixed, unit-length samples for the SRD calculation.
    # One sample per row, as expected by ProbFunction.
    rng = MersenneTwister(1234)
    # transformed_samples = randn(rng, 3, 10_000)
    # transformed_samples ./= sqrt.(sum(abs2, transformed_samples; dims=1))
    transformed_samples = randn(rng, 100_000, 3)
    transformed_samples ./= sqrt.(sum(abs2, transformed_samples; dims=2))

    probability, gradient = compute_with_SRD(
        1, 2, transformed_samples, μ,
    )

    probability_value = probability(x...)
    @test isfinite(probability_value)
    @test 0.0 <= probability_value <= 1.0

    # For Σ = I the components are independent, so a closed form exists.
    σ = ones(3)
    @test probability_value ≈ exact_one_sided(x, μ, σ) atol=5e-3

    gradient_value = zeros(length(x))
    gradient(gradient_value, x...)

    @test length(gradient_value) == length(x)
    @test all(isfinite, gradient_value)
    @test gradient_value ≈ exact_one_sided_gradient(x, μ, σ) atol=1e-2

    numeric_gradient = finite_difference_gradient(probability, x)
    @test gradient_value ≈ numeric_gradient atol=3e-3
end

@testset "compute_with_SRD: window inside a larger vector" begin
    # Window w = 2, κ = 2 of a five-dimensional ζ with Σ = I.
    μ = [0.0, 0.5, -1.0, 2.0, 1.0]
    Σ = Matrix{Float64}(I, 5, 5)
    h = [1.0, 0.0, 2.5]          # bounds of components 2:4
    idx = 2:4

    raw = randn(MersenneTwister(1234), 100_000, 6)   # blocks up to window 2: 3 * 2 columns
    transformed = ChanceConstraints.transform_srd_samples(raw, 2, Σ)

    probability, gradient = compute_with_SRD(2, 2, transformed, μ)

    @test probability(h...) ≈ exact_one_sided(h, μ[idx], ones(3)) atol=5e-3

    gradient_value = zeros(length(h))
    gradient(gradient_value, h...)
    @test gradient_value ≈ exact_one_sided_gradient(h, μ[idx], ones(3)) atol=1e-2
end

@testset "QMC and SRD agree for correlated Gaussians" begin 
    # No closed form exists here, so the two independent methods validate each other.
    μ = zeros(3)
    Σ = [1.0 0.5 0.3;
         0.5 2.0 0.4;
         0.3 0.4 1.5]
    h = [1.5, 2.0, 1.8]

    f_qmc, ∇f_qmc = compute_with_QMC(1, 2, Σ, μ; s=100_000, rng=MersenneTwister(1))

    raw = randn(MersenneTwister(2), 100_000, 3)
    transformed = ChanceConstraints.transform_srd_samples(raw, 2, Σ)
    f_srd, ∇f_srd = compute_with_SRD(1, 2, transformed, μ)

    @test f_qmc(h...) ≈ f_srd(h...) atol=5e-3

    g_qmc = zeros(3)
    g_srd = zeros(3)
    ∇f_qmc(g_qmc, h...)
    ∇f_srd(g_srd, h...)
    @test g_qmc ≈ g_srd atol=1e-2
end

@testset "compute_with_SRD: bounds below the mean" begin 
    # Currently ProbFunction throws if h - μ ≤ 1e-10.
    # Update this test if you replace the error by NaN or a clamp.
    μ = zeros(3)
    samples = randn(MersenneTwister(1234), 1_000, 3)
    samples ./= sqrt.(sum(abs2, samples; dims=2))
    probability, _ = compute_with_SRD(1, 2, samples, μ)
    @test_throws ErrorException probability(-1.0, 1.0, 1.0)
end

# ---------------------------------------------------------------------
# add_JCC_QMC! and add_JCC_SRD!
# ---------------------------------------------------------------------


using Ipopt
using JuMP

function build_fixed_point_model(add_constraint!)
    model = Model(Ipopt.Optimizer)
    set_silent(model)

    @variable(model, x[1:3])
    for i in eachindex(x)
        fix(x[i], 5.0)
    end

    add_constraint!(model, x)
    @objective(model, Min, 0.0)

    return model
end

@testset "add_JCC_QMC! one-sided fixed point example" begin
    μ = zeros(3)
    Σ = Matrix{Float64}(I, 3, 3)
    idx = [1]
    κ = 2
    p = 0.9

    model = build_fixed_point_model() do model, x
        add_JCC_QMC!(
            model, x, idx, κ, Σ, μ, p;
            s=10_000,
            rng=MersenneTwister(1234),
        )
    end

    optimize!(model)
    @test is_solved_and_feasible(model; allow_almost = true)
end

@testset "add_JCC_SRD! one-sided fixed point example" begin
    μ = zeros(3)
    Σ = Matrix{Float64}(I, 3, 3)
    idx = [1]
    κ = 2
    p = 0.9

    model = build_fixed_point_model() do model, x
        add_JCC_SRD!(
            model, x, idx, κ, Σ, μ, p;
            s=100_000,
            seed=1234,
        )
    end

    optimize!(model)
    @test is_solved_and_feasible(model; allow_almost = true)
end


@testset "add_JCC_QMC!: two-sided fixed point" begin
    μ = zeros(3)
    Σ = Matrix{Float64}(I, 3, 3)

    # P(|ζᵢ| ≤ 3 for all i) ≈ 0.992 ≥ 0.9: feasible
    feasible = build_fixed_point_model() do model, x
        x0 = [fix(x[i], 0.0; force=true) for i in 1:3]
        add_JCC_QMC!(model, x .+ 3.0, [1], 2, Σ, μ, 0.9;
                     z=x .- 3.0, two_sided=true, s=10_000, rng=MersenneTwister(1234))
    end
    optimize!(feasible)
    @test is_solved_and_feasible(feasible; allow_almost = true)

    # P(|ζᵢ| ≤ 0.5 for all i) ≈ 0.056 < 0.9: infeasible
    infeasible = build_fixed_point_model() do model, x
        x0 = [fix(x[i], 0.0; force=true) for i in 1:3]
        add_JCC_QMC!(model, x .+ 0.5, [1], 2, Σ, μ, 0.9;
                     z=x .- 0.5, two_sided=true, s=10_000, rng=MersenneTwister(1234))
    end
    optimize!(infeasible)
    @test !is_solved_and_feasible(infeasible)
end

@testset "add_JCC_QMC!: optimal value and chain rule" begin 
    # min Σxᵢ s.t. P(ζ ≤ h(x)) ≥ p with ζ ~ N(0, I₃). By symmetry and convexity
    # all bounds are equal and Φ(t)³ = p, i.e. t = Φ⁻¹(p^(1/3)).
    μ = zeros(3)
    Σ = Matrix{Float64}(I, 3, 3)
    p = 0.9
    t = quantile(Normal(), p^(1 / 3))

    # (a) h(x) = x
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:3])
    set_start_value.(x, 3.0)
    @objective(model, Min, sum(x))
    add_JCC_QMC!(model, x, [1], 2, Σ, μ, p; s=20_000, rng=MersenneTwister(1234))
    optimize!(model)
    @test is_solved_and_feasible(model; allow_almost = true)
    @test objective_value(model) ≈ 3t atol=0.05

    # (b) affine bounds h(x) = 2x + 1, so the gradient needs the chain rule: x = (t - 1)/2
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:3])
    set_start_value.(x, 1.0)
    @objective(model, Min, sum(x))
    add_JCC_QMC!(model, 2.0 .* x .+ 1.0, [1], 2, Σ, μ, p; s=20_000, rng=MersenneTwister(1234))
    optimize!(model)
    @test is_solved_and_feasible(model; allow_almost = true)
    @test objective_value(model) ≈ 3 * (t - 1) / 2 atol=0.05
end

@testset "add_JCC_SRD!: optimal value and chain rule" begin 
    μ = zeros(3)
    Σ = Matrix{Float64}(I, 3, 3)
    p = 0.9
    t = quantile(Normal(), p^(1 / 3))

    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:3])
    set_start_value.(x, 3.0)                       # h(x) > μ at the start
    @objective(model, Min, sum(x))
    add_JCC_SRD!(model, x, [1], 2, Σ, μ, p; s=100_000, seed=1234)
    optimize!(model)
    @test is_solved_and_feasible(model; allow_almost = true)
    @test objective_value(model) ≈ 3t atol=0.05

    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:3])
    set_start_value.(x, 1.0)
    @objective(model, Min, sum(x))
    add_JCC_SRD!(model, 2.0 .* x .+ 1.0, [1], 2, Σ, μ, p; s=100_000, seed=1234)
    optimize!(model)
    @test is_solved_and_feasible(model; allow_almost = true)
    @test objective_value(model) ≈ 3 * (t - 1) / 2 atol=0.05
end

@testset "add_JCC_*: input validation" begin 
    μ = zeros(3)
    Σ = Matrix{Float64}(I, 3, 3)

    function fresh()
        model = Model(Ipopt.Optimizer)
        set_silent(model)
        @variable(model, x[1:3])
        return model, x
    end

    model, x = fresh()
    @test_throws AssertionError add_JCC_QMC!(model, x, [1], 2, Σ, μ, 1.0)                 # p must be < 1
    @test_throws AssertionError add_JCC_QMC!(model, x, [1], 2, Σ, μ, 0.0)                 # p must be > 0
    @test_throws AssertionError add_JCC_QMC!(model, x, [1], 1, Σ, μ, 0.9)                 # QMC needs κ ≥ 2
    @test_throws AssertionError add_JCC_QMC!(model, x, [2], 2, Σ, μ, 0.9)                 # window exceeds dimension
    @test_throws AssertionError add_JCC_QMC!(model, x[1:2], [1], 2, Σ, μ, 0.9)            # h and μ differ in length
    @test_throws AssertionError add_JCC_QMC!(model, x, [1], 2, Σ, μ, 0.9; two_sided=true) # z missing
    @test_throws AssertionError add_JCC_QMC!(model, x, [1], 2, -Σ, μ, 0.9)                # Σ not positive definite

    @test_throws AssertionError add_JCC_SRD!(model, x, [1], 2, Σ, μ, 0.4)                 # SRD needs p > 0.5
    @test_throws AssertionError add_JCC_SRD!(model, x, [1], 2, Σ, μ, 1.0)                 # p must be < 1
    @test_throws AssertionError add_JCC_SRD!(model, x, [1], 0, Σ, μ, 0.9)                 # SRD needs κ ≥ 1
end
# end