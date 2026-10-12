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
    @test gradient_value ≈ numeric_gradient atol=3e-3
end

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


###################################
# Test add_JCC_QMC and add_JCC_SRD
###################################

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

@testset "add_JCC_QMC" begin
    μ = zeros(3)
    Σ = Matrix{Float64}(I, 3, 3)
    idx = [1]
    κ = 2
    p = 0.9

    model = build_fixed_point_model() do model, x
        add_JCC_QMC(
            model, x, idx, κ, Σ, μ, p;
            s=10_000,
            rng=MersenneTwister(1234),
        )
    end

    optimize!(model)
    @test is_solved_and_feasible(model)
end

@testset "add_JCC_SRD" begin
    μ = zeros(3)
    Σ = Matrix{Float64}(I, 3, 3)
    idx = [1]
    κ = 2
    p = 0.9

    model = build_fixed_point_model() do model, x
        add_JCC_SRD(
            model, x, idx, κ, Σ, μ, p;
            s=100_000,
            seed=1234,
        )
    end

    optimize!(model)
    @test is_solved_and_feasible(model)
end