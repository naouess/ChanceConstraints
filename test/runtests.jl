using ChanceConstraints
using Test
using Random

p = 0.9
μ = [5; 2; 4; 3;]
Σ = [
    2.0   0.5   0.3  -0.1;
    0.5   2.2   0.7   0.4;
    0.3   0.7   1.6  -0.2;
   -0.1   0.4  -0.2   1.8;
]

κ = 3

x = [8; 6; 7; 5;]

prob_qmc  = compute_with_QMC(1, κ, Σ, μ, false, 10000, MersenneTwister(1234))[1](x...)
prob_srd = compute_with_SRD(1, κ, Σ, μ, 10000, MersenneTwister(1234))[1](x...)

v1 = round.(prob_qmc, digits=6)
v2 = round.(prob_srd, digits=6)

prob_qmc1 = compute_with_QMC(1, κ, Σ, μ)[1](x...)
prob_srd1 = compute_with_SRD(1, κ, Σ, μ)[1](x...)

v3 = round.(prob_qmc1, digits=6)
v4 = round.(prob_srd1, digits=6)

prob_qmc_default_1 = compute_with_QMC(1, κ-1, Σ, μ)[1](x...)
prob_qmc_default_2 = compute_with_QMC(2, κ-1, Σ, μ)[1](x...)
prob_srd_default_1 = compute_with_SRD(1, κ-1, Σ, μ)[1](x...)
prob_srd_default_2 = compute_with_SRD(2, κ-1, Σ, μ)[1](x...)

v5 = round.([prob_qmc_default_1 prob_qmc_default_2], digits=6)
v6 = round.([prob_srd_default_1 prob_srd_default_2], digits=6)

@testset "JointChance.jl" begin
    @test v1 == 0.905377
    @test v2 == 0.909216
    @test v3 == 0.905373
    @test v4 == 0.907449
    @test v5 == [0.971655  0.920863]
    @test v6 == [0.9725  0.922655]
end

# TODO add test for the functions adding constraints to a JuMP model