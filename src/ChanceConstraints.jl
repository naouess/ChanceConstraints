module ChanceConstraints
    using JuMP, MvNormalCDF, Random, Distributions, SpecialFunctions, LinearAlgebra

    export add_JCC_SRD, add_JCC_QMC, compute_with_SRD, compute_with_QMC

    include("chi_gamma.jl")
    include("SRD.jl")
    include("QMC.jl")
end

