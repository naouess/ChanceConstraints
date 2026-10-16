# ------------------------------------------------------------------
# Out-of-sample evaluation. Every formulation is judged on the SAME
# correlated scenarios of the cumulative liabilities ξ.
# ------------------------------------------------------------------

"Draw `n` scenarios of ξ ~ N(μ, Σ); returns an n × T matrix."
function sample_liabilities(d::PortfolioData, n::Integer; rng = MersenneTwister(1234))
    μ, Σ = liability_distribution(d)
    return permutedims(rand(rng, MvNormal(μ, Σ), n))
end

"Cash minus liabilities in every scenario and year (n × T); a negative entry is a shortfall."
function cash_paths(d::PortfolioData, x::AbstractVector, ξ::AbstractMatrix)
    cash = d.K .+ cumsum(d.yields, dims = 2)' * x         # length-T vector
    return cash' .- ξ
end

"Boolean vector: does scenario m have a shortfall anywhere in the given years?"
shortfall(paths::AbstractMatrix; cols = 1:size(paths, 2), tol = 1e-12) =
    vec(any(paths[:, cols] .< tol, dims = 2))

"Fraction of scenarios with no shortfall in any year (full-horizon joint reliability)."
empirical_reliability(paths; tol = 1e-12) = 1 - mean(shortfall(paths; tol))

"Reliability of each window s:s+κ separately."
window_reliabilities(paths, window_starts, κ; tol = 1e-12) =
    [1 - mean(shortfall(paths; cols = s:s+κ, tol)) for s in window_starts]
