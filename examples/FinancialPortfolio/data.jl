# ------------------------------------------------------------------
# Problem data for the bond-portfolio example
#
# We buy x[i] ≥ 0 units of bond i. Bond i pays yields[i, t] in year t.
# Uncertain liabilities β_t ~ N(·,·) must be covered every year, so with
# cumulative liability ξ_t = β_1 + … + β_t we need
#       cash_t = K + Σ_i (Σ_{k≤t} yields[i, k]) x_i  ≥  ξ_t .
# ------------------------------------------------------------------

Base.@kwdef struct PortfolioData
    K::Float64 = 0.0                  # initial cash
    costs::Vector{Float64}            # price of each bond
    yields::Matrix{Float64}           # n_assets × n_years payments
    β::Vector{Normal{Float64}}        # yearly liabilities (independent normals)
end

n_assets(d::PortfolioData) = length(d.costs)
n_years(d::PortfolioData)  = length(d.β)

"""
    liability_distribution(d) -> (μ, Σ)

Mean vector and covariance matrix of the cumulative liabilities
ξ_t = Σ_{k≤t} β_k, which is multivariate normal:
μ_t = Σ_{k≤t} E[β_k],   Σ_ij = Σ_{k≤min(i,j)} Var[β_k].
"""
function liability_distribution(d::PortfolioData)
    v, m = var.(d.β), mean.(d.β)
    T = length(v)
    Σ = [sum(v[1:min(i, j)]) for i in 1:T, j in 1:T]
    return cumsum(m), Σ
end

function default_data()
    β_mean = [12., 18., 20., 20., 16., 15., 18., 20.]
    β_std  = [5.0 * j for j in 1:8]
    return PortfolioData(
        K = 0.0,
        costs = [102., 99., 101., 98., 98., 104., 100., 101., 102., 94.],
        yields = [
            105.  0.   0.   0.   0.   0.   0.   0.;
              6.5 106.5 0.  0.   0.   0.   0.   0.;
              5.  105.  0.  0.   0.   0.   0.   0.;
              3.5 3.5 103.5 0.   0.   0.   0.   0.;
              4.  4.   4. 104.   0.   0.   0.   0.;
              9.  9.   9.   9. 109.   0.   0.   0.;
              6.  6.   6.   6. 106.   0.   0.   0.;
              8.  8.   8.   8.   8. 108.   0.   0.;
              9.  9.   9.   9.   9.   9. 109.   0.;
              7.  7.   7.   7.   7.   7.   7. 107.
        ],
        β = [Normal(β_mean[j], β_std[j]) for j in 1:8],
    )
end
