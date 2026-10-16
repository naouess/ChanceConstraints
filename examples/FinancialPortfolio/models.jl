# ------------------------------------------------------------------
# Three formulations of the same problem. They share one base model and
# differ only in how the cash ≥ ξ requirement is imposed:
#   deterministic : cash_t ≥ E[ξ_t]
#   ICC           : P(cash_t ≥ ξ_t) ≥ p   for each year separately
#   JCC           : P(cash_t ≥ ξ_t for all t in a window) ≥ p   (ChanceConstraints.jl)
# ------------------------------------------------------------------

"""
Base model: variables x ≥ 0, affine yearly cash expressions, cost objective.
`objective_scale` only rescales the objective for the solver; the true cost is
always recomputed from `x` in `finish`.
"""
function base_model(d::PortfolioData; objective_scale = 1.0, linear_solver = nothing,
                    silent = true, solver_attrs...)
    B, T = n_assets(d), n_years(d)
    size(d.yields) == (B, T) ||
        throw(DimensionMismatch("yields must be $B × $T, got $(size(d.yields))"))

    model = Model(Ipopt.Optimizer)
    silent && set_silent(model)
    linear_solver === nothing || set_optimizer_attribute(model, "linear_solver", linear_solver)
    for (k, v) in solver_attrs
        set_optimizer_attribute(model, String(k), v)
    end

    @variable(model, x[1:B] >= 0)

    Y = cumsum(d.yields, dims = 2)                       # cumulative yields
    cash = [d.K + sum(Y[i, t] * x[i] for i in 1:B) for t in 1:T]
    @objective(model, Min, sum(d.costs[i] * x[i] for i in 1:B) / objective_scale)

    return model, x, cash
end

function finish(model, x, d::PortfolioData)
    optimize!(model)
    xv = max.(value.(x), 0.0)                            # clip solver noise (~ -1e-9)
    return (x = xv, cost = dot(d.costs, xv), status = termination_status(model))
end

# ---- 1. deterministic: replace ξ by its mean -----------------------
function solve_deterministic(d::PortfolioData; kwargs...)
    model, x, cash = base_model(d; kwargs...)
    μ, _ = liability_distribution(d)
    @constraint(model, cash .>= μ)
    return finish(model, x, d)
end

# ---- 2. individual chance constraints: closed-form quantiles ---------
function solve_icc(d::PortfolioData, p::Float64; kwargs...)
    model, x, cash = base_model(d; kwargs...)
    μ, Σ = liability_distribution(d)
    for t in 1:n_years(d)
        @constraint(model, cash[t] >= quantile(Normal(μ[t], sqrt(Σ[t, t])), p))
    end
    return finish(model, x, d)
end

# ---- 3. joint chance constraint with ChanceConstraints.jl ------------
"""
    solve_jcc(d, p; method = :SRD, window_starts = [1], κ = n_years(d) - 1, ...)

Impose, for every window start `s` in `window_starts`,

    P( cash_t ≥ ξ_t  for all t = s, …, s + κ ) ≥ p,      ξ ~ N(μ, Σ).

The default (one window starting in year 1 with κ = T - 1) is the classical
joint constraint over the whole horizon. `κ + 1` is the window length.
"""
function solve_jcc(d::PortfolioData, p::Float64;
                   method::Symbol = :SRD,
                   window_starts::AbstractVector{<:Integer} = [1],
                   κ::Integer = n_years(d) - 1,
                   s = nothing, seed::Integer = 1234, kwargs...)
    method in (:QMC, :SRD) || throw(ArgumentError("method must be :QMC or :SRD"))
    T = n_years(d)
    (minimum(window_starts) ≥ 1 && maximum(window_starts) + κ ≤ T) ||
        throw(ArgumentError("every window s:s+κ must lie inside 1:$T"))

    model, x, cash = base_model(d; objective_scale = 1000.0,
                                warm_start_init_point = "yes",
                                tol = 1e-9, constr_viol_tol = 1e-6, kwargs...)
    set_start_value.(x, 0.2)
    μ, Σ = liability_distribution(d)

    if method == :QMC
        # quasi-Monte Carlo evaluation of the multivariate normal CDF
        add_JCC_QMC(model, cash, collect(window_starts), κ, Σ, μ, p;
                    two_sided = false, s = something(s, 10_000),
                    rng = MersenneTwister(seed))
    else
        add_JCC_SRD(model, cash, collect(window_starts), κ, Σ, μ, p;
                    s = something(s, 5_000), seed = seed)
    end

    return finish(model, x, d)
end
