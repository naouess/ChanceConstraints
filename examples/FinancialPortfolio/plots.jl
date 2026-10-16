function plot_cash_paths(paths::AbstractMatrix;
                         title = "",
                         tol = 1e-12,
                         only_violations = false,
                         file = nothing)

    n_years = size(paths, 2)
    years = 1:n_years

    plt = plot(
        xlabel = "Year",
        ylabel = "Cash",
        title = title,
        legend = :outerright,
        size = (600, 400),
        dpi = 300,
        background_color = :white,
        foreground_color = :black,
        grid = :y,
        gridcolor = :gray,
        gridalpha = 0.18,
        framestyle = :box,
        tickfont = font(9),
        guidefont = font(10),
        titlefont = font(11),
        legendfont = font(5),
        # fontfamily = "Arial",
        # tickfont = font("Arial", 9),
        # guidefont = font("Arial", 10),
        # titlefont = font("Arial", 11),
        # legendfont = font("Arial", 9),
        right_margin = 5Plots.mm,
        left_margin = 5Plots.mm,
        bottom_margin = 4Plots.mm,
    )

    hline!(plt, [0.0];
        color = :gray35,
        linewidth = 1.0,
        linestyle = :dot,
        label = "Zero cash")

    plot!(plt, Float64[], Float64[];
        color = :black, linewidth = 1.2, linestyle = :solid,
        label = "Feasible")
    plot!(plt, Float64[], Float64[];
        color = :orangered3, linewidth = 1.2, linestyle = :dash,
        label = "Shortfall")

    for m in axes(paths, 1)
        bad = any(paths[m, :] .< tol)
        only_violations && !bad && continue

        plot!(plt, years, paths[m, :];
            color = bad ? :orangered3 : :black,
            linestyle = bad ? :dash : :solid,
            linewidth = 0.9,
            alpha = bad ? 0.65 : 0.35,
            label = "")
    end

    if file !== nothing
        mkpath(dirname(file))
        savefig(plt, file)
    end

    return plt
end