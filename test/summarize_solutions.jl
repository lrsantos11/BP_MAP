# Summarize the CSVs written by test/runbenchmarkprecvals.jl. For each problem (row) the
# reference objective best_obj is the smallest ‖x‖₁ among the solvers with
# ‖Ax - b‖₂ / ‖b‖₂ ≤ tol, and each solver gets the relative objective gap
# (obj - best_obj) / best_obj. Writes the input rows with the columns best_obj and
# "<solver> rel_gap" added, and prints, per solver, the number of runs with each status and
# the largest and median gap among the runs with rel_feas ≤ tol. Usage, from the project
# directory:
#
#     julia --project test/summarize_solutions.jl [--tol=1e-6] [--csv=out.csv] [file.csv ...]
#
# Without files it reads LSS_benchmark_precvals_<rhs>.csv, rhs = 1, ..., 4, from the
# current directory. The default output name replaces the right-hand side of the first
# file name by "summary", e.g. LSS_benchmark_precvals_summary.csv.

using DrWatson
@quickactivate "BP_MAP"

using CSV
using DataFrames
using Statistics

function summarize_solutions(files, tol, csvfile)
    df = vcat([CSV.read(f, DataFrame, stringtype = String) for f in files]..., cols = :union)
    solvers = [chopsuffix(c, " obj") for c in names(df) if endswith(c, " obj")]
    isfeasible(s) = coalesce.(df[!, "$s rel_feas"] .<= tol, false)  # NaN and missing are not

    best = fill(NaN, nrow(df))
    for s in solvers
        feasible, obj = isfeasible(s), df[!, "$s obj"]
        for i in findall(feasible)
            if isnan(best[i]) || obj[i] < best[i]
                best[i] = obj[i]
            end
        end
    end
    df.best_obj = best
    for s in solvers
        df[!, "$s rel_gap"] = (df[!, "$s obj"] .- best) ./ best
    end
    CSV.write(csvfile, df)

    stats = DataFrame()
    for s in solvers
        status = coalesce.(df[!, "$s status"], "missing")
        gaps = df[isfeasible(s), "$s rel_gap"]
        push!(
            stats,
            (
                solver = s,
                success = count(==("success"), status),
                failure = count(==("failure"), status),
                timeout = count(==("timeout"), status),
                error = count(==("error"), status),
                feasible = length(gaps),
                max_gap = isempty(gaps) ? NaN : maximum(gaps),
                median_gap = isempty(gaps) ? NaN : median(gaps),
            ),
        )
    end
    println(stats)
    @info "Summary of $(nrow(df)) problems saved to $csvfile"
    return df
end

function parse_args(args)
    tol, csvfile, files = 1.0e-6, nothing, String[]
    for arg in args
        if startswith(arg, "--tol=")
            tol = parse(Float64, split(arg, "=")[2])
        elseif startswith(arg, "--csv=")
            csvfile = String(split(arg, "=")[2])
        elseif startswith(arg, "--")
            error("Unknown option $arg")
        else
            push!(files, arg)
        end
    end
    if isempty(files)
        files = filter(isfile, ["LSS_benchmark_precvals_$rhs.csv" for rhs = 1:4])
        isempty(files) && error("No LSS_benchmark_precvals_<rhs>.csv in the current directory")
    end
    csvfile = something(csvfile, replace(first(files), r"(_\d+)?\.csv$" => "_summary.csv"))
    return files, tol, csvfile
end


if abspath(PROGRAM_FILE) == @__FILE__
    summarize_solutions(parse_args(ARGS)...)
end
