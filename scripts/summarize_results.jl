# Summarize a run_benchmark CSV: solved/timeout counts, median and geometric
# mean solve time (over solved instances only), and median iteration count,
# per solver. A solver's duration column is negative when it did not solve
# the instance (the convention already used by every solve_with_* function
# in test/runbenchmark.jl).
#
# Usage: julia --project scripts/summarize_results.jl path/to/results.csv

using DrWatson
@quickactivate :BP_MAP

using CSV
using DataFrames
using Statistics
using Printf

"Geometric mean of a vector of positive values"
geomean(x) = exp(mean(log.(x)))

function summarize_results(resfile)
    df = CSV.read(resfile, DataFrame)
    ntests = nrow(df)

    # A duration column belongs to a solver iff a matching "<name> dist" column exists.
    solver_names = [
        replace(c, r" dist$" => "") for
        c in names(df) if endswith(c, " dist") && replace(c, r" dist$" => "") in names(df)
    ]

    rows = NamedTuple[]
    for name in solver_names
        durations = df[!, name]
        solved = durations .> 0
        nsolved = count(solved)
        push!(
            rows,
            (
                Solver = name,
                Solved = nsolved,
                Timeout = ntests - nsolved,
                MedianTime = nsolved > 0 ? median(durations[solved]) : NaN,
                GeoMeanTime = nsolved > 0 ? geomean(durations[solved]) : NaN,
                MedianIters = if (name * " iters") in names(df) && nsolved > 0
                    median(df[!, name*" iters"][solved])
                else
                    NaN
                end,
            ),
        )
    end
    summary = DataFrame(rows)
    println(summary)
    return summary
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("Usage: julia --project scripts/summarize_results.jl path/to/results.csv")
    summarize_results(ARGS[1])
end
