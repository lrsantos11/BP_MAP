# Re-run, with the timing code of test/runbenchmark.jl, the ISAL runs of the LSS benchmark
# that stopped at ISAL1's default time limit of 900 s instead of the intended 3600 s:
# solveBP_ISAL1 did not pass the limit as a name/value pair (fixed in scripts/BP_ISAL1.jl).
# ISAL1 uses the limit only in its clock check, so the runs that ended before 900 s are
# unaffected.
#
# Each finished run is appended at once to the output CSV, with the same columns as the
# benchmark (Problem, M, N, ISAL, ISAL dist, ISAL iters), and runs already in that file are
# skipped on restart. Its rows replace the ISAL values of the same Problem in
# test/neuronio_rev1/LSS_benchmark_CUDA_<rhs>.csv. Usage, from the project directory:
#
#     julia --project test/rerun_isal_timelimit.jl [output.csv]

include(joinpath(@__DIR__, "runbenchmark.jl"))

# ISAL runs in test/neuronio_rev1/LSS_benchmark_CUDA_*.csv that took 900 s or more
const RERUN = [
    ("SC13.mat", 2),
    ("SC17.mat", 2),
    ("SC7.mat", 3),
    ("SC11.mat", 3),
    ("SC13.mat", 3),
    ("SC7.mat", 4),
    ("SC11.mat", 4),
    ("SC12.mat", 4),
    ("SC13.mat", 4),
]

function rerun_isal(resfile, instances = RERUN)
    done = isfile(resfile) ? Set(CSV.read(resfile, DataFrame).Problem) : Set{String}()
    for (k, (file, rhs)) in enumerate(instances)
        println("="^78)
        problem = "$file - $rhs"
        if problem in done
            @info "$problem is already in $resfile"
            continue
        end
        @info "$k / $(length(instances)): $problem"
        prob = readl1test(
            datadir("exp_raw", "lassobp_mat", file);
            rhs = rhs,
            mattype = mattype,
            acceltype = noaccel,
        )
        m, n = size(prob)
        duration, dist, it_ISAL = solve_with_ISAL(prob)
        row = DataFrame(
            "Problem" => [problem],
            "M" => [m],
            "N" => [n],
            "ISAL" => [duration],
            "ISAL dist" => [dist],
            "ISAL iters" => [round(Int, it_ISAL)],
        )
        CSV.write(resfile, row, append = isfile(resfile))
    end
    println("="^78)
    return CSV.read(resfile, DataFrame)
end


if abspath(PROGRAM_FILE) == @__FILE__
    println(rerun_isal(isempty(ARGS) ? "LSS_ISAL_rerun.csv" : ARGS[1]))
end
