# Re-run, with the timing code of test/runbenchmark.jl, the ISAL runs of the LPT benchmark
# whose distance to the reference solution lies in (1e-6, 1e-4). They were counted as
# failures under the old threshold of 1e-6 and may count as solved under the current 1e-4.
#
# Each finished run is appended at once to the output CSV, with the same columns as the
# benchmark (Problem, M, N, ISAL, ISAL dist, ISAL iters), and runs already in that file are
# skipped on restart. Its rows replace the ISAL values of the same Problem in
# test/neuronio_rev1/LPT_benchmark.csv; comparing ISAL iters with the old row tells whether
# the run was reproduced. Usage, from the project directory:
#
#     julia --project test/rerun_isal_dist.jl [output.csv]

include(joinpath(@__DIR__, "runbenchmark.jl"))

# ISAL runs in test/neuronio_rev1/LPT_benchmark.csv with 1e-6 < ISAL dist < 1e-4
const RERUN = [
    "spear_inst_272.mat",
    "spear_inst_217.mat",
    "spear_inst_230.mat",
    "spear_inst_535.mat",
    "spear_inst_213.mat",
    "spear_inst_267.mat",
    "spear_inst_229.mat",
    "spear_inst_156.mat",
    "spear_inst_266.mat",
    "spear_inst_214.mat",
    "spear_inst_239.mat",
    "spear_inst_446.mat",
    "spear_inst_226.mat",
    "spear_inst_456.mat",
    "spear_inst_253.mat",
    "spear_inst_438.mat",
    "spear_inst_442.mat",
    "spear_inst_225.mat",
]

function rerun_isal_dist(resfile, instances = RERUN)
    done = isfile(resfile) ? Set(CSV.read(resfile, DataFrame).Problem) : Set{String}()
    for (k, file) in enumerate(instances)
        println("="^78)
        problem = "$file - 1"
        if problem in done
            @info "$problem is already in $resfile"
            continue
        end
        @info "$k / $(length(instances)): $problem"
        prob = readl1test(
            datadir("exp_raw", "L1_Testset_mat", file);
            rhs = 1,
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
    println(rerun_isal_dist(isempty(ARGS) ? "LPT_ISAL_rerun.csv" : ARGS[1]))
end
