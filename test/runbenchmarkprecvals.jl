# Benchmark as test/runbenchmark.jl that also records the quality of every solution. Besides
# the columns of runbenchmark.jl (time, negative when not solved, distance to the reference
# solution and iteration counts), the CSV has for each solver the relative feasibility
# ‖Ax - b‖₂ / ‖b‖₂, the objective ‖x‖₁, a status (success, failure, timeout or error) and
# the solver's own exit flag. Each computed x is saved in a compressed JLD2 file.
#
# The solvers, parameters, timing and success tests are copies of those of runbenchmark.jl,
# which is left unchanged, with these differences:
# - the ℓ1-Homotopy HOC variant and BP_MAP+Bin are not run;
# - when a short run of BP-MAP without HOC is re-timed, HOC is applied to zMAP, as in the
#   solve itself (runbenchmark.jl applies it to xMAP);
# - an exception in one solver is recorded as status "error" (with time -Inf) instead of
#   stopping the whole run;
# - a solve of a small random problem first keeps compilation out of the first time.
#
# Usage, from the project directory:
#
#     julia --project test/runbenchmarkprecvals.jl [--testset=LSS|LPT] [--single]
#
# --testset picks LSS (the default: BP_HOC+Bin on CPU and GPU, HiGHS, ISAL and L1Homotopy,
# with 4 right-hand sides) or LPT (BP_MAP, BP_HOC, BP_HOC+Bin, HiGHS, ISAL and L1Homotopy).
# The results go to LSS_benchmark_precvals_<rhs>.csv or LPT_benchmark_precvals.csv in the
# current directory, and x to data/exp_pro/<testset>_solutions/<solver>/<problem>_rhs<k>.jld2.
# Times are measured as in runbenchmark.jl, repeating short runs; --single times a single
# run of each solver instead. The CSV is saved after every problem, and a restarted run
# skips the problems already in it.

using DrWatson
@quickactivate :BP_MAP

mattype::MatType = automat

using LinearAlgebra
using SparseArrays
using Statistics
using Printf
using Glob
using BenchmarkTools
using DataFrames
using CSV
using CUDA
using JLD2, CodecZlib

include(scriptsdir("BP_LP.jl"))
include(scriptsdir("BP_ISAL1.jl"))
include(scriptsdir("BP_L1Homotopy.jl"))

"Relative error assuming that b is not 0"
function relerror(a, b)
    return norm(a - b, Inf) / norm(b, Inf)
end

function dist2sol(x, prob)
    dist = length(prob.sol) > 0 && !isnan(prob.sol[1]) ? relerror(x, prob.sol) : 0.0
    @info "L1 norm = $(norm(x, 1))"
    @info "Feasibility = $(norm(prob.A*x - prob.b, Inf))"
    if dist > 0.0
        @info "Relative distance to solution = $dist"
    end
    return dist
end

# Support-thresholding tolerance for the heuristic optimality check (Lorenz et al.
# 2015 §3.6). Shared by every solver that runs HOC, so that the comparison is not
# skewed by one of them polishing at a different threshold.
const TOL_HOC = 1.0e-10

# Data directory, right-hand sides and BP-MAP variants (usehoc, binsearch, acceleration)
# of each test set, as in the paper runs
const TESTSETS = Dict(
    "LSS" => (
        dir = "lassobp_mat",
        rhs = 1:4,
        variants = [(true, true, noaccel), (true, true, CUDAaccel)],
    ),
    "LPT" => (
        dir = "L1_Testset_mat",
        rhs = 1:1,
        variants = [(false, false, noaccel), (true, false, noaccel), (true, true, noaccel)],
    ),
)

resultfile(testset, rhs) =
    testset == "LSS" ? "LSS_benchmark_precvals_$rhs.csv" : "LPT_benchmark_precvals.csv"

# Each solve_with_* function returns the time (negative when not solved), the distance to
# the reference solution, x, the status (success, failure or timeout), the solver's exit
# flag and, when the solver reports them, its iteration counts. With timed = false the
# time is that of a single run.

function solve_with_BPMAP(prob, usehoc, binsearch, timed)
    itmax = 2000
    tol, success_prec, tol_HOC = 1.0e-6, 1.0e-4, TOL_HOC

    solver_name = bpname(usehoc, binsearch, noaccel) * " "
    @info solver_name * "-"^(70 - length(solver_name))
    @info "Matrix type = $(typeof(prob.accelA))"
    duration = @elapsed xMAP, zMAP, it, inner_it, status = solveBP_MAP(
        prob,
        itmax = itmax,
        ε = tol,
        ε_MAP = tol,
        δ_HOC = tol_HOC,
        usehoc = usehoc,
        usebinsearch = binsearch,
    )
    @info "BP-MAP status is $status with $(it) iterations and $(inner_it) inner iterations"
    dist = dist2sol(xMAP, prob)

    if !usehoc
        @info "Applying HOC"
        duration += @elapsed xSol_HOC, status_hoc =
            heuristic_optimality_check(zMAP, prob, δ = tol_HOC)
        dist_HOC = dist2sol(xSol_HOC, prob)
        if status_hoc == :success
            xMAP, dist, status = xSol_HOC, dist_HOC, :Solved
        end
    end

    solved = status == :Solved && dist < success_prec

    if timed && duration < 10
        t = @benchmark begin
            solveBP_MAP(
                $prob,
                itmax = $itmax,
                ε = $tol,
                ε_MAP = $tol,
                δ_HOC = $tol_HOC,
                usehoc = $usehoc,
                usebinsearch = $binsearch,
            )
            # HOC on zMAP, as in the solve above (runbenchmark.jl uses xMAP here)
            if !$usehoc
                heuristic_optimality_check($zMAP, $prob, δ = $tol_HOC)
            end
        end seconds = 10
        duration = median(t.times) / 1.0e9
    end
    duration = solved ? duration : -duration
    # solveBP_MAP stops as :Tired both on the iteration and on the time limits
    label = solved ? "success" : (status == :Tired && it < itmax) ? "timeout" : "failure"

    @info @sprintf("Elapsed CPU time for BP_MAP %.4f s", duration)
    return (
        time = duration,
        dist = dist,
        iters = it,
        inner_iters = inner_it,
        x = xMAP,
        status = label,
        flag = string(status),
    )
end

function solve_with_LP(prob, timed)
    solver_name = "LP (highs) "
    @info solver_name * "-"^(70 - length(solver_name))
    lpsolver = HiGHS.Optimizer
    model = buildBP_LPModel(prob, solver = lpsolver)
    set_time_limit_sec(model, 3600)
    duration = @elapsed lp_sol = solveBP_LPmodel!(model)
    solved = is_solved_and_feasible(model)
    flag = termination_status(model)
    dist = dist2sol(lp_sol, prob)
    @info "Elapsed CPU time for solving with LP (highs) Solver"
    if timed && duration < 60
        t = @benchmark begin
            set_optimizer($model, $lpsolver)
            solveBP_LPmodel!($model)
        end
        duration = median(t.times) / 1.0e+9
    end
    duration = solved ? duration : -duration
    label = solved ? "success" : flag == TIME_LIMIT ? "timeout" : "failure"

    @info @sprintf("Elapsed CPU time for LP %.4f s", duration)
    return (time = duration, dist = dist, x = lp_sol, status = label, flag = string(flag))
end

function solve_with_ISAL(prob, timed)
    solver_name = "ISAL "
    @info solver_name * "-"^(70 - length(solver_name))
    @info "Elapsed CPU time for solving with ISAL1 Solver"
    xISAL, duration, it_ISAL, status_ISAL = solveBP_ISAL1(prob, compute_time = timed)
    dist = dist2sol(xISAL, prob)
    # ISAL gets the benefit of the doubt: only exit flags 0 (something went wrong) and ±2
    # (stagnation) count as failures; ±4 is its time limit
    solved = status_ISAL ∉ [0, 2, -2] && dist < 1.0e-4
    duration = solved ? duration : -duration
    flag = round(Int, status_ISAL)
    label = solved ? "success" : abs(flag) == 4 ? "timeout" : "failure"

    @info @sprintf("Median = %.4f s", duration)
    return (
        time = duration,
        dist = dist,
        iters = round(Int, it_ISAL),
        x = xISAL,
        status = label,
        flag = string(flag),
    )
end

function solve_with_L1Homotopy(prob, timed)
    tol, disttol = 1.0e-6, 1.0e-4
    solver_name = "L1Homotopy "
    @info solver_name * "-"^(70 - length(solver_name))
    @info "Elapsed CPU time for solving with L1Homotopy Solver"
    xL1H, duration, it_L1H, status_L1H =
        solveBP_L1Homotopy(prob; δ_HOC = TOL_HOC, tau = 1.0e-9, compute_time = timed)
    @info "L1Homotopy status is $status_L1H with $(it_L1H) iterations"
    dist = dist2sol(xL1H, prob)
    feasible = norm(prob.A * xL1H - prob.b, Inf) / max(norm(prob.b, Inf), 1.0) <= tol

    # A run that hit the wall-clock budget (:maxtime), ran out of homotopy steps (:maxiter)
    # or broke down (:failed) is unsolved; otherwise the homotopy path must have converged
    # to a feasible and accurate point
    solved = status_L1H == :success || (status_L1H == :solved && feasible && dist < disttol)
    duration = solved ? duration : -duration
    label = solved ? "success" : status_L1H == :maxtime ? "timeout" : "failure"

    @info @sprintf("Median = %.4f s", duration)
    return (
        time = duration,
        dist = dist,
        iters = round(Int, it_L1H),
        x = xL1H,
        status = label,
        flag = string(status_L1H),
    )
end

function bpname(usehoc, binsearch, acceleration)
    name = usehoc ? "BP_HOC" : "BP_MAP"
    if binsearch
        name *= "+Bin"
    end
    if acceleration == CUDAaccel
        name *= "+CUDA"
    end
    return name
end

"Save x compressed; the temporary file keeps an interrupted run from leaving a partial one"
function save_x(file, x)
    mkpath(dirname(file))
    jldsave(file * ".part", true; x)
    mv(file * ".part", file, force = true)
end

"""
Run `solve` (a solve_with_* call) and store its results for `solver` in `row`, adding the
objective ‖x‖₁ and the relative feasibility ‖Ax - b‖₂ / ‖b‖₂, computed in the same way for
all solvers. The x found is saved in `xfile`.
"""
function record!(row, solver, prob, xfile, solve)
    try
        r = solve()
        x = Vector{Float64}(r.x)
        obj, rel_feas = norm(x, 1), norm(prob.A * x - prob.b) / norm(prob.b)
        @info @sprintf("%s: obj = %.10e, rel_feas = %.3e", r.status, obj, rel_feas)
        row[solver] = r.time
        row["$solver dist"] = r.dist
        haskey(r, :iters) && (row["$solver iters"] = r.iters)
        haskey(r, :inner_iters) && (row["$solver inner_iters"] = r.inner_iters)
        row["$solver obj"] = obj
        row["$solver rel_feas"] = rel_feas
        row["$solver status"] = r.status
        row["$solver flag"] = r.flag
        save_x(xfile, x)
    catch e
        e isa InterruptException && rethrow()
        @error "$solver failed" exception = (e, catch_backtrace())
        row[solver] = -Inf
        row["$solver dist"] = row["$solver obj"] = row["$solver rel_feas"] = NaN
        row["$solver status"] = "error"
        row["$solver flag"] = first(split(sprint(showerror, e), '\n'))
    end
end

"Save with the column order of runbenchmark.jl, writing a temporary file first"
function save_results(df, resfile)
    first_cols = ["Problem", "M", "N"]
    df = select(df, first_cols, sort(setdiff(names(df), first_cols)))
    CSV.write(resfile * ".part", df)
    mv(resfile * ".part", resfile, force = true)
end

function run_benchmark(testset, resfile, rhs, variants, timed, soldir)
    df = isfile(resfile) ? CSV.read(resfile, DataFrame, stringtype = String) : DataFrame()
    done = Set(nrow(df) > 0 ? df.Problem : String[])
    xfile(solver, prob_name) = joinpath(soldir, solver, splitext(prob_name)[1] * "_rhs$rhs.jld2")

    testnum = 0
    ntests = length(testset)
    for instance in testset
        println("="^78)

        # Read test
        testnum += 1
        prob_name = basename(instance)
        key = prob_name * " - $rhs"
        if key in done
            @info "$key is already in $resfile"
            continue
        end
        prob = readl1test(instance; rhs = rhs, mattype = mattype)
        m, n = size(prob)
        row = Dict{String,Any}("Problem" => key, "M" => m, "N" => n)
        @info "Problem $(prob_name) - size: $(size(prob))"
        @info @sprintf("%d / %d (%.2g %%)", testnum, ntests, 100.0 * (testnum / ntests))

        # BP_MAP
        for (h, b, a) in variants
            bpprob = readl1test(instance; rhs = rhs, mattype = mattype, acceltype = a)
            name = bpname(h, b, a)
            record!(row, name, bpprob, xfile(name, prob_name), () -> solve_with_BPMAP(bpprob, h, b, timed))
        end

        # Read problem again as GPU is not supported by LP or ISAL
        cpuprob = readl1test(instance; rhs = rhs, mattype = mattype, acceltype = noaccel)
        record!(row, "HiGHS", cpuprob, xfile("HiGHS", prob_name), () -> solve_with_LP(cpuprob, timed))
        record!(row, "ISAL", cpuprob, xfile("ISAL", prob_name), () -> solve_with_ISAL(cpuprob, timed))
        record!(
            row,
            "L1Homotopy",
            cpuprob,
            xfile("L1Homotopy", prob_name),
            () -> solve_with_L1Homotopy(cpuprob, timed),
        )

        df = vcat(df, DataFrame(Dict(k => [v] for (k, v) in row)), cols = :union)
        save_results(df, resfile)
    end
    println("="^78)
    println(df)
    return df
end

"Solve a small random problem with the Julia solvers so that compilation stays out of the first time"
function warmup(variants)
    A = sprandn(50, 200, 0.3)
    b = A * [ones(5); zeros(195)]
    try
        for (h, bs, a) in variants
            solve_with_BPMAP(BPProblem(A, b; acceltype = a), h, bs, false)
        end
        solve_with_LP(BPProblem(A, b), false)
    catch e
        @warn "Warm up failed" exception = e
    end
end

function run_benchmarks(testset = "LSS", timed = true)
    set = TESTSETS[testset]
    if any(v -> v[3] == CUDAaccel, set.variants) && !CUDA.functional()
        error("CUDA is not functional here, and the $testset test set runs BP-MAP on the GPU")
    end
    files = glob("*.mat", datadir("exp_raw", set.dir))
    soldir = datadir("exp_pro", "$(testset)_solutions")
    warmup(set.variants)
    for rhs in set.rhs
        @info "Test set $testset, right-hand side $rhs" * (timed ? "" : ", single runs")
        run_benchmark(files, resultfile(testset, rhs), rhs, set.variants, timed, soldir)
    end
end

function parse_args(args)
    testset, timed = "LSS", true
    for arg in args
        if arg == "--single"
            timed = false
        elseif startswith(arg, "--testset=")
            testset = String(split(arg, "=")[2])
            haskey(TESTSETS, testset) || error("Test set must be in $(keys(TESTSETS)), got $testset")
        else
            error("Unknown option $arg")
        end
    end
    return testset, timed
end


if abspath(PROGRAM_FILE) == @__FILE__
    # Downloads Tests sets
    include(scriptsdir("downloadtestsets.jl"))
    downloadtestsets()

    run_benchmarks(parse_args(ARGS)...)
end
