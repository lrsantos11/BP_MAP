using DrWatson
@quickactivate :BP_MAP

using Test
mattype::MatType = automat

using LinearAlgebra
using Statistics
using Printf
using Glob
using BenchmarkTools
using DataFrames
using CSV

include(scriptsdir("BP_LP.jl"))
using Gurobi
const gurobi_env = Gurobi.Env()

# Include the BP_ISAL1.jl scripts
include(scriptsdir("BP_ISAL1.jl"))

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

function solve_with_BPMAP(prob, usehoc = false, binsearch = false)
    itmax = 2000
    tol, success_prec, tol_HOC = 1.0e-6, 1.0e-4, 1.0e-10

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

    if duration < 10
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
            if !$usehoc
                heuristic_optimality_check($xMAP, $prob, δ = $tol_HOC)
            end
        end seconds = 10
        duration = median(t.times) / 1.0e9
    end
    duration = solved ? duration : -duration

    @info @sprintf("Elapsed CPU time for BP_MAP %.4f s", duration)
    return duration, dist
end

function solve_with_LP(prob, solvertype = :highs)
    # LP solver, use Gurobi if avaliable.
    solver_name = "LP ($solvertype) "
    @info solver_name * "-"^(70 - length(solver_name))
    if solvertype == :gurobi
        lpsolver = () -> Gurobi.Optimizer(gurobi_env)
    else
        lpsolver = HiGHS.Optimizer
    end
    model = buildBP_LPModel(prob, solver = lpsolver)
    set_time_limit_sec(model, 3600)
    duration = @elapsed lp_sol = solveBP_LPmodel!(model)
    solved = is_solved_and_feasible(model)
    dist = dist2sol(lp_sol, prob)
    @info "Elapsed CPU time for solving with LP ($solvertype) Solver"
    if duration < 60
        t = @benchmark begin
            set_optimizer($model, $lpsolver)
            solveBP_LPmodel!($model)
        end
        duration = median(t.times) / 1.0e+9
    end
    duration = solved ? duration : -duration

    @info @sprintf("Elapsed CPU time for LP %.4f s", duration)
    return duration, dist
end

function solve_with_ISAL(prob)
    solver_name = "ISAL "
    @info solver_name * "-"^(70 - length(solver_name))
    @info "Elapsed CPU time for solving with ISAL1 Solver"
    xISAL, duration, it_ISAL, status_ISAL = solveBP_ISAL1(prob)
    dist = dist2sol(xISAL, prob)
    solved = status_ISAL ∉ [0, 2, -2] && dist < 1.0e-6
    duration = solved ? duration : -duration

    @info @sprintf("Median = %.4f s", duration)
    return duration, dist
end

function save_results(results, resfile)
    columns = [k for k in filter(x -> x ∉ ["Problem", "M", "N"], keys(results))]
    sort!(columns)
    columns = vcat(["Problem", "M", "N"], columns)

    # Convert to DataFrame and save to a file
    LPT_data = DataFrame(results)
    LPT_data = LPT_data[!, columns]
    println(LPT_data)
    CSV.write(resfile, LPT_data)
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

function run_benchmark(
    testset,
    resfile,
    rhs = 1,
    usehoc = [true],
    binsearch = [true],
    acceleration = [noaccel],
)
    results = Dict(
        "Problem" => String[],
        "M" => Int[],
        "N" => Int[],
        "Gurobi" => Float64[],
        "Gurobi dist" => Float64[],
        "HiGHS" => Float64[],
        "HiGHS dist" => Float64[],
        "ISAL" => Float64[],
        "ISAL dist" => Float64[],
    )
    for h in usehoc, b in binsearch, a in acceleration
        results[bpname(h, b, a)] = Float64[]
        results[bpname(h, b, a)*" dist"] = Float64[]
    end

    testnum = 0
    ntests = length(testset)
    savestep = max(5, ntests ÷ 20)
    for instance in testset
        println("="^78)

        # Read test
        testnum += 1
        prob_name = basename(instance)
        push!(results["Problem"], prob_name * " - $rhs")
        prob = readl1test(instance; rhs = rhs, mattype = mattype)
        m, n = size(prob)
        push!(results["M"], m)
        push!(results["N"], n)
        @info "Problem $(prob_name) - size: $(size(prob))"
        @info @sprintf("%d / %d (%.2g %%)", testnum, ntests, 100.0 * (testnum / ntests))

        # BP_MAP
        for h in usehoc, b in binsearch, a in acceleration
            prob = readl1test(instance; rhs = rhs, mattype = mattype, acceltype = a)

            duration, dist = solve_with_BPMAP(prob, h, b)
            push!(results[bpname(h, b, a)], duration)
            push!(results[bpname(h, b, a)*" dist"], dist)
        end

        # Read problem again as GPU is not supported by LP or ISAL
        prob = readl1test(instance; rhs = rhs, mattype = mattype, acceltype = noaccel)

        # Linear programming
        duration, dist = solve_with_LP(prob, :gurobi)
        push!(results["Gurobi"], duration)
        push!(results["Gurobi dist"], dist)
        duration, dist = solve_with_LP(prob, :highs)
        push!(results["HiGHS"], duration)
        push!(results["HiGHS dist"], dist)

        # ISAL
        duration, dist = solve_with_ISAL(prob)
        push!(results["ISAL"], duration)
        push!(results["ISAL dist"], dist)

        if testnum % savestep == 0
            save_results(results, resfile)
        end
    end
    println("="^78)

    save_results(results, resfile)
    return results
end

function run_benchmarks()
    @info "Testset from Lorentz, Pfetsch, and Tillmann"
    LPT_testset = glob("*.mat", datadir("exp_raw", "L1_Testset_mat"))
    run_benchmark(LPT_testset, "LPT_benchmark.csv", 1, [false, true], [false, true], [noaccel])

    @info "Testset based on Lopes, Santos, and Silva"
    LSS_testset = glob("*.mat", datadir("exp_raw", "lassobp_mat"))
    for rhs = 1:4
        run_benchmark(LSS_testset, "LSS_benchmark_CUDA_$rhs.csv", rhs, [true], [true], [CUDAaccel])
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    # Downloads Tests sets
    include(scriptsdir("downloadtestsets.jl"))
    downloadtestsets()

    run_benchmarks()
end