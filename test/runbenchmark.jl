using DrWatson
@quickactivate :BP_MAP

using Test
acceleration::AccelerationTarget = noaccel 
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

# Define the LP solver to use :gurobi or :HiGHS
solvertype = :gurobi
# Set a global gurobi enviroment to supress multiple messages
if solvertype == :gurobi
    global gurobi_env = Gurobi.Env()
end

# Include the BP_ISAL1.jl scripts
include(scriptsdir("BP_ISAL1.jl"))

"Relative error assuming that b is not 0"
function relerror(a, b)
    return norm(a - b, Inf) / norm(b, Inf)
end

function solve_with_BPMAP(prob, usehoc = false)
    itmax = 2000
    tol, success_prec, tol_HOC = 1.0e-6, 1.0e-4, 1.0e-10

    @info "BP_MAP, HOC = $usehoc " * "-"^60
    duration = @elapsed xMAP, zMAP, it, inner_it, status = solveBP_MAP(
        prob,
        itmax = itmax,
        ε = tol,
        ε_MAP = tol,
        δ_HOC = tol_HOC,
        usehoc = usehoc,
    )
    @info "BP-MAP status is $status with $(it) iterations and $(inner_it) inner iterations"
    @info "xMAP - sol = $(relerror(xMAP, prob.sol))"

    if !usehoc
        @info "Applying HOC"
        xSol_HOC, status_hoc = heuristic_optimality_check(zMAP, prob, δ = tol_HOC)
        @info "xSol_HOC - sol = $(relerror(xSol_HOC, prob.sol))"
        if status_hoc == :success
            xMAP = xSol_HOC
        end
    end

    dist = relerror(xMAP, prob.sol)
    solved = dist < success_prec

    if duration < 10
        t = @benchmark begin
            solveBP_MAP(
                $prob,
                itmax = $itmax,
                ε = $tol,
                ε_MAP = $tol,
                δ_HOC = $tol_HOC,
                usehoc = $usehoc,
            )
            heuristic_optimality_check($xMAP, $prob, δ = $tol_HOC)
        end seconds = 10
        duration = median(t.times)
    else
        duration *= 1.0e9
    end
    @info @sprintf("Elapsed CPU time for BP_MAP %.4f s", duration / 1.0e9)
    return solved ? duration : -duration, dist
end

function save_results(results)
    # Convert to DataFrame and save to a file
    LPT_data = DataFrame(results)
    # Convert time to seconds
    # LPT_data[!, [:LP, :BP_MAP, :BP_HOC]] ./= 1.0e9
    LPT_data[!, [:BP_MAP, :BP_HOC]] ./= 1.0e9
    println(LPT_data)
    CSV.write("LPT_benchmark_auto_tmp.csv", LPT_data)
end

function lpt_bechmark()
    # USe the testset from Lorentz, Pfetsch, and Tillmann
    LPT_testset = glob("*.mat", datadir("exp_raw", "L1_Testset_mat"));

    results = Dict(
        :Problem => String[],
        :M => Int[],
        :N => Int[],
        :BP_MAP => Float64[],
        :BP_HOC => Float64[],
        # :LP => Float64[],
        # :ISAL => Float64[],
        :BP_MAP_dist => Float64[],
        :BP_HOC_dist => Float64[],
    #     :LP_dist => Float64[],
    #     :ISAL_dist => Float64[],
    )

    testnum = 0
    ntests = length(LPT_testset)
    savestep = ntests ÷ 20
    for instance in LPT_testset
        # Read test
        testnum += 1
        prob_name = basename(instance)
        push!(results[:Problem], prob_name)
        prob_path = joinpath("L1_Testset_mat", prob_name)
        prob = readl1test(prob_path; mattype = mattype, acceltype = acceleration)
        m, n = size(prob)
        push!(results[:M], m)
        push!(results[:N], n)
        sol = prob.sol
        @info "Problem $(prob_name) - size: $(size(prob))"
        @info @sprintf("%d / %d (%.2g %%)", testnum, ntests, 100.0 * (testnum / ntests))
        @info "Acceleration Matrix type: $(typeof(prob.accelA))"

        # BP_MAP + HOC
        duration, dist = solve_with_BPMAP(prob, false)
        push!(results[:BP_MAP], duration)
        push!(results[:BP_MAP_dist], dist)

        # BP_MAP with HOC
        duration, dist = solve_with_BPMAP(prob, true)
        push!(results[:BP_HOC], duration)
        push!(results[:BP_HOC_dist], dist)

        # # Read problem again as GPU is not supported by LP or ISAL
        # prob = readl1test(prob_path; mattype = mattype, acceltype = noaccel)

        # # LP solver, use Gurobi if avaliable.
        # @info "LP " * "-"^60
        # if solvertype == :gurobi
        #     solver = () -> Gurobi.Optimizer(gurobi_env)
        # else
        #     solver = HiGHS.Optimizer
        # end
        # model = buildBP_LPModel(prob, solver = solver)
        # lp_sol = solveBP_LPmodel!(model)
        # solved = is_solved_and_feasible(model)
        # dist = relerror(lp_sol, sol)
        # @info "Elapsed CPU time for solving with LP Solver"
        # t = @benchmark begin
        #     set_optimizer($model, $solver)
        #     set_silent($model)
        #     solveBP_LPmodel!($model)
        # end
        # push!(results[:LP], solved ? median(t.times) : -median(t.times))
        # push!(results[:LP_dist], dist)
        # @info @sprintf("Median = %.4f s", results[:LP][end] / 1.0e9)

        # # ISAL1
        # @info "ISAL1 " * "-"^60
        # @info "Elapsed CPU time for solving with ISAL1 Solver"
        # xISAL, time_ISAL, it_ISAL, status_ISAL = solveBP_ISAL1(prob)
        # dist = relerror(xISAL, sol)
        # @info "Dist to solution = $dist"
        # solved = dist < 1.0e-6
        # @info @sprintf("Mean = %.4f s", time_ISAL)
        # push!(results[:ISAL], solved ? time_ISAL : -time_ISAL)
        # push!(results[:ISAL_dist], dist)

        println("="^72)
        if testnum % savestep == 0
            save_results(results)
        end
    end
    save_results(results)
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    # Downloads Tests sets
    include(scriptsdir("downloadtestsets.jl"))
    downloadtestsets()
   
    lpt_bechmark() 
end