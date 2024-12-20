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
    return norm(a - b) / norm(b)
end

##
# Downloads Tests sets
include(scriptsdir("downloadtestsets.jl"))
downloadtestsets()

# USe the testset from Lorentz, Pfetsch, and Tillmann
LPT_testset = glob("*.mat", datadir("exp_raw", "L1_Testset_mat"));

pnames = String[]
pms, pns = Int[], Int[]
bp_map_dist, bp_hoc_dist, lp_dist, isal_dist = Float64[], Float64[], Float64[], Float64[]
bp_map_times, bp_hoc_times, lp_times, isal_times =
    Float64[], Float64[], Float64[], Float64[]

itmax = 2000
tol, tol_HOC = 1e-6, 1.0e-10
success_prec = 10 * tol
ntests = length(LPT_testset)
testnum = 0
for instance in LPT_testset
    # Read test
    global testnum += 1
    prob_name = basename(instance)
    push!(pnames, prob_name)
    prob_path = joinpath("L1_Testset_mat", prob_name)
    prob = readl1test(prob_path; mattype = mattype, acceltype = acceleration)
    m, n = size(prob)
    push!(pms, m)
    push!(pns, n)
    sol = prob.sol
    @info "Problem $(prob_name) - size: $(size(prob))"
    @info @sprintf("%d / %d (%.2g %%)", testnum, ntests, 100.0*(testnum / ntests))
    @info "Acceleration Matrix type: $(typeof(prob.accelA))"

    # BP_MAP followed by a HOC
    @info "BP_MAP + HOC " * "-"^60
    xMAP, zMAP, it, inner_it, status =
        solveBP_MAP(prob, itmax = itmax, ε = tol, ε_MAP = tol)
    @info "BP-MAP status is $status with  $(it) iterations and $(inner_it) inner iterations"
    @info "xMAP - sol = $(relerror(xMAP, sol))"
    @info "Applying HOC"
    xSol_HOC, status_hoc = heuristic_optimality_check(zMAP, prob, δ = tol_HOC)
    @info "xSol_HOC - sol = $(norm(xSol_HOC - sol, 2) / norm(sol, 2))"
    dist = status_hoc == :success ? relerror(xSol_HOC, sol) : relerror(xMAP, sol)
    solved = (status_hoc == :success) || (dist < success_prec)
    @info "Elapsed CPU time for BP_MAP + HOC"
    t = @benchmark begin
        solveBP_MAP($prob, itmax = $itmax, ε = $tol, ε_MAP = $tol)
        heuristic_optimality_check($xMAP, $prob, δ = $tol_HOC)
    end
    push!(bp_map_times, solved ? median(t.times) : -median(t.times))
    push!(bp_map_dist, dist)
    @info @sprintf("Median = %.2f s", bp_map_times[end] / 1.0e9)

    # BP_MAP with HOC
    @info "BP_MAP with HOC " * "-"^60
    xMAP, zMAP, it, inner_it, status =
        solveBP_MAP(prob, itmax = itmax, ε = eps(1.0), ε_MAP = tol, usehoc = true)
    @info "BP-MAP with HOC status is $status with  $(it) iterations and $(inner_it) inner iterations"
    @info "xMAP - sol = $(relerror(xMAP, sol))"
    dist = relerror(xMAP, sol)
    solved = dist < success_prec
    @info "Elapsed CPU time for BP_MAP with HOC"
    t = @benchmark solveBP_MAP($prob, itmax = $itmax, ε = $tol, ε_MAP = $tol, usehoc = true)
    push!(bp_hoc_times, solved ? median(t.times) : -median(t.times))
    push!(bp_hoc_dist, dist)
    @info @sprintf("Median = %.2f s", bp_hoc_times[end] / 1.0e9)

    # Read problem again as GPU is not supported by LP or ISAL
    prob = readl1test(prob_path; mattype = mattype, acceltype = noaccel)

    # LP solver, use Gurobi if avaliable.
    @info "LP " * "-"^60
    if solvertype == :gurobi
        solver = () -> Gurobi.Optimizer(gurobi_env)
    else
        solver = HiGHS.Optimizer
    end
    model = buildBP_LPModel(prob, solver = solver)
    lp_sol = solveBP_LPmodel!(model)
    solved = is_solved_and_feasible(model)
    dist = relerror(lp_sol, sol)
    @info "Elapsed CPU time for solving with LP Solver"
    t = @benchmark begin
        set_optimizer($model, $solver)
        set_silent($model)
        solveBP_LPmodel!($model)
    end
    push!(lp_times, solved ? median(t.times) : -median(t.times))
    push!(lp_dist, dist)
    @info @sprintf("Median = %.2f s", lp_times[end] / 1.0e9)

    # ISAL1
    @info "ISAL1 " * "-"^60
    @info "Elapsed CPU time for solving with ISAL1 Solver"
    xISAL, time_ISAL, it_ISAL, status_ISAL = solveBP_ISAL1(prob)
    dist = relerror(xISAL, sol)
    solved = dist < success_prec
    @info @sprintf("Mean = %.2f s", time_ISAL)
    push!(isal_times, solved ? time_ISAL : -time_ISAL)
    push!(isal_dist, dist)

    println("="^72)
end

# Convert to DataFrame and save to a file
LPT_data = DataFrame(
    Problems = pnames,
    M = pms,
    N = pns,
    LP = lp_times,
    LP_dist = lp_dist,
    ISAL = isal_times,
    ISAL_dist = isal_dist,
    BP_MAP = bp_map_times,
    BP_MAP_dist = bp_map_dist,
    BP_HOC = bp_hoc_times,
    BP_HOC_dist = bp_hoc_dist,
)
# Convert time to seconds
LPT_data[!, [:LP, :BP_MAP, :BP_HOC]] ./= 1.0e9
println(LPT_data)
CSV.write("LPT_benchmark_auto.csv", LPT_data)
