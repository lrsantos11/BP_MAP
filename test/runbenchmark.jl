using DrWatson, Test
@quickactivate :BP_MAP
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

# Define the LP solver to use
# solvertype = :gurobi
solvertype = :HiGHS
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
# Downloads Tests from from the Lorentz, Pfetsch, and Tillmann and Lopes, Santos 
# and Silva collections
include(scriptsdir("downloadtestsets.jl"))

# LPT tests

LPT_testset = glob("*.mat", datadir("exp_raw", "L1_Testset_mat"));
# This is an example where BP_MAP fails if it does not use HOC

pnames = String[]
bp_map_dist, bp_hoc_dist, lp_dist = Float64[], Float64[], Float64[]
bp_map_times, bp_hoc_times, lp_times = Float64[], Float64[], Float64[]

itmax = 2000
tol, tol_HOC = 1e-6, 1.0e-10
success_prec = 1.0e-5
for instance in LPT_testset
    prob_name = basename(instance)
    push!(pnames, prob_name)
    prob = readl1test(
        joinpath("L1_Testset_mat", prob_name);
        mattype = mattype,
        acceltype = acceleration,
    )
    sol = prob.sol
    @info "Problem $(prob_name) - size: $(size(prob))"
    @info "Acceleration Matrix type: $(typeof(prob.accelA))"

    @info "BP_MAP + HOC" * "-"^25
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
    push!(bp_map_times, solved ? median(t.times) : -mean(t.times))
    push!(bp_map_dist, dist)
    @info @sprintf("Median = %.2f s", bp_map_times[end] / 1.0e9)

    @info "BP_MAP with HOC" * "-"^25
    xMAP, zMAP, it, inner_it, status =
        solveBP_MAP(prob, itmax = itmax, ε = eps(1.0), ε_MAP = tol, usehoc = true)
    @info "BP-MAP with HOC status is $status with  $(it) iterations and $(inner_it) inner iterations"
    @info "xMAP - sol = $(relerror(xMAP, sol))"
    dist = relerror(xMAP, sol)
    solved = dist < success_prec
    @info "Elapsed CPU time for BP_MAP with HOC"
    t = @benchmark solveBP_MAP($prob, itmax = $itmax, ε = $tol, ε_MAP = $tol, usehoc = true)
    push!(bp_hoc_times, solved ? median(t.times) : -mean(t.times))
    push!(bp_hoc_dist, dist)
    @info @sprintf("Median = %.2f s", bp_hoc_times[end] / 1.0e9)

    # Use Gurobi if avaliable.
    @info "LP" * "-"^25
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

    println("="^72)
end

# Convert to DataFrame and save to a file
LPT_data = DataFrame(
    Problems = pnames,
    LP = lp_times,
    LP_dist = lp_dist,
    BP_MAP = bp_map_times,
    BP_MAP_dist = bp_map_dist,
    BP_HOC = bp_hoc_times,
    BP_HOC_DIST = bp_hoc_dist,
)
# Convert time to seconds
LPT_data[!, [:LP, :BP_MAP, :BP_HOC]] ./= 1.0e9
println(LPT_data)
CSV.write("LPT_benchmark_auto.csv", LPT_data)
