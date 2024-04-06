using DrWatson, Test
@quickactivate :BP_MAP

cpu_model = Sys.cpu_info()[1].model
if occursin("Intel", cpu_model) || occursin("AMD", cpu_model)
    using MKL
end

using LinearAlgebra
using Glob
using BenchmarkTools
using JuMP
using HiGHS
# using Gurobi

# Define the LP solver to use
# solvertype = :gurobi 
solvertype = :HiGHS
# Set a global gurobi enviroment to supress multiple messages
if solvertype ==:gurobi
    global gurobi_env = Gurobi.Env()
end

"Relative error assuming that b is not 0"
function relerror(a, b)
    return norm(a - b) / norm(b)
end

##
@testset "Example B1 [HL2014]" begin
    @info "Example B.1 of Hesse and Luke 2014"
    itmax = 50
    A =
        [
            1.0 1.0 1.0 1.0 1.0 1.0 1.0 1
            1.0 1.0 1.0 1.0 -1.0 -1.0 -1.0 -1
            1.0 1.0 -1.0 -1.0 1.0 1.0 -1.0 -1
            1.0 -1.0 1.0 -1.0 1.0 -1.0 1.0 -1
            1.0 1.0 -1.0 -1.0 -1.0 -1.0 1.0 1
            1.0 -1.0 -1.0 1.0 1.0 -1.0 -1.0 1
            1.0 -1.0 1.0 -1.0 -1.0 1.0 -1.0 1
        ] ./ sqrt(8)
    sol = [10.0, 0, 0, 0, 0, 0, 0, 0]
    probB1_HL14 = BPProblem(sol, A)
    Affine = IndAffine(probB1_HL14)
    tol = 1e-3
    xMAP, zMAP, it, inner_it, status =
        solveBP_MAP(Affine, itmax = itmax, ε = tol, ε_MAP = tol, BP_solution = sol)
    @info "BP-MAP status is $status with  $(it) iterations and $(inner_it) inner iterations"
    @info "xMAP - sol = $(relerror(xMAP, sol))"
    @test status == :Solved
    @info "Using HOC"
    xSol, _ = heuristic_optimality_check(zMAP, Affine, δ = tol^4)
    @info "Solucao:  $xSol"
    @info "xSol - sol = $(relerror(xSol, sol))"
    @test xSol ≈ sol
    xMAP, zMAP, it, inner_it, status = solveBP_MAP(
        Affine,
        itmax = itmax,
        ε = tol,
        ε_MAP = tol,
        BP_solution = sol,
        usehoc = true,
    )
    @info "BP-MAP with HOC status is $status with  $(it) iterations and $(inner_it) inner iterations"
    @info "xMAP - sol = $(relerror(xMAP, sol))"
    @test status == :Solved
end

##
@testset "Example B2 [HL2014]" begin
    @info "Example B.2 of Hesse and Luke 2014"
    itmax = 500
    A = [
        1 -0.5 0
        0 0.5 -1
    ]
    m, n = size(A)
    b = [-5.0, 5]
    prob_B2_HL14 = BPProblem(A, b)
    # This problem has infinite solutions. BPMAP is converging to one of them.  
    sol = [-10 / 3, 10 / 3, -10 / 3]
    Affine = IndAffine(prob_B2_HL14)
    tol = 1e-3
    xMAP, zMAP, it, inner_it, status =
        solveBP_MAP(Affine, itmax = itmax, ε = tol, ε_MAP = tol)
    @info "BP-MAP status is $status with  $(it) iterations and $(inner_it) inner iterations"
    @info "Solution not unique. Not calling HOC here"
    @info "xMAP - sol = $(relerror(xMAP, sol))"
    @test status == :Solved
    @test xMAP ≈ sol
    xMAP, zMAP, it, inner_it, status =
        solveBP_MAP(Affine, itmax = itmax, ε = tol, ε_MAP = tol, usehoc = true)
    @info "BP-MAP status is $status with  $(it) iterations and $(inner_it) inner iterations"
    @info "xMAP - sol = $(relerror(xMAP, sol))"
    @test status == :Solved
end

##
# Downloads Tests from from the Lorentz, Pfetsch, and Tillmann and Lopes, Santos 
# and Silva collections
include(scriptsdir("downloadtestsets.jl"))

# LPT tests

LPT_testset = glob("*.mat", datadir("exp_raw", "L1_Testset_mat"));
# This is an example where BP_MAP fails if it does not use HOC
#pushfirst!(LPT_testset, "spear_inst_400.mat")

@testset "Instances from the LPT collection" begin
    itmax = 2000
    tol, tol_HOC = 1e-3, 1.0e-12
    for instance in LPT_testset[1:11]
        prob_name = basename(instance)
        prob = readl1test(joinpath("L1_Testset_mat", prob_name))
        affine = IndAffine(prob)
        @info "Problem $(prob_name) - size: $(size(prob.A))"

        xMAP, zMAP, it, inner_it, status =
            solveBP_MAP(affine, itmax = itmax, ε = tol, ε_MAP = tol, verbose = false)
        @info "BP-MAP status is $status with  $(it) iterations and $(inner_it) inner iterations"
        @info "xMAP - sol = $(relerror(xMAP, prob.sol))"
        @info "Applying HOC"
        xSol_HOC, status_hoc = heuristic_optimality_check(zMAP, affine, δ = tol_HOC)
        @info "xSol_HOC - sol = $(norm(xSol_HOC - prob.sol, 2) / norm(prob.sol, 2))"
        @test relerror(xSol_HOC, prob.sol) < 1e-8
        @info "Elapsed CPU time for BP_MAP + HOC"
        @btime begin
            affine = IndAffine($prob)
            solveBP_MAP(affine, itmax = $itmax, ε = $tol, ε_MAP = $tol)
            heuristic_optimality_check($xMAP, affine, δ = $tol_HOC)
        end

        xMAP, zMAP, it, inner_it, status =
            solveBP_MAP(affine, itmax = itmax, ε = eps(1.0), ε_MAP = tol, usehoc = true)
        @info "BP-MAP with HOC status is $status with  $(it) iterations and $(inner_it) inner iterations"
        @info "xMAP - sol = $(relerror(xMAP, prob.sol))"
        @info "Elapsed CPU time for BP_MAP with HOC"
        @btime begin
            affine = IndAffine($prob)
            solveBP_MAP(affine, itmax = $itmax, ε = $tol, ε_MAP = $tol, usehoc = true)
        end

        @info "Elapsed CPU time for solving with LP Solver"
        # Use Gurobi if avaliable.
        if solvertype == :gurobi
            solver = () -> Gurobi.Optimizer(gurobi_env) 
        else
            solver = HiGHS.Optimizer
        end
        model = buildBP_LPModel(prob, solver=solver)
        @btime begin
            m = copy($model)
            set_optimizer(m, $solver)
            set_silent(m)
            solveBP_LPmodel!(m)
        end

        println("="^10)
    end
end

# LSS tests

LSS_testset = glob("*.mat", datadir("exp_raw", "lassobp_mat"));
@testset "Instances from the LSS collection" begin
    itmax = 250 
    tol, tol_HOC = 1.0e-3, 1.0e-12
    for instance in ["SC6.mat"] #LSS_testset[3:4]
        prob_name = basename(instance)
        prob = readl1test(joinpath("lassobp_mat", prob_name))
        @info "Problem $(prob_name) - size: $(size(prob.A))"
        @info "Starting first QR factorization"
        affine = IndAffine(prob)
        @info "End factorization"

        xMAP, zMAP, it, inner_it, status =
            solveBP_MAP(affine, itmax = itmax, ε = tol, ε_MAP = tol)
        @info "BP-MAP status is $status with  $(it) iterations and $(inner_it) inner iterations"
        @info "Applying HOC"
        xSol_HOC, status_hoc = heuristic_optimality_check(zMAP, affine, δ = tol_HOC)
        @info "HOC status = $status_hoc"
        @info "Feasibility = $(relerror(prob.A*xSol_HOC, prob.b))"
        @info "Optimal value = $(norm(xSol_HOC, 1))"
        @info "Elapsed CPU time for BP_MAP + HOC"
        @btime begin
            affine = IndAffine($prob)
            solveBP_MAP(affine, itmax = $itmax, ε = $tol, ε_MAP = $tol)
            heuristic_optimality_check($xMAP, affine, δ = $tol_HOC)
        end

        xMAP, zMAP, it, inner_it, status =
            solveBP_MAP(affine, itmax = itmax, ε = eps(1.0), ε_MAP = tol, usehoc = true)
        @info "BP-MAP with HOC status is $status with  $(it) iterations and $(inner_it) inner iterations"
        @info "Feasibility = $(relerror(prob.A*xMAP, prob.b))"
        @info "Optimal value = $(norm(xMAP, 1))"
        @info "Elapsed CPU time for BP_MAP with HOC"
        @btime begin
            affine = IndAffine($prob)
            solveBP_MAP(affine, itmax = $itmax, ε = $tol, ε_MAP = $tol, usehoc = true)
        end

        @info "Elapsed CPU time for solving with LP Solver"
        # Use Gurobi if avaliable.
        if solvertype == :gurobi
            solver = () -> Gurobi.Optimizer(gurobi_env) 
        else
            solver = HiGHS.Optimizer
        end
        model = buildBP_LPModel(prob, solver=solver)
        @btime begin
            m = copy($model)
            set_optimizer(m, $solver)
            set_silent(m)
            solveBP_LPmodel!(m)
            global optval = objective_value(m)
        end
        @info "Optimal value = $optval"

        println("="^10)
    end
end
