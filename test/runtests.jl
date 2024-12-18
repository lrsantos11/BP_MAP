using DrWatson, Test
@quickactivate :BP_MAP
acceleration::AccelerationTarget = CUDAaccel
mattype::MatType = automat

using LinearAlgebra
using Glob
using BenchmarkTools

include(scriptsdir("BP_LP.jl"))
using Gurobi

# Define the LP solver to use
# solvertype = :gurobi
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
# HL2014 tests

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
    tol = 1e-3
    xMAP, zMAP, it, inner_it, status =
        solveBP_MAP(probB1_HL14, itmax = itmax, ε = tol, ε_MAP = tol, BP_solution = sol)
    @info "BP-MAP status is $status with  $(it) iterations and $(inner_it) inner iterations"
    @info "xMAP - sol = $(relerror(xMAP, sol))"
    @test status == :Solved
    @info "Using HOC"
    xSol, _ = heuristic_optimality_check(zMAP, probB1_HL14, δ = tol^4)
    @info "Solucao: $xSol"
    @info "xSol - sol = $(relerror(xSol, sol))"
    @test xSol ≈ sol
    xMAP, zMAP, it, inner_it, status = solveBP_MAP(
        probB1_HL14,
        itmax = itmax,
        ε = tol,
        ε_MAP = tol,
        BP_solution = sol,
        usehoc = true,
    )
    @info "BP-MAP with HOC status is $status with  $(it) iterations and $(inner_it) inner iterations"
    @info "xMAP - sol = $(relerror(xMAP, sol))"
    @test status == :Solved
    xISAL, time_ISAL, it_ISAL, status_ISAL = solveBP_ISAL1(probB1_HL14, verbose = false)
    @info "ISAL with HOC status is $status_ISAL with  $(it_ISAL) iterations"
    @info "xISAL - sol = $(relerror(xISAL, sol))"
    @info "Elapsed CPU time for ISAL1: $time_ISAL"
end

##
@testset "Example B2 [HL2014]" begin
    @info "Example B.2 of Hesse and Luke 2014"
    itmax = 500
    A = [1 -0.5 0
        0 0.5 -1 ]
    m, n = size(A)
    b = [-5.0, 5]
    probB2_HL14 = BPProblem(A, b)
    # This problem has infinite solutions. BPMAP is converging to one of them.  
    sol = [-10 / 3, 10 / 3, -10 / 3]
    tol = 1e-3
    xMAP, zMAP, it, inner_it, status =
        solveBP_MAP(probB2_HL14, itmax = itmax, ε = tol, ε_MAP = tol)
    @info "BP-MAP status is $status with  $(it) iterations and $(inner_it) inner iterations"
    @info "Solution not unique. Not calling HOC here"
    @info "xMAP - sol = $(relerror(xMAP, sol))"
    @test status == :Solved
    @test xMAP ≈ sol
    xMAP, zMAP, it, inner_it, status =
        solveBP_MAP(probB2_HL14, itmax = itmax, ε = tol, ε_MAP = tol, usehoc = true)
    time_BP_MAP = @belapsed solveBP_MAP($probB2_HL14, itmax = $itmax, ε = $tol, ε_MAP = $tol, usehoc = true)
    @info "Elapsed CPU time for BP_MAP: $time_BP_MAP"
    @info "BP-MAP status is $status with  $(it) iterations and $(inner_it) inner iterations"
    @info "xMAP - sol = $(relerror(xMAP, sol))"
    @test status == :Solved
    xISAL, time_ISAL, it_ISAL, status_ISAL = solveBP_ISAL1(probB2_HL14, verbose = false)
    @info "xISAL - sol = $(relerror(xISAL, sol))"
    @info "Elapsed CPU time for ISAL1: $time_ISAL"
end

##
# Downloads Tests from from the Lorentz, Pfetsch, and Tillmann and Lopes, Santos 
# and Silva collections
include(scriptsdir("downloadtestsets.jl"))
downloadtestsets()

# LPT tests

LPT_testset = glob("*.mat", datadir("exp_raw", "L1_Testset_mat"));
# This is an example where BP_MAP fails if it does not use HOC
#pushfirst!(LPT_testset, "spear_inst_400.mat")

@testset "Instances from the LPT collection" begin
    itmax = 2000
    tol, tol_HOC = 1e-6, 1.0e-10
    for instance in [] # LPT_testset[81:85]
        prob_name = basename(instance)
        prob = readl1test(joinpath("L1_Testset_mat", prob_name); acceltype = acceleration)
        sol = prob.sol
        @info "Problem $(prob_name) - size: $(size(prob))"
        @info "Matrix type: $(typeof(prob.accelA))"

        xMAP, zMAP, it, inner_it, status =
            solveBP_MAP(prob, itmax = itmax, ε = tol, ε_MAP = tol)
        @info "BP-MAP status is $status with  $(it) iterations and $(inner_it) inner iterations"
        @info "xMAP - sol = $(relerror(xMAP, sol))"
        @info "Applying HOC"
        xSol_HOC, status_hoc = heuristic_optimality_check(zMAP, prob, δ = tol_HOC)
        @info "xSol_HOC - sol = $(norm(xSol_HOC - sol, 2) / norm(sol, 2))"
        @test relerror(xSol_HOC, sol) < 1e-8
        @info "Elapsed CPU time for BP_MAP + HOC"
        @btime begin
            solveBP_MAP($prob, itmax = $itmax, ε = $tol, ε_MAP = $tol)
            heuristic_optimality_check($xMAP, $prob, δ = $tol_HOC)
        end

        xMAP, zMAP, it, inner_it, status =
            solveBP_MAP(prob, itmax = itmax, ε = eps(1.0), ε_MAP = tol, usehoc = true)
        @info "BP-MAP with HOC status is $status with  $(it) iterations and $(inner_it) inner iterations"
        @info "xMAP - sol = $(relerror(xMAP, sol))"
        @info "Elapsed CPU time for BP_MAP with HOC"
        @btime solveBP_MAP($prob, itmax = $itmax, ε = $tol, ε_MAP = $tol, usehoc = true)

        @info "Elapsed CPU time for solving with LP Solver"
        # Use Gurobi if avaliable.
        if solvertype == :gurobi
            solver = () -> Gurobi.Optimizer(gurobi_env)
        else
            solver = HiGHS.Optimizer
        end
        model = buildBP_LPModel(prob, solver = solver)
        @btime begin
            m = copy($model)
            set_optimizer(m, $solver)
            set_silent(m)
            solveBP_LPmodel!(m)
        end

        @info "Elapsed CPU time for solving with ISAL1 Solver"
        xISAL, time_ISAL, it_ISAL, status_ISAL = solveBP_ISAL1(prob)
        @info "xISAL - sol = $(relerror(xISAL, sol))"
        @info "Elapsed CPU ISAL1:\n  $time_ISAL s"

        println("="^10)
    end
end

# LSS tests

LSS_testset = glob("*.mat", datadir("exp_raw", "lassobp_mat"));
@testset "Instances from the LSS collection" begin
    itmax = 2000
    tol, tol_HOC = 1.0e-6, 1.0e-10
    for instance in [["SC6.mat"]; LSS_testset]
        prob_name = basename(instance)
        prob = readl1test(joinpath("lassobp_mat", prob_name); rhs = 2, mattype = mattype, acceltype = acceleration)
        @info "Problem $(prob_name) - size: $(size(prob.A))"
        @info "Matrix type: $(typeof(prob.accelA))"

        # # BP_MAP + HOC
        # @info "Elapsed CPU time for solving with BP_MAP + HOC"
        # @time xMAP, zMAP, it, inner_it, status =
        #     solveBP_MAP(prob, itmax = itmax, ε = tol, ε_MAP = tol)
        # @info "BP-MAP status is $status with  $(it) iterations and $(inner_it) inner iterations"
        # @info "Applying HOC"
        # @time xSol_HOC, status_hoc = heuristic_optimality_check(zMAP, prob, δ = tol_HOC)
        # @info "HOC status = $status_hoc"
        # @info "Feasibility = $(relerror(prob.A*xSol_HOC, prob.b))"
        # @info "Objective = $(norm(xSol_HOC, 1))"
        # @info "Elapsed CPU time for BP_MAP + HOC"
        # # @btime begin
        # #     solveBP_MAP($prob, itmax = $itmax, ε = $tol, ε_MAP = $tol)
        # #     heuristic_optimality_check($xMAP, $prob, δ = $tol_HOC)
        # # end

        # LP
        @info "Elapsed CPU time for solving with LP Solver"
        # Use Gurobi if avaliable.
        if solvertype == :gurobi
            solver = () -> Gurobi.Optimizer(gurobi_env)
        else
            solver = HiGHS.Optimizer
        end
        model = buildBP_LPModel(prob, solver = solver, verbose = true)
        @time begin
            # m = copy($model)
            # set_optimizer(m, $solver)
            m = copy(model)
            set_optimizer(m, solver)
            set_time_limit_sec(m, 3600)
            #set_silent(m)
            solveBP_LPmodel!(m)
            global optval = objective_value(m)
        end
        @info "Optimal value = $optval"

        # BP_MAC with HOC
        @info "Elapsed CPU time for solving with BP_MAP with HOC"
        @time xMAP, zMAP, it, inner_it, status =
            solveBP_MAP(prob, itmax = itmax, ε = eps(1.0), ε_MAP = tol, usehoc = true; verbose = true)
        @info "BP-MAP with HOC status is $status with  $(it) iterations and $(inner_it) inner iterations"
        @info "Feasibility = $(relerror(prob.A*xMAP, prob.b))"
        @info "Objective = $(norm(xMAP, 1))"
        @info "Elapsed CPU time for BP_MAP with HOC"
        # @btime solveBP_MAP($prob, itmax = $itmax, ε = $tol, ε_MAP = $tol, usehoc = true)

        # ISAL1
        @info "Elapsed CPU time for solving with ISAL1 Solver"
        @time xISAL, time_ISAL, it_ISAL, status_ISAL = solveBP_ISAL1(prob)
        @info "xISAL - sol = $(relerror(xISAL, prob.sol))"
        @info "Elapsed CPU ISAL1:\n  $time_ISAL s"
        println("="^10)

    end
end
