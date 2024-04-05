using DrWatson, Test
@quickactivate :BP_MAP

cpu_model = Sys.cpu_info()[1].model
if occursin("Intel", cpu_model) || occursin("AMD", cpu_model)
    using MKL
end

using LinearAlgebra

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
    xMAP, zMAP, it, inner_it, status = solveBP_MAP(
        Affine,
        itmax = itmax,
        ε = tol,
        ε_MAP = tol,
        BP_solution = sol,
        verbose = false, 
    )
    @info "BP-MAP status is $status with  $(it) iterations and $(inner_it) inner iterations"
    @info "xMAP - sol = $(norm(xMAP - sol, 2))"
    @test status == :Solved
    @info "Using HOC"
    xSol, _ = heuristic_optimality_check(zMAP, Affine, δ = tol^4)
    @info "Solucao:  $xSol"
    @info "xSol - sol = $(norm(xSol - sol, 2))"
    @test xSol ≈ sol
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
        solveBP_MAP(Affine, itmax = itmax, ε = tol, ε_MAP = tol, verbose = false)
    @info "BP-MAP status is $status with  $(it) iterations and $(inner_it) inner iterations"
    @info "Solution not unique. Not calling HOC here"
    @info "xMAP - sol = $(norm(xMAP - sol, 2))"
    @test status == :Solved
    @test xMAP ≈ sol

end

##
"Downloads Tests from from the Lorentz, Pfetsch, and Tillmann collection"

include(scriptsdir("downloadtestsets.jl"))
using Glob
using BenchmarkTools
using Gurobi
LPT_testset = glob("*.mat", datadir("exp_raw", "L1_Testset_mat"));

@testset "Instances from the LPT collection" begin
    itmax = 2000
    tol = 1e-3
    for instance in LPT_testset[1:10]
        prob_name = basename(instance)
        prob = readl1test(prob_name)
        affine = IndAffine(prob)
        @info "Problem $(prob_name) - size: $(size(prob.A))"
        xMAP, zMAP, it, inner_it, status =
            solveBP_MAP(affine, itmax = itmax, ε = tol, ε_MAP = tol, verbose = false)
        @info "xMAP - sol = $(relerror(xMAP, prob.sol))"
        @info "BP-MAP status is $status with  $(it) iterations and $(inner_it) inner iterations"
        @info "Applying HOC"
        xSol_HOC, status_hoc = heuristic_optimality_check(zMAP, affine, δ = tol^4)
        @info "xSol_HOC - sol = $(norm(xSol_HOC - prob.sol, 2) / norm(prob.sol, 2))"
        @test relerror(xSol_HOC, prob.sol) < 1e-8
        @info "Elapsed CPU time for BP_MAP + HOC"
        @btime begin
            affine = IndAffine($prob)
            solveBP_MAP(
                IndAffine($prob),
                itmax = $itmax,
                ε = $tol,
                ε_MAP = $tol,
                verbose = false,
            )
            heuristic_optimality_check($xMAP, affine, δ = $tol)
        end
        @info "Elapsed CPU time for solving with LP Solver"
        # Use Gurobi if avaliable.
        # @btime solveBP_LP($prob, Solver = Gurobi)
        @btime solveBP_LP($prob)

        println("="^10)
    end
end
