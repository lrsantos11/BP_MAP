using DrWatson, Test
@quickactivate "BP_MAP"

if occursin("Intel", Sys.cpu_info()[1].model)
    using MKL
end


# Here you include files using `srcdir`
include(srcdir("BP.jl"))
include(srcdir("BP_MAP.jl"))

using .BP

##
@testset "Example B1 [HL2014]" begin 
    @info "Example B.1 of Hesse and Luke 2014"
    itmax = 50
    A = [
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
    xMAP, it, inner_it, status = BP_MAP(Affine, itmax=itmax, ε = tol, ε_MAP = tol, BP_solution = sol, verbose = true)
    @info "BP-MAP status is $status with  $(it) iterations and $(inner_it) inner iterations"
    @info "xMAP - sol = $(norm(xMAP - sol, 2))"
    @test status == :Solved
    @info "Using HOC"
    xSol, _ = heuristic_optimality_check(xMAP, Affine, δ = tol)
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
    sol = [-10/3, 10/3, -10/3]
    Affine = IndAffine(prob_B2_HL14)
    tol = 1e-3
    xMAP, it, inner_it, status = BP_MAP(
        Affine,
        itmax = itmax,
        ε = tol,
        ε_MAP = tol,
        verbose = false,
    )
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
        prob = readl1test(instance)
        prob_name = instance[55:end-4]
        Affine = IndAffine(prob)
        @info "Problem $(prob_name) - size: $(size(Affine.A))"
        xMAP, it, inner_it, status = BP_MAP(Affine, itmax = itmax, ε = tol, ε_MAP = tol,  verbose = false);
        @info "xMAP - sol = $(norm(xMAP - prob.sol, 2))"
        @info "BP-MAP status is $status with  $(it) iterations and $(inner_it) inner iterations"
        @info "Applying HOC"
        xSol_HOC, status_hoc = heuristic_optimality_check(xMAP, Affine, δ = tol)
        @info "xSol_HOC - sol = $(norm(xSol_HOC - prob.sol, 2))"
        @test norm(xSol_HOC - prob.sol) < 1e-8
        @info "Elapsed CPU time for BP_MAP + HOC"
        @btime begin
            BP_MAP(
            $Affine,
            itmax = $itmax,
            ε = $tol,
            ε_MAP = $tol,
            verbose = false,
        )
            heuristic_optimality_check($xMAP, $Affine, δ = $tol)
        end
        @info "Elapsed CPU time for solving with LP Solver"
        # Use Gurobi if avaliable.
        # @btime solvewithLP($prob, Solver = Gurobi)
        @btime solvewithLP($prob)

      println("="^10)
    end
end
