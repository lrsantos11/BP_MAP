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
    xsol = [10.0, 0, 0, 0, 0, 0, 0, 0]
    probHL14 = BPProblem(xsol, A)
    Affine = IndAffine(probHL14)
    tol = 1e-4
    xMAP, it, inner_it, status = BP_MAP(Affine, itmax=itmax, ε=tol, ε_MAP = 1e-4, BP_solution = xsol, verbose = true)
    @test norm(xMAP - xsol, 2) < tol
    @test status == :Solved
end

##
"Downloads Tests from from the Lorentz, Pfetsch, and Tillmann collection"

include(scriptsdir("downloadtestsets.jl"))
using Glob
using BenchmarkTools
LPT_testset = glob("*.mat", datadir("exp_raw", "L1_Testset_mat"));


@testset "Ten instances from the LPT collection" begin
    itmax = 1000
    tol = 1e-4
    for inst in LPT_testset[1:10]
        prob = readl1test(inst)
        Affine = IndAffine(prob)
        xMAP, it, inner_it, status = BP_MAP(Affine, itmax = itmax, ε = tol, ε_MAP = tol,  BP_solution = prob.sol, verbose = false);
        @btime BP_MAP($Affine, itmax = $itmax, ε = $tol, ε_MAP = $tol,  BP_solution = $prob.sol, verbose = false)
        @info "Instance $(inst[55:end-4]) status is $status with  $(it) iterations and $(inner_it) inner iterations"
    end
end
