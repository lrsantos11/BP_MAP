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
    tol = 1e-6
    xMAP, it, inner_it, status = BP_MAP(Affine, itmax=itmax, ε=tol, BP_solution = xsol, verbose = true)
    @test norm(xMAP - xsol, 2) < tol
    @test status == :Solved
end

##
"Downloads Tests from from the Lorentz, Pfetsch, and Tillmann collection"

include(scriptsdir("downloadtestsets.jl"))
using Glob
using BenchmarkTools
LPT_testset = glob("*.mat", datadir("exp_raw", "L1_Testset_mat"));


@testset "Tests from the LPT collection" begin
    LPT_testset = glob("*.mat", datadir("exp_raw", "L1_Testset_mat"))
    itmax = 1000
    tol = 1e-6
    for inst in LPT_testset[1:10]
        prob = readl1test(inst)
        Affine = IndAffine(prob)
        @btime xMAP, it, inner_it, status = BP_MAP(Affine, itmax = itmax,  BP_solution = prob.sol, verbose = false);
        @info "Test $(inst) status is $status with  $(it) iterations and $(inner_it) inner iterations"
    end
end

itmax = 1100
prob = readl1test(LPT_testset[3])
Affine = IndAffine(prob)
xMAP, it, inner_it, status = BP_MAP(Affine, itmax = itmax,  BP_solution = prob.sol, verbose = true)
@info "Test $(inst) status is $status with  $(it) iterations and $(inner_it) inner iterations"

sol = prob.sol
sol[findall(x-> x > 1e-6, sol)] 