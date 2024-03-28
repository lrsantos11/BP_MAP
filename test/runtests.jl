using DrWatson, Test
@quickactivate "BP_MAP"

# Here you include files using `srcdir`
include(srcdir("BP_MAP.jl"))



@testset "Example B1 [HL2014]" begin 
##
    @info "Example B.1 of Hesse and Luke 2014"
    itmax = 5000
    A = [
        1.0 1.0 1.0 1.0 1.0 1.0 1.0 1
        1.0 1.0 1.0 1.0 -1.0 -1.0 -1.0 -1
        1.0 1.0 -1.0 -1.0 1.0 1.0 -1.0 -1
        1.0 -1.0 1.0 -1.0 1.0 -1.0 1.0 -1
        1.0 1.0 -1.0 -1.0 -1.0 -1.0 1.0 1
        1.0 -1.0 -1.0 1.0 1.0 -1.0 -1.0 1
        1.0 -1.0 1.0 -1.0 -1.0 1.0 -1.0 1
    ] ./ sqrt(8)
    m, n = size(A)
    xsol = [10.0, 0, 0, 0, 0, 0, 0, 0]
    b = A * xsol
    Affine = IndAffine(A, b)
    tol = 1e-6
    xMAP, it, inner_it, status = BP_MAP(Affine, itmax=itmax, ε=tol, BP_solution = xsol, verbose = true)
    @test norm(xMAP - xsol, 2) < tol
    @test status == :Solved
end
