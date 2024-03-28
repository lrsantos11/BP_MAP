# author: LRS
include(srcdir("MAP_utils.jl"))

"""
Using the Method of Alternating Projections for the Basis Pursuit problem
min ||x||₁ 
s.t. Ax = b 

BP_MAP(Affine; itmax=100, EPSVAL=1e-6, verbose=true, x₀=Float64[], kwargs...) → xMAP, it, inner_it, status
"""
function BP_MAP(Affine;
    itmax::Int=100,
    ε::Number=1e-6,
    verbose::Bool=false,
    x₀::Vector{Float64}=Float64[],
    BP_solution::Vector{Float64}=Float64[],
    kwargs...)
    m, n = size(Affine.A)
    ProjAffine(x) = ProjectIndicator(Affine, x)
    if isempty(x₀)
        x₀ = zeros(n)
    end
    BP_solution_given = !isempty(BP_solution)
    xMAP = ProjAffine(x₀)
    distance = norm(xMAP, 2)
    radius = 0.0
    solved = false
    tired = false
    it = 0
    inner_it_total = 0
    status = :Tired
    tolBP = 1.0
    while !(solved || tired)
        radius += distance
        BallL1 = IndBallL1(radius)
        Proj_BallL1(x) = ProjectIndicator(BallL1, x)
        zMAP, inner_it, _ = MAP(xMAP, ProjAffine, Proj_BallL1, itmax=itmax, kwargs...)
        inner_it_total += inner_it
        xMAP = ProjAffine(zMAP)
        distance = norm(xMAP - zMAP, 2)
        BP_solution_given ? tolBP = norm(xMAP - BP_solution, 2) : tolBP = distance
        it += 1
        solved = tolBP < ε
        if solved
            verbose && @info "solved"
            verbose && @info "it = $it"
            verbose && @info "inner_it_total = $inner_it_total"
            status = :Solved
        end
        tired = it >= itmax
    end
    return xMAP, it, inner_it_total, status
end

