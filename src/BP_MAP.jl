# author: LRS
include(scr("MAP_utils.jl"))

"""
Using the Method of Alternating Projections for the Basis Pursuit problem
min ||x||₁ 
s.t. Ax = b 

BP_MAP(Affine; itmax=100, EPSVAL=1e-6, verbose=true, x₀=Float64[], kwargs...) → xMAP, it, inner_it, status
"""
function BP_MAP(Affine;
    itmax::Int=100,
    EPSVAL::Number=1e-6,
    verbose::Bool=true,
    x₀::Vector{Float64}=Float64[],
    kwargs...)
    m, n = size(Affine.A)
    ProjAffine(x) = ProjectIndicator(Affine, x)
    if isempty(x₀)
        x₀ = zeros(n)
    end
    xMAP = ProjAffine(x₀)
    distance = norm(xMAP, 2)
    radius = 0.0
    solved = false
    tired = false
    it = 0
    inner_it = 0
    status = :Tired
    verbose && @info "m = $m, n = $n"
    while !(solved || tired)
        radius += distance
        BallL1 = IndBallL1(radius)
        Proj_BallL1(x) = ProjectIndicator(BallL1, x)
        resultMAP = MAP(xMAP, ProjAffine, Proj_BallL1, itmax=itmax, gap_distance=false, kwargs...)
        inner_it += resultMAP.iter_total
        zMAP = resultMAP.xApprox
        xMAP = ProjAffine(zMAP)
        distance = norm(xMAP - zMAP, 2)
        it += 1
        solved = distance < EPSVAL
        if solved
            verbose && @info "solved"
            verbose && @info "it = $it"
            verbose && @info "inner_it = $inner_it"
            status = :Solved
        end
        tired = it >= itmax
    end
    return xMAP, it, inner_it, status
end

