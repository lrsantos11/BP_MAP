using LinearAlgebra, SparseArrays
using ProximalOperators



"""
    ProjectIndicator
"""
####################################
"""
    proj = ProjectIndicator(indicator,x)
    Projection using Indicator Function from `ProximalOperators.jl`
    """
function ProjectIndicator(indicator_func, x)
    proj, _ = prox(indicator_func, x)
    return proj
end

"""
    MAPiteration!(xMAP,ProjA,ProjectB)

Computes a MAP iteration
"""
function MAP_iteration!(xMAP::AbstractArray,
                        ProjA::AbstractArray,
                        ProjectB::Function)
    xMAP .= ProjectB(ProjA)
    return nothing
end

"""
    Method of Alternating Projections
    MAP(x₀, ProjectA, ProjectB; itmax=100) → xApprox, iter_total
"""


"""
    MAP(x₀,ProjectA, ProjectB)

    Method of Alternating Projections
"""
function MAP(x₀::Vector, ProjectA::Function, ProjectB::Function;
    ε_MAP::Float64=1e-6,
    itmax_MAP::Int=100,
    xSol::Vector=[])
    solution_given = !isempty(xSol)
    iter = 0
    xMAP = x₀
    ProjA = ProjectA(xMAP)
    solved = false
    tired  =  false
    tolMAP = 1.0
    while !(solved || tired)
        MAP_iteration!(xMAP, ProjA, ProjectB)    
        ProjA = ProjectA(xMAP)
        solution_given ? tolMAP = norm(xMAP - xSol, Inf) : tolMAP = norm(ProjA - xMAP, Inf)
        solved = tolMAP < ε_MAP
        iter += 1
        tired = iter >= itmax_MAP
    end
    return xMAP, iter, tolMAP
end