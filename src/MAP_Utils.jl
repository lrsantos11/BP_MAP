import ProximalOperators: prox

export ProjectIndicator, MAP

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
function MAP_iteration!(xMAP::AbstractArray, ProjA::AbstractArray, ProjectB::Function)
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
function MAP(
    x₀::AbstractArray,
    ProjectA::Function,
    ProjectB::Function;
    ε_MAP::Float64 = 1e-6,
    itmax_MAP::Int = 100,
    xSol::AbstractArray = [],
    verbose::Bool = false,
    timeout = 3600,
)
    start = time()

    solution_given = !isempty(xSol)
    iter = 0
    xMAP = x₀
    ProjA = ProjectA(xMAP)
    dist_AB = norm(ProjA - xMAP, 2)
    solved = false
    tired = false
    tolMAP = 1.0
    status = :IterMax
    while !(solved || tired)
        MAP_iteration!(xMAP, ProjA, ProjectB)
        ProjA = ProjectA(xMAP)
        iter += 1
        # Check for infeasibility
        dist_AB_Old = dist_AB
        dist_AB = norm(ProjA - xMAP, 2)
        infeasible = iter > 1 && isapprox(dist_AB, dist_AB_Old, rtol = ε_MAP, atol = 0.0)
        if infeasible
            status = :Infeasible
            break
        end
        if solution_given
            solved =
                isapprox(xMAP, xSol; rtol = ε_MAP, atol = ε_MAP, norm = x -> norm(x, Inf))
        else
            solved =
                isapprox(xMAP, ProjA; rtol = ε_MAP, atol = ε_MAP, norm = x -> norm(x, Inf))
        end
        solved && (status = :Solved)
        tired = (iter >= itmax_MAP || time() - start > timeout)
    end
    verbose && @info "MAP: Status $status"
    return xMAP, iter, status
end
