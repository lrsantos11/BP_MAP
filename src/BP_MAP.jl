"""
Solves the Basis Pursuit Problem

min_x || x ||_1
s.t.  Ax = b

with different methods including a MAP based solver.

Authors: LRS an PJSS
"""

module BP_MAP

using LinearAlgebra
using JuMP
using HiGHS
import ProximalOperators: IndBallL1

export solveBP_MAP, solveBP_LP, buildBP_LPModel, solveBP_LPmodel!

include("BP_Utils.jl")
include("MAP_Utils.jl")

"""
Using the Method of Alternating Projections for the Basis Pursuit problem
min ||x||₁ 
s.t. Ax = b 

solveBP_MAP(Affine; itmax=100, ε=1e-6, verbose=false, x₀=Float64[], kwargs...) → xMAP, it, inner_it, status
"""
function solveBP_MAP(
    Affine;
    usehoc = false,
    itmax::Int = 1000,
    ε::Number = 1e-6,
    ε_MAP::Number = 1e-6,
    verbose::Bool = false,
    x₀::AbstractVector = [],
    BP_solution::AbstractVector = [],
    kwargs...,
)
    # Parameter to control support identification
    δ = 1.0e-12

    m, n = size(Affine.A)
    Ta = eltype(Affine.A)
    ProjAffine(x) = ProjectIndicator(Affine, x)
    if isempty(x₀)
        x₀ = zeros(Ta, n)
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
    repsupport = 0
    support = findall(x -> abs(x) > δ, xMAP)
    while !(solved || tired)
        radius += distance
        BallL1 = IndBallL1(radius)
        global Proj_BallL1 = x -> ProjectIndicator(BallL1, x)
        zMAP, inner_it, inner_status = MAP(
            xMAP,
            ProjAffine,
            Proj_BallL1,
            itmax_MAP = itmax,
            verbose = verbose,
            ε_MAP = ε_MAP,
            kwargs...,
        )
        xMAP = ProjAffine(zMAP)
        it += 1
        inner_it_total += inner_it

        # Try to identify the support and apply HOC if reasonable 
        new_support = findall(x -> abs(x) > δ, zMAP)
        if usehoc && (new_support == support) 
            repsupport += 1
            if repsupport == 2
                verbose && @info "Applying HOC"
                xhoc, hocstatus = heuristic_optimality_check(zMAP, Affine; δ=δ)
                if hocstatus == :success 
                    verbose && @info "HOC declared success"
                    xMAP = xhoc
                    status = :Solved
                    break
                end
            end
        else
            repsupport = 0
            support = new_support
        end

        distance = norm(xMAP - zMAP, 2)
        if inner_status == :Solved
            verbose && @info "Inner Solved"
            verbose && @info "Distance = $distance"
            verbose && @info "Applying HOC"
            if usehoc
                xhoc, hocstatus = heuristic_optimality_check(zMAP, Affine; δ=δ)
                if hocstatus == :success 
                    verbose && @info "HOC declared success"
                    xMAP = xhoc
                end
            end
            status = :Solved
            break
        end
        tolBP = BP_solution_given ? norm(xMAP - BP_solution, 2) : distance
        solved = ((tolBP < ε) || (distance < ε))
        if solved
            verbose && @info "Solved"
            verbose && @info "it = $it"
            verbose && @info "distance = $distance"
            verbose && @info "inner_it_total = $inner_it_total"
            status = :Solved
        end
        tired = it >= itmax
    end
    return xMAP, Proj_BallL1(xMAP), it, inner_it_total, status
end

"Solve a BP problem using MAP"
function solveBP_MAP(prob::BPProblem; kwags...)
    affine = IndAffine(prob)
    x, _, _, status = solveBP_MAP(affine; kwags...)
    @assert status == :Solved
    return x
end

"Solve a BP problem using Linear Programming"
function solveBP_LP(prob::BPProblem; Solver = HiGHS)
    model = buildBP_LPModel(prob, Solver = Solver)
    return solveBP_LPmodel!(model)
end

"Build a LP model that describes the Basis Pursuit problem"
function buildBP_LPModel(prob::BPProblem; solver = HiGHS.Optimizer)
    # Create an LP model that represents the basis pursuit problem
    model = Model(solver)
    set_silent(model)

    m, n = size(prob.A)
    @variable(model, xplus[1:n] >= 0)
    @variable(model, xminus[1:n] >= 0)
    @objective(model, Min, sum(xplus) + sum(xminus))
    @constraint(model, prob.A * xplus - prob.A * xminus == prob.b)

    # Solve the model and return the solution
    return model
end

"Solve an LP model describing BP"
function solveBP_LPmodel!(model)
    optimize!(model)
    @assert termination_status(model) == OPTIMAL
    return value.(model[:xplus]) - value.(model[:xminus])
end

end
