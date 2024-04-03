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

export solveBP_MAP, solveBP_LP

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
    itmax::Int = 1000,
    ε::Number = 1e-6,
    ε_MAP::Number = 1e-6,
    verbose::Bool = false,
    x₀::AbstractVector = [],
    BP_solution::AbstractVector = [],
    kwargs...,
)
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
    while !(solved || tired)
        radius += distance
        BallL1 = IndBallL1(radius)
        Proj_BallL1(x) = ProjectIndicator(BallL1, x)
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
        distance = norm(xMAP - zMAP, 2)
        if inner_status == :Solved
            verbose && @info "Inner Solved"
            verbose && @info "Distance = $distance"
            status = :Solved
            break
        end
        BP_solution_given ? tolBP = norm(xMAP - BP_solution, 2) : tolBP = distance
        solved = ((tolBP < ε) || (distance < ε))
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

"Solve a BP problem using MAP"
function solveBP_MAP(prob::BPProblem; kwags...)
    affine = IndAffine(prob)
    x, _, _, status = solveBP_MAP(affine; kwags...)
    @assert status == :Solved
    return x
end

"Solve a BP problem using Linear Programming"
function solveBP_LP(prob::BPProblem; Solver = HiGHS)
    # Create an LP model that represents the basis pursuit problem
    model = Model(Solver.Optimizer)
    set_silent(model)

    m, n = size(prob.A)
    @variable(model, xplus[1:n] >= 0)
    @variable(model, xminus[1:n] >= 0)
    @objective(model, Min, sum(xplus) + sum(xminus))
    @constraint(model, prob.A * xplus - prob.A * xminus == prob.b)

    # Solve the model and return the solution
    optimize!(model)
    @assert termination_status(model) == OPTIMAL
    return value.(xplus) - value.(xminus)
end

end
