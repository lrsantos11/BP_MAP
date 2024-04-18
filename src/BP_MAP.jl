"""
Solves the Basis Pursuit Problem

min_x || x ||_1
s.t.  Ax = b

with different methods including a MAP based solver.

Authors: LRS an PJSS
"""

module BP_MAP

using LinearAlgebra
using LinearOperators
import ProximalOperators: IndBallL1
import Krylov: CgneSolver, cgne!, CgSolver, cg!

export solveBP_MAP, affproxproj, affkrylovproj, affkktproj

include("BP_Utils.jl")
include("MAP_Utils.jl")

"""
Using the Method of Alternating Projections for the Basis Pursuit problem
min ||x||₁ 
s.t. Ax = b 

solveBP_MAP(
    prob; affproj_factory=proxproj, usehoc=false, itmax=100, ε=1e-6, 
    verbose=false, x₀=Float64[], kwargs...) → xMAP, it, inner_it, status
"""
function solveBP_MAP(
    prob::BPProblem;
    affproj_factory = smartproj,
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

    m, n = size(prob.A)
    Ta = eltype(prob.A)
    ProjAffine = affproj_factory(prob)
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
                xhoc, hocstatus = heuristic_optimality_check(zMAP, prob ; δ = δ)
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
                xhoc, hocstatus = heuristic_optimality_check(zMAP, prob; δ = δ)
                if hocstatus == :success
                    verbose && @info "HOC declared success"
                    xMAP = xhoc
                else
                    verbose && @info "HOC failed"
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


"Factory for the projection function onto Ax = b using ProximalOperators"
function affproxproj(prob::BPProblem)
        affine = IndAffine(prob)
        return (x -> ProjectIndicator(affine, x))
end

"Factory for the projection onto AX = b based on Krylov methods"
function affkrylovproj(prob::BPProblem)
    OpA = LinearOperator(prob.A)
    cgne_solver = CgneSolver(OpA, prob.b)
    pre_proj = cgne_solver.x
    function proj(x)
        cgne!(cgne_solver, OpA, prob.b - OpA*x)
        return pre_proj + x
    end 
    return proj
end

"Factory for the projection onto AX = b based on KKT"
function affkktproj(prob::BPProblem)
    T = eltype(prob)
    # Mount the linear operator for the system of equations
    m, _ = size(prob)
    Op1 = LinearOperator(prob.A)
    # Op2 = LinearOperator(prob.A)
    Op = Op1 * Op1'
    cg_solver = CgSolver(m, m, Vector{T})
    λ = cg_solver.x
    previousλ = fill(NaN, m)
    function proj(x)
        b = prob.A*x - prob.b
        if isnan(previousλ[1])
            λ₀ = zeros(m)
        else
            λ₀ = previousλ
        end
        cg!(cg_solver, Op, b, λ₀)
        previousλ .= λ
        return x - prob.A'*λ
    end 
    return proj
end

"Factory for the projection onto Ax = b that choses the strategy based on whether prob.A 
is sparse or not."
function smartproj(prob::BPProblem)
    if issparse(prob.A)
        return affkktproj(prob)
    else
        return affproxproj(prob)
    end
end

end
