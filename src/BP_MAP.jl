"""
Solves the Basis Pursuit Problem

min_x || x ||_1
s.t.  Ax = b

with different methods including a MAP based solver.

Authors: LRS an PJSS
"""

module BP_MAP

cpu_model = Sys.cpu_info()[1].model
@info cpu_model
if occursin("Intel", cpu_model) || occursin("AMD", cpu_model)
    global islinux = true
    using MKL
    using MKLSparse
    @info "Using MKL and MKLSparse"
else
    global islinux = false
    using AppleAccelerate
    using ThreadedSparseArrays
end

using Printf
using LinearAlgebra
using SparseArrays
using LinearOperators
using QRMumps
using CUDA
using CUDA.CUSPARSE
import ProximalOperators: IndBallL1, IndAffine
#import Krylov: CgneSolver, cgne!, CgSolver, cg!, cg, statistics, MinaresSolver, minares!
using Krylov
using Gurobi, JuMP

export solveBP_MAP, affproxproj, affqrmumpsproj, affkrylovproj, affkktproj, affgurobiproj

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
    prob::AbstractBPP;
    usehoc = false,
    itmax::Int = 1000,
    ε::Number = 1e-6,
    ε_MAP::Number = 1e-6,
    verbose::Bool = false,
    x₀::AbstractVector = [],
    BP_solution::AbstractVector = [],
    timeout = 3600,
    kwargs...,
)
    start = time()
    # Parameter to control support identification
    δ = 1.0e-12

    _, n = size(prob.A)
    Ta = eltype(prob.A)
    ProjAffine = projBPP(prob, verbose)
    if isempty(x₀)
        x₀ = zeros(Ta, n)
    end
    BP_solution_given = !isempty(BP_solution)
    verbose && @printf("%6d: ", 0)
    xMAP = ProjAffine(x₀)
    verbose && println()
    distance = norm(xMAP, 2)
    radius = 0.0
    solved = false
    tired = false
    it = 0
    inner_it_total = 0
    status = :Tired
    tolBP = 1.0
    repsupport = 0
    support = Int[]
    while !(solved || tired)
        verbose && @printf("%6d: ", it + 1)
        radius += distance
        BallL1 = IndBallL1(radius)
        global Proj_BallL1 = x -> ProjectIndicator(BallL1, x)
        zMAP, inner_it, inner_status = MAP(
            xMAP,
            ProjAffine,
            Proj_BallL1,
            itmax_MAP = itmax,
            verbose = false,
            ε_MAP = ε_MAP,
            kwargs...,
        )
        xMAP = ProjAffine(zMAP)
        distance = norm(xMAP - zMAP, 2)
        it += 1
        inner_it_total += inner_it

        # Try to identify the support and apply HOC if reasonable 
        new_support = findall(x -> abs(x) > δ, zMAP)
        if usehoc && (new_support == support)
            repsupport += 1
            if repsupport == 1
                verbose && print("H")
                xhoc, hocstatus = heuristic_optimality_check(zMAP, prob; δ = δ)
                if hocstatus == :success
                    verbose && (println(); @info "HOC declared success")
                    xMAP = xhoc
                    status = :Solved
                    break
                end
            end
        else
            repsupport = 0
            support = new_support
        end

        if inner_status == :Solved
            verbose && (println(); @info "Inner Solved")
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
            verbose && println(); @info "Solved"
            verbose && @info "it = $it"
            verbose && @info "distance = $distance"
            verbose && @info "inner_it_total = $inner_it_total"
            status = :Solved
        else
            verbose && println()
        end
        tired = (it >= itmax || time() - start > timeout)
    end
    return xMAP, Proj_BallL1(xMAP), it, inner_it_total, status
end

"Factory for the projection function onto Ax = b using ProximalOperators"
function affproxproj(prob::AbstractBPP, verbose = false)
    affine = IndAffine(prob)
    function proj(x)
        p = ProjectIndicator(affine, x)
        verbose && print(".")
        return p
    end
    return proj
end

"Factory for the projection function onto Ax = b using QRMumps"
function affqrmumpsproj(prob::SparseCSCBPP, verbose = false)
    m, n = size(prob)
    if "OMP_NUM_THREADS" in keys(ENV)
        n_threads = parse(Int, ENV["OMP_NUM_THREADS"])
    else
        n_threads = Threads.nthreads()
    end
    qrm_init(n_threads)
    spmat = qrm_spmat_init(prob.accelA)
    spfct = qrm_spfct_init(spmat)
    qrm_analyse!(spmat, spfct, transp='t')
    qrm_set(spfct, "qrm_keeph", 0)
    qrm_factorize!(spmat, spfct, transp='t')
    λ = similar(prob.accelb)
    b = similar(prob.accelb)
    z = zeros(eltype(prob.A), n) 
    function proj(x)
        # Solve the minimal norm problem
        mul!(b, prob.accelAt', x)
        b .-= prob.accelb
        qrm_solve!(spfct, b, z, transp='t')
        qrm_solve!(spfct, z, λ, transp='n')
        # Apply one step of iterative refiment
        res = b - prob.accelAt'*(prob.accelA'*λ)
        cr1 = qrm_solve(spfct, res, transp='t')
        cr = qrm_solve(spfct, cr1, transp='n')
        λ += cr
        if verbose
            print(".")
        end
        return x - prob.accelA'*λ
    end
    return proj
end

"Factory for the projection onto Ax = b based on Krylov methods"
function affkrylovproj(prob::AbstractSparseBPP, verbose = false)
    OpA = Afactory(prob)
    m, n = size(prob)
    solver = CrmrSolver(m, n, typeof(prob.accelb))
    pre_proj = solver.x
    function proj(x)
        b = prob.accelb - OpA * convert(typeof(prob.accelb), x)
        crmr!(solver, OpA, b, itmax = 10*(m + n))
        if verbose
            if solver.stats.solved
                print(".")
            else
                print("F")
            end
        end
        return Vector(pre_proj) + x
    end
    return proj
end

"Factory for the projection onto Ax = b based on KKT"
function affkktproj(prob::AbstractSparseBPP, verbose = false)
    # Create the linear operator for the system of equations
    m, _ = size(prob)
    Op = AAtfactory(prob)
    solver = CgSolver(m, m, typeof(prob.accelb))
    # D = CuSparseMatrixCSR(spdiagm([1 / norm(prob.A[i, :]) for i in 1:m]))
    λ = solver.x
    function proj(x)
        b = prob.accelAt' * convert(typeof(prob.accelb), x) - prob.accelb
        cg!(solver, Op, b)
        if verbose
            if solver.stats.solved
                print(".")
            else
                print("F")
            end
        end
        return x - Vector(prob.accelA' * λ)
    end
    return proj
end

"Factory for the projection onto Ax = b using Gurobi"
function affgurobiproj(prob::AbstractSparseBPP, verbose = false)
    _, n = size(prob)

    # Model projection onto Ax = b as a quadratic problem
    model = Model(Gurobi.Optimizer)
    set_silent(model)
    @variable(model, p[1:n])
    # Initial objetive is to projetction zero
    @objective(model, Min, 1 / 2 * sum(p .^ 2))
    @constraint(model, prob.A * p == prob.b)

    function proj(x)
        set_objective_coefficient(model, p, -x)
        optimize!(model)
        if is_solved_and_feasible(model)
            verbose && print(".")
            return value.(p)
        else
            verbose && print("F")
            return zeros(n)
        end
    end
    return proj
end

"Preferred projection for problems that use desnse matrices"
projBPP(prob::DenseBPP, verbose = false) = affproxproj(prob, verbose)

"Preferred projection for problems that use sparse matrices"
projBPP(prob::AbstractSparseBPP, verbose = false) = affkktproj(prob, verbose)

end
