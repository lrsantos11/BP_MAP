# Utilities for Basis Pursuit problems.
cpu_model = Sys.cpu_info()[1].model
@info cpu_model
if occursin("Intel", cpu_model) || occursin("AMD", cpu_model)
    global islinux = true
    using MKLSparse
    @info "Using MKL and MKLSparse"
else
    global islinux = false
    using AppleAccelerate
    using ThreadedSparseArrays
end

using DrWatson
using LinearAlgebra
using MAT
using SparseArrays
using CUDA
using CUDA.CUSPARSE
import ProximalOperators: IndAffine
import Base: size, eltype
import Krylov: cgls, lslq

cpu_model = Sys.cpu_info()[1].model

export AccelerationTarget, noaccel, CUDAaccel
export MatType, densemat, sparsemat, automat
export AbstractBPP, BPProblem, IndAffine, readl1test, heuristic_optimality_check, size, eltype

"""
Valid acceleration targets
"""
@enum AccelerationTarget noaccel CUDAaccel

"""
A Basis Pursuit problem data: ``\\min_x \\| x \\|_1`` s.t. ``Ax = b``

# Fields
- A: constraint Matrix
- b: constraint right hand side
- sol: a solution, if known, or a vector of NaNs
- optval: optimal value, if known, or NaN
"""
abstract type AbstractBPP end

abstract type AbstractDirBPP <: AbstractBPP end

abstract type AbstractItBPP <: AbstractBPP end

"Abstract sparse matrix where matrix-vector multiply is efficient with the transpose"
abstract type AbstractItCSCBPP <: AbstractItBPP end

"Abstract sparse matrix where direct matrix-vector multiply is efficient"
abstract type AbstractItCSRBPP <: AbstractItBPP end

"BPProblem with dense matrices"
struct DenseBPP{T} <: AbstractDirBPP
    A::Matrix{T}
    b::Vector{T}
    sol::Vector{T}
    optval::T
    accelA::Matrix{T}
    accelAt::Matrix{T}
    accelb::Vector{T}
end

"BPProblem with sparse matrices in CPU"
struct SparseCSCBPP{T} <: AbstractItCSCBPP
    A::SparseMatrixCSC{T}
    b::Vector{T}
    sol::Vector{T}
    optval::T
    accelA::SparseMatrixCSC{T}
    accelAt::SparseMatrixCSC{T}
    accelb::Vector{T}
end

"BPProblem with sparse matrices in NVidia GPU"
struct CuSparseBPP{T} <: AbstractItCSRBPP
    A::SparseMatrixCSC{T}
    b::Vector{T}
    sol::Vector{T}
    optval::T
    accelA::CuSparseMatrixCSR{T}
    accelAt::CuSparseMatrixCSR{T}
    accelb::CuVector{T}
end

# Implement simple functions for BPProblems
size(prob::AbstractBPP) = size(prob.A)
eltype(prob::AbstractBPP) = eltype(prob.A)

"""
    BPProblem(A, b, sol, optval)

Factory function to create an AbstractBPProblem of the right type,
checking dimensions and enforcing a single eltype.
"""
function BPProblem(
    A::AbstractMatrix{T},
    b::Vector{T},
    sol::Vector{T},
    optval::T;
    acceltype::AccelerationTarget = noaccel,
) where {T<:AbstractFloat}
    # Assert that dimensions are sane
    m, n = size(A)
    @assert length(b) == m
    @assert length(sol) == n

    # Check whether solution is feasible and optval comes from it
    small = sqrt(eps(T))
    if !isnan(sol[1])
        @assert norm(A * sol - b) < small
    end
    if !isnan(sol[1]) && !isnan(optval)
        @assert abs(optval - norm(sol, 1)) < small
    end

    if issparse(A)
        if acceltype == noaccel 
            return SparseCSCBPP(A, b, sol, optval, A, convert(SparseMatrixCSC, A'), b)
        elseif acceltype == CUDAaccel
            return CuSparseBPP(
                A,
                b,
                sol,
                optval,
                CuSparseMatrixCSR(A),
                CuSparseMatrixCSR(A'),
                CuVector(b),
            )
        else
            throw(ErrorException("Unknown acceleration target"))
        end
    else
        return DenseBPP(A, b, sol, optval, A, Matrix(A'), b)
    end
end

"Construct a BPproblem without a solution or optimal value"
function BPProblem(A, b; acceltype::AccelerationTarget = noaccel)
    T = eltype(A)
    Tnan = zero(T) / zero(T)

    m, n = size(A)
    sol = fill(Tnan, n)
    optval = Tnan
    return BPProblem(A, b, sol, optval; acceltype = acceltype)
end

"Construct a BPproblem from a solution and a matrix"
function BPProblem(
    sol::AbstractVector{T},
    A::AbstractMatrix{T};
    acceltype::AccelerationTarget = noaccel,
) where {T<:AbstractFloat}
    b = A * sol
    optval = norm(sol, 1)
    return BPProblem(A, b, sol, optval; acceltype = acceltype)
end

"Construct a BPProblem computing the optimal value from the given solution"
BPProblem(A, b, sol::AbstractVector; acceltype::AccelerationTarget = noaccel) =
    BPProblem(A, b, sol, norm(sol, 1); acceltype = acceltype)

"Construct a BPProblem giving the optimal value but no solution"
function BPProblem(A, b, optval::AbstractFloat; acceltype = acceltype)
    Tnan = zero(optval) / zero(optval)
    m, n = size(A)
    sol = fill(Tnan, n)
    return BPProblem(A, b, sol, optval; acceltype = acceltype)
end

"Builds a LinearOperator to represent AA'"
function AAtfactory(p::AbstractItCSCBPP)
    T = eltype(p.accelA)
    m, n = size(p)
    ytemp = typeof(p.accelb)(undef, n)
    function AAt!(y, v)
        mul!(ytemp, p.accelA', v)
        mul!(y, p.accelAt', ytemp)
    end
    return LinearOperator(T, m, m, true, true, AAt!, nothing, nothing)
end

"Builds a LinearOperator to represent AA'"
function AAtfactory(p::AbstractItCSRBPP)
    T = eltype(p.accelA)
    m, n = size(p)
    ytemp = typeof(p.accelb)(undef, n)
    function AAt!(y, v)
        mul!(ytemp, p.accelAt, v)
        mul!(y, p.accelA, ytemp)
    end
    return LinearOperator(T, m, m, true, true, AAt!, nothing, nothing)
end

"Construct a IndAffine from ProximalOperators.jl using BPP that should use a direct solver"
function IndAffine(prob::AbstractDirBPP)
    return IndAffine(prob.accelA, prob.accelb)
end

"Represents possible matrices types to store"
@enum MatType densemat sparsemat automat

"Read a test."
function readl1test(
    filename;
    rhs = 1,
    mattype::MatType = automat,
    acceltype::AccelerationTarget = noaccel,
)
    # Verify if the test is available
    dir = datadir("exp_raw")
    filename = joinpath(dir, filename)
    if !isfile(filename)
        @error "Test file does not exist"
    end

    # Read the actual data
    data = matread(filename)
    if mattype == densemat
        A = Matrix(data["A"])
    elseif mattype == sparsemat
        A = sparse(data["A"])
    else
        A = data["A"]
    end

    # The solution is represented as a one column matrix. Get the respective vector instead.
    if haskey(data, "x")
        return BPProblem(A, Vector(data["b"][:, rhs]), Vector(data["x"][:]); acceltype = acceltype)
    else
        return BPProblem(A, Vector(data["b"][:, rhs]); acceltype = acceltype)
    end
end

"Get a view of the columns of A indexed by S that is suited to solve least square systems using lsHOC"
function select(prob::AbstractDirBPP, S)
    Aₛ = @view prob.A[:, S]
    return qr(Aₛ)
end

"Get a view of the columns of A indexed by S that is suited to solve least square systems using lsHOC"
function select(prob::SparseCSCBPP, S)
    # Convert A[:, S] and its tranpose to the type in prob
    Aₛ = prob.A[:, S]
    M = Aₛ
    Mt = SparseMatrixCSC(Aₛ') 

    # Create a LinearOperator to represent this matrix
    T = eltype(M)
    m, n = size(M)
    mulM!(y, x) = mul!(y, Mt', x)
    mulMt!(y, x) = mul!(y, M', x)
    return LinearOperator(T, m, n, false, false, mulM!, mulMt!, nothing)
end

"Get a view of the columns of A indexed by S that is suited to solve least square systems using lsHOC"
function select(prob::CuSparseBPP, S)
    # Convert A[:, S] and its tranpose to the type in prob
    Aₛ = prob.A[:, S]
    M = CuSparseMatrixCSR(Aₛ)
    Mt = CuSparseMatrixCSR(sparse(Aₛ'))

    # Create a LinearOperator to represent this matrix
    T = eltype(M)
    m, n = size(M)
    mulM!(y, x) = mul!(y, M, x)
    mulMt!(y, x) = mul!(y, Mt, x)
    return LinearOperator(T, m, n, false, false, mulM!, mulMt!, nothing)
end

"Default solver of least squares for the HOC routine. Fallback to default backslash operator"
function lsHOC(A, b)
    ## TODO: Maybe improve with CG instead of "small" QR (See [Lorenz2015, pg. 4])
    return  A \ b    
end

# "Solve a least squares using an iterative solver"
function lsHOC(A::AbstractLinearOperator, b) 
    return Vector(lslq(A, b)[1])
end

"""
    Heuristic for optimality Check from [Lorenz2014, Alg. 2]

"""
function heuristic_optimality_check(
    xSol,
    prob;
    δ::AbstractFloat = 1e-4,
    tol::AbstractFloat = 1e-6,
)
    xSol = copy(xSol)
    T = eltype(prob)
    m, _ = size(prob)
    vectype = typeof(prob.accelb)
    A = prob.A
    b = prob.b
    S = findall(x -> abs(x) > δ, xSol) # [Lorenz2014, Eq. (1)]
    # Avoid overdetermined system not supported error. See TODO below.
    if length(S) > m
        return xSol, :failure
    end
    Aₛ = select(prob, S)
    xSolₛ = @view xSol[S]
    
    ## TODO: See where the try-catch is needed
    try
        w = lsHOC(Aₛ', vectype(sign.(xSolₛ)))
        if isapprox(norm(A'w, Inf), one(T), atol = tol)
            xSol .= 0.0
            xSolₛ .= lsHOC(Aₛ, prob.accelb)

            norm_xSol_1 = norm(xSol, 1)
            solve_Axb = norm(A * xSol - b, Inf) / max(norm(b, Inf), 1.0) <= tol
            is_solution =
                isapprox((norm_xSol_1 - dot(w, b)) / norm_xSol_1, zero(T), atol = tol)
            if solve_Axb && is_solution
                return sparse(xSol), :success
            end
        end
        return xSol, :failure
    catch
        return xSol, :failure
    end
end