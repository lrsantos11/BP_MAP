# Utilities for Basis Pursuit problems.
using DrWatson
using LinearAlgebra
using SparseArrays
using MAT
import ProximalOperators: IndAffine
import Base: size, eltype

export BPProblem, IndAffine, readl1test, heuristic_optimality_check, size, eltype

"""
A Basis Pursuit problem data: ``\\min_x \\| x \\|_1`` s.t. ``Ax = b``

# Fields
- A: constraint Matrix
- b: constraint right hand side
- sol: a solution, if known, or a vector of NaNs
- optval: optimal value, if known, or NaN
"""
struct BPProblem{T<:AbstractFloat}
    A::AbstractMatrix{T}
    b::AbstractVector{T}
    sol::AbstractVector{T}
    optval::T

    """
        BPProblem(A, b, sol, optval)

    Construct a BPProblem checking dimensions and enforcing a single eltype.
    """
    function BPProblem(
        A::AbstractMatrix{T},
        b::AbstractVector{T},
        sol::AbstractVector{T},
        optval::T,
    ) where {T<:AbstractFloat}

        small = sqrt(eps(T))

        # Assert that dimensions are sane
        m, n = size(A)
        @assert length(b) == m
        @assert length(sol) == n
        if !isnan(sol[1])
            @assert norm(A * sol - b) < small
        end
        if !isnan(sol[1]) && !isnan(optval)
            @assert abs(optval - norm(sol, 1)) < small
        end

        new{T}(A, b, sol, optval)
    end
end

size(prob::BPProblem) = size(prob.A)
eltype(prob::BPProblem) = eltype(prob.A)

"Construct a BPproblem without a solution or optimal value"
function BPProblem(A, b)
    T = eltype(A)
    Tnan = zero(T) / zero(T)

    m, n = size(A)
    sol = fill(Tnan, n)
    optval = Tnan
    return BPProblem(A, b, sol, optval)
end

"Construct a BPproblem from a solution and a matrix"
function BPProblem(sol::AbstractVector{T}, A::AbstractMatrix{T}) where {T<:AbstractFloat}
    b = A * sol
    optval = norm(sol, 1)
    return BPProblem(A, b, sol, optval)
end

"Construct a BPProblem computing the optimal value from the given solution"
BPProblem(A, b, sol::AbstractVector) = BPProblem(A, b, sol, norm(sol, 1))

"Construct a BPProblem giving the optimal value but no solution"
function BPProblem(A, b, optval::AbstractFloat)
    Tnan = zero(optval) / zero(optval)
    m, n = size(A)
    sol = fill(Tnan, n)
    return BPProblem(A, b, sol, optval)
end

"Construct a IndAffine from ProximalOperators.jl using a BPProblem"
function IndAffine(prob::BPProblem)
    return IndAffine(prob.A, prob.b)
end

"Read a test from the Lorentz, Pfetsch, and Tillmann testset"
function readl1test(filename; sparse_matrix::Bool=false)
    # Verify if the test is available
    dir = datadir("exp_raw")
    filename = joinpath(dir, filename)
    if !isfile(filename)
        @error "Test file does not exist"
    end

    # Read the actual data
    data = matread(filename)
    if sparse_matrix
        A = sparse(data["A"])
    else
        A = data["A"]
    end

    # The solution is represented as a one column matrix. Get the respective vector instead.
    if haskey(data, "x")
        return BPProblem(A, Vector(data["b"][:]), data["x"][:])
    else
        return BPProblem(A, Vector(data["b"][:]))
    end
end

"""
    Heuristic for optimality Check from [Lorenz2014, Alg. 2]

"""
function heuristic_optimality_check(xSol, prob; δ::AbstractFloat = 1e-4, tol::AbstractFloat = 1e-12)
    T = eltype(xSol)
    m, n = size(prob)
    b = @views prob.b
    A = @views prob.A
    S = findall(x -> abs(x) > δ, xSol) # [Lorenz2014, Eq. (1)]
    # Avoid overdetermined system not supported error. See TODO below.
    if length(S) > m
        return xSol, :failure
    end
    Aₛ = @views A[:, S]
    xSolₛ = @views xSol[S]
    ## TODO: Improve with CG instead of "small" QR (See [Lorenz2015, pg. 4])
    F = qr(Aₛ)
    opAₛ = LinearOperator(Aₛ)
    ## TODO: See where the try-catch is needed
    try 
        w = F' \ sign.(xSolₛ)
        if isapprox(norm(A'w, Inf), one(T), atol = tol)
            xSol .= 0.0
            ldiv!(xSolₛ, F, b)

            norm_xSol_1 = norm(xSol, 1)
            solve_Axb = norm(A*xSol - b, Inf) / max(norm(b, Inf), 1.0) <= tol
            is_solution = isapprox((norm_xSol_1 - dot(w, b))/ norm_xSol_1, zero(T), atol = tol)
            if solve_Axb && is_solution        
                return sparse(xSol), :success
            end
        end
        return xSol, :failure
    catch
        return xSol, :failure
    end
end