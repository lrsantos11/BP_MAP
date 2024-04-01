"""
Defines the Basis Porsuit problem type (BPProblem) and auxiliary functions.
"""
module BP

using LinearAlgebra

export BPProblem
export readl1test, solvewithLP

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

    Construct a BPProblem checking dimensions.
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

"Construct a BPproblem without a solution or optimal value"
function BPProblem(A, b)
    T = eltype(A)
    Tnan = zero(T) / zero(T)

    m, n = size(A)
    sol = fill(Tnan, n)
    optval = Tnan
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

include("BPUtils.jl")
using .BPUtils

end