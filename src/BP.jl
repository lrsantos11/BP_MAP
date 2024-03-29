"""
Defines the Basis Porsuit peoblem type (BPProblem) and auxliary functions.
"""

module BP

export BPProblem
export readl1test, solvewithLP

using DrWatson
@quickactivate "BP_MAP"

using LinearAlgebra
using JuMP
using HiGHS
using MAT

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
function BPProblem(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T<:AbstractFloat}
    Tnan = zero(T) / zero(T)
    
    m, n = size(A)
    sol = fill(Tnan, n)
    optval = Tnan
    return BPProblem(A, b, sol, optval)
end

"Construct a BPProblem computing the optimal value from the given solution"
BPProblem(
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    sol::AbstractVector{T},
) where {T<:AbstractFloat} = BPProblem(A, b, sol, norm(sol, 1))

"Construct a BPProblem giving the optimal value but no solution"
function BPProblem(
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    optval::T,
) where {T<:AbstractFloat}
    Tnan = zero(T) / zero(T)
    m, n = size(A)
    sol = fill(Tnan, n)
    return BPProblem(A, b, sol, optval)
end

"Read a test from the Lorentz, Pfetsch, and Tillmann testset"
function readl1test(filename)
    # Verify if the test is available
    dir = datadir("exp_raw", "L1_Testset_mat")
    if !isdir(dir)
        println(dir)
        @error "Test set is not available"
    end
    filename = joinpath(dir, filename)
    if !isfile(filename)
        @error "Test file does not exist"
    end

    data = matread(filename)
    # The solution is represented as a one column matrix. Get the respective vector instead.
    return BPProblem(data["A"], data["b"][:, 1], data["x"][:, 1])
end

"Solve a BPProblem using a regular linear programming solver"
function solvewithLP(prob::BPProblem, Solver = HiGHS)

    # Create a LP model that represents the basis pursuit problem
    model = Model(Solver.Optimizer)
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