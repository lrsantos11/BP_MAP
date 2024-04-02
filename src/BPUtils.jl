"""
Utilities for Basis Pursuit problems.
"""

@reexport module BPUtils 

using ..BP

using LinearAlgebra
using SparseArrays
using JuMP
using HiGHS
using MAT

export readl1test, solvewithLP, heuristic_optimality_check

"Read a test from the Lorentz, Pfetsch, and Tillmann testset"
function readl1test(filename; sparse_matrix::Bool=false)
    # Verify if the test is available
    # dir = datadir("exp_raw", "L1_Testset_mat")
    # if !isdir(dir)
    #     println(dir)
    #     @error "Test set is not available"
    # end
    # filename = joinpath(dir, filename)
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
    return BPProblem(A, data["b"][:], data["x"][:])
end

"Solve a BPProblem using a regular linear programming solver"
function solvewithLP(prob::BPProblem; Solver = HiGHS)

    # Create a LP model that represents the basis pursuit problem
    model = Model(Solver.Optimizer)
    set_optimizer_attribute(model, "log_to_console", false)
    set_optimizer_attribute(model, "OutputFlag",  0)

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


"""
    Heuristic for optimality Check from [Lorenz2014, Alg. 2]

"""
function heuristic_optimality_check(xSol, Affine; δ::AbstractFloat = 1e-4, tol::AbstractFloat = 1e-12)
    T = eltype(xSol)
    m, n = size(Affine.A)
    b = @views Affine.b
    A = @views Affine.A
    S = findall(x -> abs(x) > δ, xSol) # [Lorenz2014, Eq. (1)]
    Aₛ = @views A[:, S]
    xSolₛ = @views xSol[S]
    ## TODO: Improve with CG instead of "small" QR (See [Lorenz2014, pg. 4])
    F = qr(Aₛ)
    w = F' \ sign.(xSolₛ)
    if isapprox(norm(A'w, Inf), one(T), atol = tol)
        xSol .= 0.0
        ldiv!(xSolₛ, F, b)
        norm_xSol_1 = norm(xSol, 1)
        if isapprox((norm_xSol_1 - dot(w, b))/ norm_xSol_1, zero(T), atol = tol)
            return sparse(xSol), :success
        end
    end
    return xSol, :failure
end


end
