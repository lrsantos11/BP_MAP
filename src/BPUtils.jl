"""
Utilities for Basis Pursuit problems.
"""

module BPUtils 

using ..BP

using LinearAlgebra
using JuMP
using HiGHS
using MAT

export readl1test, solvewithLP

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