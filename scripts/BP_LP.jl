using DrWatson
@quickactivate "BP_MAP"

using :BP_MAP

using LinearAlgebra
using JuMP
using HiGHS

"Solve a BP problem using Linear Programming"
function solveBP_LP(prob::AbstractBPP; solver = HiGHS.Optimizer, verbose = false)
    model = buildBP_LPModel(prob, solver = solver, verbose = verbose)
    return solveBP_LPmodel!(model)
end

"Build a LP model that describes the Basis Pursuit problem"
function buildBP_LPModel(prob::AbstractBPP; solver = HiGHS.Optimizer, verbose = false)
    # Create an LP model that represents the basis pursuit problem
    model = Model(solver)
    !verbose && set_silent(model)

    m, n = size(prob)
    @variable(model, xplus[1:n] >= 0)
    @variable(model, xminus[1:n] >= 0)
    @objective(model, Min, sum(xplus) + sum(xminus))
    @constraint(model, prob.A * xplus - prob.A * xminus == prob.b)

    return model
end

"Solve an LP model describing BP"
function solveBP_LPmodel!(model)
    optimize!(model)
    @assert termination_status(model) == OPTIMAL
    return value.(model[:xplus]) - value.(model[:xminus])
end


 