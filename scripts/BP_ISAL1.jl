using DrWatson
@quickactivate "BP_MAP"


using :BP_MAP
using LinearAlgebra

# Add MATLAB to the environment path - change this to your MATLAB installation path
matlab_env = "/opt/matlab/R2021b/"
try 
    using MATLAB
catch 
    using Pkg
    ENV["MATLAB_ROOT"] = matlab_env
    Pkg.build("MATLAB")
    using MATLAB
end


function solveBP_ISAL1(prob::AbstractBPP; verbose = false)
    # Bring variables into scope
    @views A = prob.A
    @views b = prob.b
    # Add ISAL1 path to MATLAB
    ISAL1_path = scriptsdir("ISAL1_v1.0/")
    mxcall(:addpath, 0, ISAL1_path)

    # run MATLAB ISAL_1 function to solve the problem
    x, fval, err, exfl, it = mxcall(:ISAL_1, 5, A, b)
    return x
end