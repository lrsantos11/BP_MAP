using DrWatson
@quickactivate :BP_MAP


using LinearAlgebra
using BenchmarkTools

include(scriptsdir("download_utils.jl"))


function download_ISAL1()
    # Download ISAL1 from the repository
    ISAL1_url = "http://wwwopt.mathematik.tu-darmstadt.de/spear/software/ISAL1_v1.0.zip"
    ISAL1_path = datadir("exp_raw", "ISAL1_v1.0")
    ISAL1zip_file = datadir("exp_raw", "ISAL1_v1.0.zip")

    if isdir(ISAL1_path)
        @info ISAL1_path * " already exists, so no download is necessary."
        
    else

        # Create a progress bar as the download is long
        updatebar = progressbar_factory()
        updatebar(0, 0)

        # Download test set
        Downloads.download(ISAL1_url, ISAL1zip_file; progress = updatebar)

        # Unzip test set
        unizp_path = datadir("exp_raw")
        unzipcmd = `unzip $ISAL1zip_file -d $unizp_path`
        run(unzipcmd)

        # Delete zip file
        rm(ISAL1zip_file)
    end

    #Apply patch to ISAL1
    ISALpatch = scriptsdir("ISAL1.patch")
    ISAL1patch_path = "data/exp_raw/ISAL1_v1.0"
    try
        apply_git_patch(ISALpatch, ISAL1patch_path)
    catch e
        @info "Patch already applied"
    end
    return ISAL1_path
end


global ISAL1_path = download_ISAL1()

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


function solveBP_ISAL1(prob::AbstractBPP;
    # usehoc = false,
    # itmax::Int = 1_000,
    # ε::Number = 1e-6,
    verbose::Bool = false,
    ISAL1_path::String = ISAL1_path,
    kwargs...,
    )
    # Bring variables into scope
    @views A = prob.A
    @views b = prob.b
    # Add ISAL1 path to MATLAB
    mxcall(:addpath, 0, ISAL1_path)
    verbose ? displ = 1 : displ = Inf
    # run MATLAB ISAL_1 function to solve the problem
    mat"""
    tic
    % Initialization
    [$x, $fval, $err, $exfl, $it] = ISAL1($A, $b, 1, -1, $displ);
    $time = toc; 
    """
    verbose && begin @info "ISAL1 status: $exfl"
        @info "ISAL1 iterations: $it"
        @info "ISAL1 error: $err"
        @info "ISAL1 time: $time"
    end
    return x, time
end
