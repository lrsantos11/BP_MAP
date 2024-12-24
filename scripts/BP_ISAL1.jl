using DrWatson
@quickactivate :BP_MAP
using LinearAlgebra
using BenchmarkTools
using MATLAB
using :BP_MAP

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
    ISAL1patch_path = datadir("exp_raw", "ISAL1_v1.0")
    try
        apply_patch(ISALpatch, ISAL1patch_path)
    catch e
        @info "Patch already applied"
    end
    return ISAL1_path
end

# Add ISAL1 path to MATLAB
mxcall(:addpath, 0, download_ISAL1())

function solveBP_ISAL1(
    prob::AbstractBPP;
    verbose::Bool = false,
    compute_time::Bool = true,
    kwargs...,
)
    # Bring variables into scope
    A = prob.A
    b = prob.b
    displ = verbose ? 1 : Inf

    # run MATLAB ISAL_1 function to solve the problem
    _, num_cols = size(A)
    xISAL = similar(b, num_cols)
    it_total = 0
    status = 0
    mat"""
    tic
    % Initialization
    [$x, $fval, $err, $exfl, $it] = ISAL1($A, $b, 1, -1, $displ);
    $matlab_time = toc; 
    """

    # If we need to compute time and the solution was too fast
    elapsed_time = 0.0
    if compute_time && matlab_time < 10
        rounds = 10 ÷ matlab_time
        for _ = 1:rounds
            mat"""
            tic
            % Initialization
            [$x, $fval, $err, $exfl, $it] = ISAL1($A, $b, 1, -1, $displ);
            $matlab_time = toc; 
            """
            elapsed_time += matlab_time
        end
        elapsed_time /= rounds
    else
        elapsed_time = matlab_time
    end
    xISAL .= x
    it_total = it
    status = exfl
    if verbose
        @info "ISAL1 status: $exfl"
        @info "ISAL1 iterations: $it"
        @info "ISAL1 error: $err"
        @info "ISAL1 time: $elapsed_total"
    end
    return xISAL, elapsed_time, it_total, status
end
