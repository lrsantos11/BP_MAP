using DrWatson
@quickactivate :BP_MAP
using Statistics
using LinearAlgebra
using MATLAB

include(scriptsdir("download_utils.jl"))
function download_L1Homotopy()
    # Download L1-homotopy from the repository
    L1H_url = "https://github.com/sasif/L1-homotopy"
    L1H_commit = "12c201f" # frozen master HEAD (repo has no tags/releases)
    L1H_path = datadir("exp_raw", "L1-homotopy")

    if isdir(L1H_path)
        @info L1H_path * " already exists, so no download is necessary."
    else
        run(`git clone $L1H_url $L1H_path`)
        run(`git -C $L1H_path checkout $L1H_commit`)
    end

    # Apply patch to L1-homotopy (MATLAB R2021b compatibility: `break` inside a
    # script-file that shares the caller's loop is no longer legal; see patch)
    L1Hpatch = scriptsdir("L1Homotopy.patch")
    try
        apply_patch(L1Hpatch, L1H_path)
    catch e
        @info "Patch already applied"
    end
    return L1H_path
end

# Add L1-homotopy paths to MATLAB (only the folders BPDN_homotopy_function.m needs)
let L1H_path = download_L1Homotopy()
    mxcall(:addpath, 0, joinpath(L1H_path, "Pursuits_Homotopy"))
    mxcall(:addpath, 0, joinpath(L1H_path, "utils"))
end

function solveBP_L1Homotopy(
    prob::AbstractBPP;
    verbose::Bool = false,
    compute_time::Bool = true,
    usehoc::Bool = false,
    δ_HOC::AbstractFloat = 1e-9,   # HOC support-thresholding tolerance (Lorenz et al. 2015 §3.6)
    tau::AbstractFloat = 1e-10,    # homotopy path stopping parameter, NOT δ_HOC (different
                                    # concept: Lorenz et al. 2015 §5.3 confirm tau=0 exactly is
                                    # degenerate for this implementation ("theory only guarantees
                                    # convergence... if the final regularization parameter is set
                                    # to 0. However, if we use 10^-9 instead of 0, we get a
                                    # completely different picture... solves all instances with
                                    # high accuracy"); we use 1e-10, slightly more conservative
    kwargs...,
)
    # Bring variables into scope
    A = prob.A
    b = prob.b
    _, n = size(A)

    xL1H = similar(b, n)
    it_total = 0
    mat"""
    tic
    in = struct();
    in.tau = $tau;
    in.x_orig = zeros($n,1);
    in.record = 0;
    in.delx_mode = 'qr';
    out_l1h = BPDN_homotopy_function($A, $b, in);
    $x = out_l1h.x_out;
    $it = out_l1h.iter;
    $matlab_time = toc;
    """

    # median-of-repeats timing trick, same idea as solveBP_ISAL1
    times = Float64[]
    if compute_time && matlab_time < 10
        rounds = 10 ÷ matlab_time
        for _ = 1:rounds
            mat"""
            tic
            out_l1h = BPDN_homotopy_function($A, $b, in);
            $matlab_time = toc;
            """
            push!(times, matlab_time)
        end
    else
        push!(times, matlab_time)
    end
    elapsed_time = median(times)
    xL1H .= x
    it_total = it

    status = :completed
    if usehoc
        xhoc, hocstatus = heuristic_optimality_check(xL1H, prob; δ = δ_HOC)
        if hocstatus == :success
            xL1H = xhoc
        end
        status = hocstatus
    end

    if verbose
        @info "L1Homotopy iterations: $it_total"
        @info "L1Homotopy time: $elapsed_time"
        @info "L1Homotopy status: $status"
    end
    return xL1H, elapsed_time, it_total, status
end
