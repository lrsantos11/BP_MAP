"""
Download test sets for the Basis Pursuit problem.
"""

using DrWatson
@quickactivate :BP_MAP

using LinearAlgebra
import Downloads
using SparseArrays
using MAT
using Lasso
using Krylov

include(scriptsdir("download_utils.jl"))

"Download a testset"
function dowload_l1testset(testurl, destdir)
    # Avoid downloading multiple times
    destdir = datadir("exp_raw", destdir)
    if isdir(destdir)
        @info destdir * " already exists, so no download is necessary."
        return false
    end

    # Create a progress bar as the download is long
    updatebar = progressbar_factory()
    updatebar(0, 0)

    # Download test set
    basedatadir = datadir("exp_raw")
    if !isdir(basedatadir)
        mkpath(basedatadir)
    end
    filename = joinpath(basedatadir, "tmpbptestset.zip")
    Downloads.download(testurl, filename; progress = updatebar)

    # Unzip test set
    unzipcmd = `unzip $filename -d $basedatadir`
    run(unzipcmd)

    # Delete zip file
    rm(filename)
    return true
end

"Eliminate reduntant lines from A and RHS"
function simplify(A, b)
    @info "Trying to delete redundant lines from A"
    if 8 * prod(size(A)) > Int(Sys.total_memory()) ÷ 4
        @error "Not enough memory"
        return A, b
    end

    m, _ = size(A)
    At = convert(SparseMatrixCSC, A')
    fact = qr(At)
    valid =
        [abs(fact.R[i, i] / norm(At[:, fact.pcol[i]])) > sqrt(eps(eltype(A))) for i = 1:m]
    @info "Deleted $(m - sum(valid)) lines"
    valid = fact.pcol[valid]
    At, b = At[:, valid], b[valid, :]
    @info "Done"
    return convert(SparseMatrixCSC, At'), b
end

"Try to solve Ax = b to see if the matrix is too badly conditioned"
function check_residual(A, b, threshold = 1.0e-10)
    @info "Solving a linear system Ax = b to check residual"
    # If the dense version of the matrix s less than 1 / 4 of the memory
    # available a full factorization.
    if 8 * prod(size(A)) < Int(Sys.total_memory()) ÷ 4
        @info "Using factorization"
        xsol = A \ b
        resnorm = norm(A * xsol - b) / max(1.0, norm(b))
        solved = resnorm < threshold
    else
        @info "Using CGNE"
        xsol, stats = cgne(A, b)
        solved = stats.solved
    end
    if solved
        @info "Done"
    else
        @error "Could not solve system to desired precision"
    end
    return solved
end

"Convert ther Lasso problem in filename to BP format"
function Lasso2BP(filename)
    # Sparsity targets
    targets = [0.01, 0.05, 0.10, 0.20]
    @info "Converting $(basename(filename))"
    p = matread(filename)
    A, b = p["A"], p["b"][:, 1]
    @info "Dimensions of A: $(size(A))"
    m, _ = size(A)
    try
        lastnnzratio, λminratio = 0.0, 1.0e-2
        while lastnnzratio < targets[end]
            λminratio /= 10
            global lf = fit(
                LassoPath,
                A,
                b;
                α = 1.0,
                intercept = false,
                standardize = false,
                λminratio = λminratio,
            )
            lastnnzratio = nnz(lf.coefs[:, end]) / m
        end
        b = Matrix{Float64}(undef, m, 0)
        for t in targets
            best = argmin(abs.([nnz(lf.coefs[:, i]) / m for i = 1:length(lf.λ)] .- t))
            b = hcat(b, A * lf.coefs[:, best])
            @info "NNZ for target $t is $(nnz(lf.coefs[:, best]) / m)"
        end
        
        A, b = simplify(A, b)
        good_condition = check_residual(A, Vector(b[:, 2]))
        if good_condition
            p["optval"] = NaN
            p["b"] = b
            p["A"] = A
            delete!(p, "ftarget")
            delete!(p, "lambda")
            rm(filename)
            matwrite(filename, p; compress = true)
        else
            @error "Deleting $filename"
            rm(filename)
        end
    catch e
        if isa(e, OutOfMemoryError) || isa(e, SparseArrays.CHOLMOD.CHOLMODException)
            @error "Out of memory"
            @error "Deleting $filename"
            rm(filename)
        else
            throw(e)
        end
    end
end

"Dowload and convert Lasso test set from Lopes, Santos and Silva"
function getLasso2BP()
    testurl = "https://drive.usercontent.google.com/download?id=1T4gCmV9rJ86jzPhRZ6B7ERQVSbvdsagU&export=download&authuser=1&confirm=t&uuid=46a74bc8-6ae6-40f8-9e75-f25228b2796f&at=APZUnTVUk8zQ_rr4YwCrQwdySM60:1712342173600"
    destdir = "Data-Lasso"
    downloaded = dowload_l1testset(testurl, destdir)
    if true #downloaded
        fullpath = datadir("exp_raw", destdir)
        for filename in readdir(fullpath; join = true)
            if occursin("C", filename)
                Lasso2BP(filename)
            else
                @info "Ignoring $filename"
            end
        end
    end
end

"Download and unpack all testsets"
function downloadtestsets()
    # Download testset from Lorentz, Pfetsch, and Tillmann
    testurl = "http://wwwopt.mathematik.tu-darmstadt.de/spear/software/L1_Comparison/SPEAR_L1_Testset_mat.zip"
    destdir = "L1_Testset_mat"
    dowload_l1testset(testurl, destdir)

    # Download and convert to BP format testset from Lopes, Santos e Silva
    testurl = "https://www.ime.unicamp.br/~pjssilva/data/lassobp_mat.zip"
    destdir = "lassobp_mat"
    dowload_l1testset(testurl, destdir)
end

if abspath(PROGRAM_FILE) == @__FILE__
    downloadtestsets()
end