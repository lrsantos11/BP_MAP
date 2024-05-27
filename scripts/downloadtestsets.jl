"""
Download test sets for the Basis Pursuit problem.
"""

using DrWatson
@quickactivate "BP_MAP"

using LinearAlgebra
import Downloads
using ProgressBars
using SparseArrays
using MAT
using Lasso

"Create a progress bar to track downloads"
function progressbar_factory()
    pbar = ProgressBar(total = 100)

    bar_status = 0
    function updatebar(total, now)
        total = total == 0 ? 1 : total
        new_status = div(100 * now, total)
        update(pbar, max(0, new_status - bar_status))
        bar_status = new_status
    end

    return updatebar
end

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

"Convert ther Lasso problem in filename to BP format"
function Lasso2BP(filename)
    # Sparsity targets
    targets = [0.01, 0.05, 0.10, 0.20]
    @info "Converting $(basename(filename))"
    p = matread(filename)
    m, n = size(p["A"])
    try
        lastnnzratio, λminratio = 0.0, 1.0e-2
        while lastnnzratio < targets[end]
            λminratio /= 10
            global lf = fit(
                LassoPath,
                p["A"],
                p["b"][:, 1];
                α = 1.0,
                intercept = false,
                standardize = false,
                λminratio = λminratio
            )
            lastnnzratio = nnz(lf.coefs[:, end]) / m
        end
        p["optval"] = NaN
        b = Matrix{Float64}(undef, m, 0)
        for t in targets
            best = argmin(abs.([nnz(lf.coefs[:,i]) / m for i = 1:length(lf.λ)] .- t))
            b = hcat(b, p["A"] * lf.coefs[:, best])
            @info "NNZ for target $t is $(nnz(lf.coefs[:, best]) / m)"
        end
        p["b"] = b
        delete!(p, "ftarget")
        delete!(p, "lambda")
        rm(filename)
        matwrite(filename, p; compress = true)
    catch e
        if isa(e, OutOfMemoryError)
            @error "Out of memory"
            @error "Deleting $filename"
            rm(filename)
        else
            throw(e)
        end
    end
end

"Dowload and covert Lasso test set from Lopes, Santos and Silva"
function getLasso2BP()
    testurl = "https://drive.usercontent.google.com/download?id=1T4gCmV9rJ86jzPhRZ6B7ERQVSbvdsagU&export=download&authuser=1&confirm=t&uuid=46a74bc8-6ae6-40f8-9e75-f25228b2796f&at=APZUnTVUk8zQ_rr4YwCrQwdySM60:1712342173600"
    destdir = "Data-Lasso"
    downloaded = dowload_l1testset(testurl, destdir)
    if downloaded
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
function main()
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
    main()
end