"""
Download test sets for the Basis Pursuit problem.
"""

using DrWatson
@quickactivate "BP_MAP"

import Downloads
using ProgressBars

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

"Download the basis porsuit testset from Lorentz, Pfetsch, and Tillmann"
function dowload_l1testset()
    testurl = "http://wwwopt.mathematik.tu-darmstadt.de/spear/software/L1_Comparison/SPEAR_L1_Testset_mat.zip"

    # Avoid downloading multiple times
    destdir = datadir("exp_raw", "L1_Testset_mat")
    if isdir(destdir)
        @info destdir * " already exists, so no download is necessary."
        return nothing
    end

    # Create a progress bar as the download is long
    updatebar = progressbar_factory()
    updatebar(0, 0)

    # Download test set
    basedatadir = datadir("exp_raw")
    if !isdir(basedatadir)
        mkpath(basedatadir)
    end
    filename = joinpath(basedatadir, "l1testset.zip")
    Downloads.download(testurl, filename; progress = updatebar)

    # Unzip test set
    unzipcmd = `unzip $filename -d $basedatadir`
    run(unzipcmd)

    # Delete zip file
    rm(filename)
end

"Download and unpack all testsets"
function main()
    dowload_l1testset()
end

main()