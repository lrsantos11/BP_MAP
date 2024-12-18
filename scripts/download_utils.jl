"""
Download utils for BP MAP.
"""

using DrWatson
@quickactivate :BP_MAP

using Downloads
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

function apply_patch(patch_file, target_dir)
    # Comando para aplicar o patch
    patchcmd = pipeline(`patch -p1 --directory=$target_dir`; stdin=patch_file)
    run(patchcmd)
end
