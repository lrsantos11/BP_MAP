# Replace, in benchmark CSVs written by test/runbenchmark.jl, the values of the runs redone
# by test/rerun_isal_timelimit.jl or test/rerun_isal_dist.jl. Every column of the re-run
# CSV other than Problem, M and N overwrites the same column of the benchmark row with the
# same Problem; as Problem ends with the right-hand side (" - k"), each re-run row matches
# a single LSS_benchmark_CUDA_k.csv. Every benchmark CSV that changes is first copied to
# <file>.bak, and the merge refuses to run if such a backup already exists, so the original
# data are never overwritten twice. Usage, from the project directory:
#
#     julia --project test/merge_reruns.jl rerun.csv benchmark1.csv [benchmark2.csv ...]

using DrWatson
@quickactivate "BP_MAP"

using CSV
using DataFrames

function merge_reruns(rerunfile, benchfiles)
    rerun = CSV.read(rerunfile, DataFrame)
    cols = setdiff(names(rerun), ["Problem", "M", "N"])
    allunique(rerun.Problem) || error("$rerunfile has repeated problems")

    # Check everything before touching any file
    benches = Dict(file => CSV.read(file, DataFrame) for file in benchfiles)
    for (file, bench) in benches
        any(in(rerun.Problem), bench.Problem) || continue
        cols ⊆ names(bench) || error("$file lacks some of the columns $cols")
        isfile(file * ".bak") && error("$(file).bak already exists; move it away to merge again")
    end
    unmatched = filter(p -> !any(b -> p in b.Problem, values(benches)), rerun.Problem)
    isempty(unmatched) || error("Re-run problems not found in the benchmark files: $unmatched")

    for file in benchfiles
        bench = benches[file]
        rows = findall(in(rerun.Problem), bench.Problem)
        isempty(rows) && continue
        for i in rows
            new = rerun[findfirst(==(bench.Problem[i]), rerun.Problem), :]
            (bench.M[i], bench.N[i]) == (new.M, new.N) ||
                error("Size of $(new.Problem) differs in $rerunfile and $file")
            change = join(["$c: $(bench[i, c]) -> $(new[c])" for c in cols], ", ")
            println("$(new.Problem) in $(basename(file)): $change")
            for c in cols
                bench[i, c] = new[c]
            end
        end
        cp(file, file * ".bak")
        CSV.write(file, bench)
        @info "Updated $(length(rows)) rows of $file (original kept in $(file).bak)"
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) >= 2 || error("Usage: merge_reruns.jl rerun.csv benchmark1.csv [benchmark2.csv ...]")
    merge_reruns(ARGS[1], ARGS[2:end])
end
