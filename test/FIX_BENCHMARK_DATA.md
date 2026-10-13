# Fixing the ISAL results of the benchmark

## What was wrong

1. **ISAL ran with a 900 s time limit instead of 3600 s.** `solveBP_ISAL1` in
   `scripts/BP_ISAL1.jl` called `ISAL1(A, b, 1, -1, displ, time=3600)`. ISAL1 reads its
   options only as name/value pairs and silently ignores anything else, so every option was
   ignored and ISAL kept its default limit of 900 s. The other ignored values (`useHOC = 1`,
   `dLBmode = -1`, `dspl = displ`) equal ISAL1's defaults, so only the limit was affected.
   ISAL1 uses the limit only in its clock check, so runs that ended before 900 s are
   unaffected. That covers all of LPT (longest run 302 s) and all of LSS except 9 runs:
   SC13 (rhs 2, 3, 4), SC17 (rhs 2), SC7 (rhs 3, 4), SC11 (rhs 3, 4) and SC12 (rhs 4).
2. **Different distance thresholds on LPT.** A run counted as solved only with
   `dist < 1e-6` for ISAL and ℓ1-Homotopy, but `dist < 1e-4` for BP-MAP. All three now use
   1e-4. In `test/neuronio_rev1/LPT_benchmark.csv` no ℓ1-Homotopy run has
   1e-6 < dist < 1e-4, so only 18 ISAL runs can change.
3. **ℓ1-Homotopy's λ̄.** `test/runbenchmark.jl` now passes `tau = 1.0e-9`, the value stated
   in the paper; the earlier runs used the wrapper's default, 1e-10. The ℓ1-Homotopy columns
   are re-run separately and are not covered by these steps.

## Files of the fix

- `test/runbenchmark.jl`: distance thresholds and `tau = 1.0e-9`.
- `scripts/BP_ISAL1.jl`: ISAL1 options passed as name/value pairs.
- `test/rerun_isal_dist.jl`: re-times the 18 LPT ISAL runs with 1e-6 < dist < 1e-4.
- `test/rerun_isal_timelimit.jl`: re-times the 9 LSS ISAL runs that stopped at 900 s.
- `test/merge_reruns.jl`: writes the re-run values into the benchmark CSVs.

Both re-run scripts use `solve_with_ISAL` from `test/runbenchmark.jl` unchanged.

## Steps on neuronio

1. Pull the commit after the current τ = 1e-9 run has finished. The re-run scripts time
   ISAL, and any other job on the node would distort the timings.
2. From the project directory, run the two scripts one after the other. Each finished run
   is appended to the output CSV at once, so a script can be restarted after an
   interruption and skips the runs already in its output.

   ```
   julia --project test/rerun_isal_dist.jl LPT_ISAL_rerun.csv          # a few minutes
   julia --project test/rerun_isal_timelimit.jl LSS_ISAL_rerun.csv     # up to ~9 h
   ```

3. Merge the re-run values into the CSVs taken as final. This step needs neither Matlab nor
   a GPU, so it can run on any machine that has the two re-run CSVs; the paths below assume
   the final CSVs are in `test/neuronio_rev1`:

   ```
   julia --project test/merge_reruns.jl LPT_ISAL_rerun.csv test/neuronio_rev1/LPT_benchmark.csv
   julia --project test/merge_reruns.jl LSS_ISAL_rerun.csv test/neuronio_rev1/LSS_benchmark_CUDA_{1,2,3,4}.csv
   ```

   For each re-run row, `ISAL`, `ISAL dist` and `ISAL iters` overwrite the row with the
   same `Problem`. `Problem` ends with the right-hand side (`SC13.mat - 2`), so each LSS
   row goes to its own `LSS_benchmark_CUDA_k.csv`. Each changed CSV is first copied to
   `<file>.bak`, and the merge refuses to run while that backup exists, so the original
   data are never lost. The merge prints every old and new value.
4. Check the result. A positive `ISAL` time means solved. If `ISAL iters` equals the old
   value, neuronio reproduced the original run.

## If the final ISAL columns come from another run

The lists in the two re-run scripts were taken from the ISAL results in
`test/neuronio_rev1`. If the ISAL columns used in the paper come from a different run:

- `rerun_isal_dist.jl` is needed only if that run judged ISAL with the old 1e-6 threshold.
- `rerun_isal_timelimit.jl` is needed whenever that run used `scripts/BP_ISAL1.jl` from
  before this fix.
- In both cases the lists must be rebuilt from that run's CSVs: the LSS ISAL runs with
  |time| ≥ 900 s, and the LPT ISAL runs with 1e-6 < dist < 1e-4.
