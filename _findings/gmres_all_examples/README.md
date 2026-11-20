# GMRES-Compatible Examples

This campaign follows the CG benchmark protocol: CPU `spsolve` provides the
reference, five GPU preconditioner cases and four CPU counterparts are timed,
and independent campaigns are repeated ten times by default.

## Selection

The harness reads the `GMRES` column of
[`../solver_suitability.csv`](../solver_suitability.csv), currently examples
**1, 2, 4, 5, 18, 24, 25, 26, 28, 34, 36, 37, 47, 50**.  `EXAMPLES` overrides
that selection.  The survey uses mutually exclusive solver categories; this
does not imply that CG-compatible matrices cannot also be solved by GMRES.
Structural eligibility does not guarantee iterative convergence.

Examples run as Python modules so relative imports, including example 26's
import of example 17, work.  The harness sets `GMRES_BENCHMARK_TARGET_FILE`
so only the first explicit default `solve(...)` called by the target example
is benchmarked.  Library-internal projections and solves made by imported
examples do not consume that measurement.  In particular, example 24 measures
its Stokes system rather than the inlet projection, and example 26 measures
its own restricted system rather than example 17's system.
Benchmark subprocesses also set `GMRES_BENCHMARK_EXIT_AFTER_SOLVE=1` and exit
successfully after that solve's measurements, before unrelated later solves,
postprocessing, or nonlinear/time-stepping iterations.  Normal example runs
do not enable this mode.

## Run

From the repository root:

```sh
docs/examples/run_gmres_compatible_examples.sh
EXAMPLES=47 MAX_LEVEL=1 REPEATS=3 docs/examples/run_gmres_compatible_examples.sh
```

Results are isolated under
`docs/examples/gmres_all_examples_results/<timestamp>/` unless `OUTPUT_ROOT`
is set.  Each `run_N/` contains `results.csv`, `status.csv`, `campaign.log`,
and per-example `run_logs/`.  `run.env` records the campaign settings.
Reference results are not overwritten.

The shared CG aggregator writes `aggregate.csv` with timing means and sample
standard deviations for common `(example, refined)` pairs.  At least two
numeric samples are required per metric; `REPEATS=1` keeps raw measurements
but produces no aggregate timing rows.

## Cases And Limits

GPU cases are `none`, `jacobi`, `ilu0`, `block_jacobi`, and `polynomial`.
The `ilu0` case explicitly requests `spilu(..., fill_factor=1)`, CuPy's
zero-fill GPU factorization without pivoting.  A factorization failure is
reported as `error`, not silently substituted with Jacobi.  Older campaigns
using the default `spilu(...)` instead measured CPU-built threshold ILU with
GPU triangular solves, despite the `ilu0` label; those timings are not
measurements of the corrected case.
CPU cases are `cpu_none`, `cpu_jacobi`, `cpu_block_jacobi`, and
`cpu_polynomial`; set `GMRES_BENCH_CPU=0` for GPU-only measurements.
CPU cases use the GPU dtype, controlled by `HALF_PRECISION`; the wrapper
defaults to `1` (float32); `HALF_PRECISION=0` enables float64 diagnostics.
CPU timings exclude GPU transfer and readback.
Both devices recompute the residual using the original host system.  A solver
success flag with a nonfinite residual or residual above `TOLERANCE` is reported
as `residual_mismatch`, with `converged=0`; it is not a valid speedup sample.

The wrapper's defaults reproduce the settings logged by the CG server
campaign
[`20261019T103612`](../../docs/examples/cg_all_examples_results/server_results/20261019T103612/run_1/campaign.log),
translated to GMRES:

| Setting | Wrapper Default |
|---------|----------------:|
| `REPEATS` | `10` |
| `START_LEVEL` / `MAX_LEVEL` | `1` / `100` |
| `PER_RUN_TIMEOUT_S` | `1000` |
| `DOF_CAP` | `5000000` |
| `SPSOLVE_CAP_S` | `300` |
| `GMRES_CASE_TIMEOUT_S` | `30` |
| `GMRES_CPU_CASE_TIMEOUT_S` | `100` |
| `GMRES_BENCH_CPU` | `1` |
| `GMRES_MAXITER` | `5000` |
| `GMRES_RESTART` | `20` (GMRES-specific) |
| `GMRES_BLOCK_SIZE` | `8` |
| `GMRES_POLY_DEGREE` | `3` |
| `TOLERANCE` | `1e-5` |

These values are exported explicitly and recorded in `run.env`, together
with `protocol_reference` identifying the CG campaign.  Environment overrides
remain supported.  The historical CG campaign selected 24 CG-marked examples;
this script selects the current GMRES-marked examples instead.

All ten historical CG repetitions logged identical settings.  Each produced
396 measurements across 22 examples; two selected examples had no default
linear solve.  Individual repetitions took 32,676--33,018 seconds (about
9.1 hours).  These durations describe the CG campaign, not an estimate for
GMRES.  Of 3,953 aggregate timing rows, 3,949 used all ten samples; the
remaining four used 9, 5, 3, and 2 numeric samples respectively.

When invoking the Python harness directly, `GMRES_CASE_TIMEOUT_S` defaults to
60 seconds and
`GMRES_CPU_CASE_TIMEOUT_S` to 100 seconds per solve.  `GMRES_MAXITER` defaults
to 5000 inner iterations; `GMRES_RESTART` defaults to the solver's restart
length of 20, limited by system size.  The CPU iteration cap is converted to
restart cycles, matching CuPy's full-cycle rounding.  SciPy uses left
preconditioning whereas CuPy uses right preconditioning; these are library
comparisons, not identical implementations.  `iters` counts restart-boundary
solution callbacks, and CuPy includes an initial-iterate callback.

`GMRES_BLOCK_SIZE`, `GMRES_POLY_DEGREE`, and `TOLERANCE` control the
preconditioners and convergence tolerance.  `START_LEVEL`, `MAX_LEVEL`,
`DOF_CAP`, `SPSOLVE_CAP_S`, and `PER_RUN_TIMEOUT_S` bound refinement and
runtime.  The wrapper defaults to 1000 seconds per subprocess.

Cases reporting `maxiter`, `timeout`, `error`, or `residual_mismatch` are
disabled at later levels of that example, independently for CPU and GPU.
When all enabled cases have
failed, `all_cases_disabled` stops refinement, even at level 1; `MAX_LEVEL`
is a ceiling, not a required number of successful measurements.  Refinement
also stops on resource or time
limits, a subprocess error, unchanged mesh size, or failure of all enabled
cases.  A subprocess timeout retains any already emitted measurements and
diagnostic output.  Convergence flags and the measured relative residual must be checked
before interpreting timings as speedups.

Single precision can be the limiting factor even for small systems.  In a
local diagnostic, example 34 stalled near a `7e-3` true residual in float32;
float64 with a full 32-vector restart converged near `4.8e-11`.  Increasing
`MAX_LEVEL` alone cannot fix that accuracy floor.  Precision and restart
changes should be recorded as separate experiments, not mixed into an
existing campaign.

## Local Validation: 2026-11-20

One diagnostic campaign completed with a ten-level ceiling on an NVIDIA
GeForce RTX 4060 Laptop GPU (8 GB): **30 recorded configurations across all
14 selected examples in 2,671 seconds** (44 minutes 31 seconds).  It used
float32, tolerance `1e-5`, restart 20, 5000 inner iterations, and the wrapper's
existing time/resource limits.  Python 3.13.18, NumPy 2.5.5, SciPy 1.18.1,
and the prebuilt `cupy-cuda13x` 14.1.1 package were used in a temporary UV
environment; project dependency declarations were not changed.

Raw [measurements](../../docs/examples/gmres_all_examples_results/local_validation_corrected/direct_run/results.csv),
[stop records](../../docs/examples/gmres_all_examples_results/local_validation_corrected/direct_run/status.csv),
and [per-configuration logs](../../docs/examples/gmres_all_examples_results/local_validation_corrected/direct_run/run_logs/)
are preserved separately from the historical reference data.

| Examples | Recorded Levels | Stop Reason |
|----------|----------------:|-------------|
| 1, 2, 4, 18, 24, 26, 28, 34 | 1 each | All enabled cases failed |
| 5 | 2 | All enabled cases failed |
| 25, 50 | 3 each | All enabled cases failed |
| 36 | 2 | Subprocess timeout at level 2 |
| 37 | 2 | Level 3 timed out before producing measurements |
| 47 | 10 | Reached the requested level ceiling |

At the largest configuration with a valid GPU solution for each example:

| Example | DoFs | GPU Case | CPU `spsolve` (s) | GPU Total (s) | CPU / GPU |
|--------:|-----:|----------|-----------------:|--------------:|----------:|
| 5 | 4,161 | none | 0.008841 | 0.297728 | 0.030 |
| 25 | 161,028 | none | 1.456132 | 0.866619 | 1.68 |
| 36 | 12,188 | ILU(0) | 1.869069 | 0.804593 | 2.32 |
| 37 | 106,920 | Jacobi | 86.250364 | 0.162232 | 531.65 |
| 47 | 91 | polynomial | 0.000102 | 0.009593 | 0.011 |
| 50 | 2,373 | none | 0.004844 | 0.385857 | 0.013 |

Only finite true residuals at or below `1e-5` qualify.  Examples
1, 2, 4, 18, 24, 26, 28, and 34 had no valid GPU solution at the attempted
levels.  This does not invalidate their mathematical GMRES eligibility.
Example 47 reaches ten levels but remains too small to amortize GPU overhead.

The diagnosis found and fixed incorrect first-solve selection (example 24's
inlet projection and example 26's imported example), a mislabeled threshold
ILU case, discarded partial timeout output, and continued retries of errored
cases.  Strict residual flags and benchmark-only early exit were added too.
The completed campaign started before those last two reporting/runtime changes
were loaded: its raw CPU success flags for example 5 level 1 (`cpu_none`,
`1.05e-5`) and example 25 level 1 (`cpu_jacobi`, `1.001e-5`) must be excluded.
The summary above applies that check without rewriting the original CSV.

Example 36's first level-2 CPU solve took 394 seconds; after the benchmark,
later Newton solves continued until the 1000-second timeout.  Early exit
now prevents that unrelated continuation and was verified separately on the
real level-1 example.  Example 37's level-3 timeout has no emitted benchmark
header, so no GPU measurement or convergence conclusion is available there.

Separate [precision/restart controls](../../docs/examples/gmres_all_examples_results/local_validation_corrected/precision_probe_single_thread.csv)
show example 1 stagnating near `1.1e-4` in float32 even with restart 100, but
converging below `1e-5` in float64.  The actual GPU
[float64 check](../../docs/examples/gmres_all_examples_results/local_validation_corrected/fp64_ex01.log)
also converged (true residual `9.749e-6`).  Example 2 still failed in float64;
restart 100 improved its residual from about `0.101` to `0.00387` without
meeting the target.  Thus precision floors and restarted-GMRES stagnation
are distinct causes of the early stops, not failures to apply refinement.

These are diagnostic single-run observations, not publication-ready repeated
statistics.  Preliminary interrupted attempts remain separate.  The final
regression suite contains 23 passing tests, and a real GPU ILU(0) application
plus the real benchmark-only early-exit path were checked independently.