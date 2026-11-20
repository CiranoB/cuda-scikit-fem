# GMRES-Compatible Examples

This campaign follows the CG benchmark protocol: CPU `spsolve` provides the
reference, five GPU preconditioner cases and four CPU counterparts are timed,
and independent campaigns are repeated three times by default.

## Selection

The harness reads the `GMRES` column of
[`../solver_suitability.csv`](../solver_suitability.csv), currently examples
**1, 2, 4, 5, 18, 24, 25, 26, 28, 34, 36, 37, 47, 50**.  `EXAMPLES` overrides
that selection.  The survey uses mutually exclusive solver categories; this
does not imply that CG-compatible matrices cannot also be solved by GMRES.
Structural eligibility does not guarantee iterative convergence.

Examples run as Python modules so relative imports, including example 26's
import of example 17, work.  Only the first default linear solve per process
is benchmarked, including solves performed during such imports.

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
CPU cases are `cpu_none`, `cpu_jacobi`, `cpu_block_jacobi`, and
`cpu_polynomial`; set `GMRES_BENCH_CPU=0` for GPU-only measurements.
CPU cases use the GPU dtype, controlled by `HALF_PRECISION`; the wrapper
sets it to `1` (float32).  CPU timings exclude GPU transfer and readback.

`GMRES_CASE_TIMEOUT_S` defaults to 60 seconds and
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

Cases reporting `maxiter` or `timeout` are disabled at later levels of that
example, independently for CPU and GPU.  Refinement stops on resource or time
limits, a subprocess error, unchanged mesh size, or failure of all enabled
cases.  Convergence flags and the measured relative residual must be checked
before interpreting timings as speedups.