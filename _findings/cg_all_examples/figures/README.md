# Figures and numbers for the repeated CG campaign

Scripts that turn `docs/examples/cg_all_examples_results/server_results/<campaign>/run_*/results.csv`
(10 rounds of `../benchmark.py`) into the figures, the table and the statistics of the paper.
Run from this folder with `uv run python <script>`; the optional first argument is the campaign
directory (default: `20261019T103612`).

| Script | Output |
|--------|--------|
| `plot_scaling.py [results_dir] [out_dir]` | `exNN_cg_scaling.{pdf,png}` and `cg_all_examples_scaling.{pdf,png}` in `img/` (spsolve, GPU CG solid, CPU CG dashed) |
| `summarize.py [results_dir]` | Table rows (LaTeX), headline counts, preconditioner comparison, round-to-round statistics, accuracy statistics, CPU-vs-GPU CG statistics, stop reasons |
| `campaign.py` | Shared loader (not run directly) |

Conventions: a point is kept only if present in all rounds; a GPU variant counts as converged at a
point only if it converged in every round; times are means over the rounds, error bars and `±` are
the sample standard deviation.

Caveats:

- The campaign ran with `HALF_PRECISION=1`, i.e. GPU and CPU CG in `float32` while `spsolve` is
  `float64`. The solver's own convergence flag therefore does not guarantee a true relative
  residual below the tolerance; `summarize.py` reports the recomputed residuals.
- Example 15 reaches only 899 DoF in the campaign. The 8,999-DoF row of the paper table comes from
  `../../ex15_scaling/results.csv` (a single double-precision round) and is read by `summarize.py`.
- The Example 33 figures (`cprofile`, `cpu_vs_gpu`, `precision`) come from the dedicated sweeps in
  `_findings/ex33_*`, not from this campaign.
