# General Rules
In case of doubts, or missing information, you must inquiry more accurate information before doing anything.
Does not change stuff where is not required, if a specific function is the target of a request just change the specific target. You may warn if this change can break something elsewhere.
Docstrings that repeat what is on the method signature are useless. You only must insert something in docstring if there's is something assumed that are not possible to check in the source code.

# Project Context
This is `cuda-scikit-fem`, a GPU-accelerated fork of [`scikit-fem`](https://github.com/kinnala/scikit-fem). Upstream is a pure-Python FEM assembly library (no compiled code). This fork adds CUDA/GPU sparse linear solvers via CuPy on top of that assembly layer.
- The active development branch is `feat/multiple-solvers-on-gpu`; upstream default is `master`.
- The importable package is still `skfem/`.

# GPU / Solver Modules (fork-specific, prioritize these)
- `skfem/sparse_solvers.py`: GPU solvers (CG, GMRES, spsolve) via `cupyx.scipy.sparse.linalg`, plus preconditioners (Jacobi, ILU). Honors the `HALF_PRECISION` env var. This is the core of the fork.
- `skfem/solvers_aplicability.py`: CPU-side checks (`is_symmetric`, `is_cg_compatible`, ...) that decide which solver is suitable for a given matrix.
- `skfem/utils.py`: `solver_direct_scipy` returns the CPU `spsolve` result and, via `run_enabled_gpu_solvers`, also runs each GPU solver opted-in with `{NAME}_GPU_ENABLE` (`SPSOLVE`/`SUPERLU`/`CG`/`GMRES`, e.g. `CG_GPU_ENABLE=1`); default off => CPU-only. CG uses a Jacobi preconditioner.
- `_findings/`: benchmarks and empirical results backing the thesis (e.g. `gmres_gpu_vs_spsolve_cpu/`, `ex33_cprofile/`). Consult these before making performance claims.

# UV Package manager
In order to run the programs, tests and anything that lies on the virtual env of this project, you can run using `uv run` before the prompt. It will automatically use the right env. for you.
- Some scripts (benchmarks under `_findings/`) require the repo root on the path: `PYTHONPATH=. uv run python <script>`.


# Thesis
This fork is highly related to a Thesis to obtain Master Degree, so it is possible to find the main article on _thesis/thesis/seminar-progress-english.tex
If there's a new find about it, leverage in consideration to update it.
- Focus: CG applied to the CG-compatible exercises. `is_cg_compatible` requires SPD matrices (square, symmetric, positive-definite); present CG as the completed solver work and keep GMRES in Future Work unless asked otherwise.
- `_thesis/thesis/Makefile` is stale: it targets `seminar-progress.tex` (`PAPER=seminar-progress`), which does not exist (the tracked sources are `seminar-progress-english.tex`, `seminar-progress-portuguese.tex`, and the unrelated `source.tex`), so `make` fails. Compile `seminar-progress-english.tex` directly when validating that file.

# Latex
It is not needed to run latexmk after editing, the user will check manually. 