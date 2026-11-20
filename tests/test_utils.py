from unittest import TestCase
import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse
from numpy.testing import assert_almost_equal

import skfem.utils as utils
from skfem.assembly import CellBasis, Basis, LinearForm, asm, BilinearForm
from skfem.element import ElementTriP1, ElementQuad1, ElementTriP2
from skfem.mesh import MeshTri, MeshQuad
from skfem.utils import projection, enforce, condense, solve, mpc
from skfem.models import laplace, mass, unit_load


class InitializeScalarField(TestCase):

    def runTest(self):

        mesh = MeshTri().refined(5)
        basis = CellBasis(mesh, ElementTriP1())

        def fun(X):
            x, y = X
            return x ** 2 + y ** 2

        x = projection(fun, basis)
        y = fun(mesh.p)

        normest = np.linalg.norm(x - y)

        self.assertTrue(normest < 0.011,
                        msg="|x-y| = {}".format(normest))


class TestEnforce(TestCase):

    mesh = MeshTri()

    def runTest(self):

        m = self.mesh
        e = ElementTriP1()
        basis = CellBasis(m, e)

        A = laplace.assemble(basis)
        M = mass.assemble(basis)
        D = m.boundary_nodes()

        assert_almost_equal(enforce(A, D=D).toarray(), np.eye(A.shape[0]))
        assert_almost_equal(enforce(M, D=D, diag=0.).toarray(),
                            np.zeros(M.shape))

        enforce(A, D=D, overwrite=True)
        assert_almost_equal(A.toarray(), np.eye(A.shape[0]))


def test_simple_cg_solver():

    m = MeshTri().refined(3)
    basis = CellBasis(m, ElementTriP1())

    A0 = laplace.coo_data(basis)
    A1 = laplace.assemble(basis)

    f = unit_load.assemble(basis)

    D = m.boundary_nodes()

    x1 = solve(*condense(A1, f, D=D))

    f[D] = 0

    x0 = A0.solve(f, D=D)

    assert_almost_equal(x0, x1)


@pytest.mark.parametrize("half_precision", [False, True])
@pytest.mark.parametrize("budget,maxiter,restart,expected_status", [
    (5., 100, 4, "converged"),
    (-1., 100, 4, "timeout"),
    (5., 1, 1, "maxiter"),
])
def test_gmres_cpu_benchmark(monkeypatch, capsys, half_precision, budget,
                             maxiter, restart, expected_status):
    monkeypatch.setattr(utils, "HALF_PRECISION", half_precision)
    monkeypatch.setenv("GMRES_BLOCK_SIZE", "3")
    monkeypatch.setenv("GMRES_POLY_DEGREE", "2")
    matrix = scipy.sparse.diags(
        [-np.ones(11), 4. * np.ones(12), .5 * np.ones(11)],
        [-1, 0, 1], format="csr")
    rhs = np.arange(1., 13.)
    reference = utils.spl.spsolve(matrix, rhs)
    solver = utils.spl.gmres
    calls = []

    def tracked_solver(matrix_arg, rhs_arg, **kwargs):
        calls.append((matrix_arg.dtype, rhs_arg.dtype, kwargs))
        return solver(matrix_arg, rhs_arg, **kwargs)

    monkeypatch.setattr(utils.spl, "gmres", tracked_solver)
    utils._run_gmres_cpu_cases(matrix, rhs, reference, 1e-5, maxiter,
                               restart, budget, set())
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == len(calls) == 4
    dtype = np.float32 if half_precision else np.float64
    for matrix_dtype, rhs_dtype, kwargs in calls:
        assert matrix_dtype == rhs_dtype == dtype
        assert kwargs["maxiter"] == (maxiter + restart - 1) // restart
        assert kwargs["callback_type"] == "x"
    for case, line in zip(utils._GMRES_CPU_BENCH_CASES, lines):
        assert f"case=cpu_{case} status={expected_status}" in line
        fields = dict(token.split("=", 1) for token in line.split()[1:])
        assert float(fields["readback_s"]) == 0.
        assert float(fields["total_s"]) >= 0.
        if expected_status == "converged":
            assert float(fields["relres"]) < 2e-5


def test_gmres_cpu_benchmark_disabled_and_error(monkeypatch, capsys):
    matrix = scipy.sparse.diags([2., 3., 4.], format="csr")
    rhs = np.ones(3)
    reference = utils.spl.spsolve(matrix, rhs)
    make_preconditioner = utils._make_gmres_preconditioner_cpu

    def fail_jacobi(case, matrix_arg):
        if case == "jacobi":
            raise RuntimeError("preconditioner failed")
        return make_preconditioner(case, matrix_arg)

    monkeypatch.setattr(utils, "_make_gmres_preconditioner_cpu", fail_jacobi)
    utils._run_gmres_cpu_cases(matrix, rhs, reference, 1e-5, 100, None,
                               5., {"cpu_none"})
    output = capsys.readouterr().out
    assert "case=cpu_none status=skipped" in output
    assert "case=cpu_jacobi status=error error=RuntimeError" in output
    assert "case=cpu_block_jacobi status=converged" in output
    assert "case=cpu_polynomial status=converged" in output


def test_gmres_ilu0_zero_fill(monkeypatch):
    matrix = SimpleNamespace(shape=(3, 3))
    calls = []
    factor = SimpleNamespace(solve=lambda rhs: rhs)

    def factorize(matrix_arg, **kwargs):
        calls.append((matrix_arg, kwargs))
        return factor

    monkeypatch.setattr(utils, "gpu_spilu", factorize)
    monkeypatch.setattr(utils, "LinearOperator", lambda shape, matvec: (shape, matvec))
    preconditioner = utils._make_gmres_preconditioner("ilu0", matrix)
    assert calls == [(matrix, {"fill_factor": 1})]
    assert preconditioner == (matrix.shape, factor.solve)

    def fail_factorization(*args, **kwargs):
        raise RuntimeError("zero pivot")

    monkeypatch.setattr(utils, "gpu_spilu", fail_factorization)
    with pytest.raises(RuntimeError, match="zero pivot"):
        utils._make_gmres_preconditioner("ilu0", matrix)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("candidate", [0., np.nan])
def test_gmres_benchmark_checks_true_residual(monkeypatch, capsys, device,
                                             candidate):
    matrix = scipy.sparse.eye(3, format="csr")
    rhs = np.ones(3)

    def false_success(matrix_arg, rhs_arg, **kwargs):
        return np.full_like(rhs_arg, candidate), 0

    if device == "cpu":
        monkeypatch.setattr(utils, "_GMRES_CPU_BENCH_CASES", ("none",))
        monkeypatch.setattr(utils.spl, "gmres", false_success)
        utils._run_gmres_cpu_cases(matrix, rhs, rhs, 1e-5, 100, 3, 5., set())
    else:
        stream = SimpleNamespace(synchronize=lambda: None)
        pool = SimpleNamespace(free_all_blocks=lambda: None)
        gpu = SimpleNamespace(
            zeros=np.zeros, asnumpy=np.asarray,
            cuda=SimpleNamespace(Stream=SimpleNamespace(null=stream)),
            get_default_memory_pool=lambda: pool)
        monkeypatch.setattr(utils, "cp", gpu)
        monkeypatch.setattr(utils, "load_to_gpu", lambda matrix_arg, rhs_arg: (matrix_arg, rhs_arg))
        monkeypatch.setattr(utils, "gmres", false_success)
        monkeypatch.setattr(utils, "_GMRES_BENCH_CASES", ("none",))
        monkeypatch.setattr(utils, "_gmres_bench_done", False)
        monkeypatch.setattr(utils, "_gmres_bench_matches_target", lambda: True)
        monkeypatch.setattr(utils, "_gmres_cpu_bench_enabled", lambda: False)
        monkeypatch.setenv("GMRES_PRECOND_BENCHMARK", "1")
        monkeypatch.setenv("GMRES_BENCHMARK_EXIT_AFTER_SOLVE", "1")
        with pytest.raises(SystemExit) as exited:
            utils.run_gmres_preconditioner_benchmark(matrix, rhs, rhs)
        assert exited.value.code == 0
    output = capsys.readouterr().out
    assert "status=residual_mismatch" in output
    assert "converged=0" in output


def test_gmres_target_excludes_auxiliary_solves(monkeypatch, tmp_path):
    monkeypatch.setenv("GMRES_BENCHMARK_TARGET_FILE", __file__)
    observations = []

    def benchmark(*args, **kwargs):
        observations.append(utils._gmres_bench_matches_target())

    monkeypatch.setattr(utils, "run_gmres_preconditioner_benchmark", benchmark)
    matrix = scipy.sparse.eye(3, format="csr")
    rhs = np.ones(3)
    utils.solve(matrix, rhs)
    exec(compile("utils.solve(matrix, rhs)", str(tmp_path / "imported.py"), "exec"),
         {"utils": utils, "matrix": matrix, "rhs": rhs})
    Basis(MeshTri(), ElementTriP1()).project(lambda coordinates: coordinates[0])
    assert observations == [True, False, False]
    monkeypatch.delenv("GMRES_BENCHMARK_TARGET_FILE")
    assert utils._gmres_bench_matches_target()


@pytest.fixture
def gmres_harness():
    path = Path(__file__).resolve().parents[1] / "_findings/gmres_all_examples/benchmark.py"
    spec = importlib.util.spec_from_file_location("gmres_harness_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("as_bytes", [False, True])
def test_gmres_timeout_preserves_partial_output(monkeypatch, gmres_harness,
                                               as_bytes):
    output = ("[gmresbench] ndofs=4 nnz=4 spsolve_cpu_s=0.1 transfer_s=0.01 "
              "tol=1e-5 maxiter=5000 restart=20\n"
              "[gmrescase] case=none status=converged total_s=0.02 converged=1\n")
    error = "partial diagnostic\n"
    if as_bytes:
        output, error = output.encode(), error.encode()

    def timeout(*args, **kwargs):
        assert kwargs["env"]["GMRES_BENCHMARK_TARGET_FILE"].endswith("ex01.py")
        assert kwargs["env"]["GMRES_BENCHMARK_EXIT_AFTER_SOLVE"] == "1"
        raise subprocess.TimeoutExpired("example", 1, output=output, stderr=error)

    monkeypatch.setattr(gmres_harness.subprocess, "run", timeout)
    row, reason, raw = gmres_harness.run_one(1, 1, set())
    assert reason == "timeout"
    assert row["ndofs"] == 4
    assert row["none_status"] == "converged"
    assert "partial diagnostic" in raw


def test_gmres_stops_after_all_cases_fail(monkeypatch, tmp_path, gmres_harness):
    monkeypatch.delenv("EXAMPLES", raising=False)
    monkeypatch.setattr(gmres_harness, "gmres_compatible_examples", lambda: [1])
    monkeypatch.setattr(gmres_harness, "MAX_LEVEL", 10)
    monkeypatch.setattr(gmres_harness, "RESULTS_CSV", tmp_path / "results.csv")
    monkeypatch.setattr(gmres_harness, "STATUS_CSV", tmp_path / "status.csv")
    monkeypatch.setattr(gmres_harness, "RUN_LOGS_DIR", tmp_path / "logs")
    calls = []

    def fail_cases(example, level, disabled):
        calls.append(level)
        row = {"example": example, "refined": level, "ndofs": 4}
        row.update({f"{case}_status": "error" if case == "ilu0" else "maxiter"
                    for case in gmres_harness.CASES})
        row["none_status"] = "residual_mismatch"
        return row, "", ""

    monkeypatch.setattr(gmres_harness, "run_one", fail_cases)
    gmres_harness.main()
    assert calls == [1]
    assert "all_cases_disabled" in (tmp_path / "status.csv").read_text()


def test_mpc_periodic():

    m = MeshQuad().refined(3).with_defaults()
    basis = Basis(m, ElementQuad1())
    A = laplace.assemble(basis)
    b = unit_load.assemble(basis)
    y = solve(*mpc(A, b,
                   S=np.concatenate((
                       m.nodes_satisfying(lambda x: (x[0] == 0) * (x[1] < 1) * (x[1] > 0)),
                       basis.get_dofs({'top', 'bottom'}),
                   )),
                   M=m.nodes_satisfying(lambda x: (x[0] == 1) * (x[1] < 1) * (x[1] > 0))))

    assert_almost_equal(y[basis.get_dofs('left')], y[basis.get_dofs('right')])
    assert_almost_equal(y[basis.get_dofs('left')], y[basis.get_dofs(lambda x: x[0] == .5)])


def test_mpc_2x_periodic():

    m = MeshTri.init_sqsymmetric().refined(4).with_defaults()
    e = ElementTriP2()

    basis = Basis(m, e)

    left = basis.get_dofs(facets='left').sort()[1:-1]
    right = basis.get_dofs(facets='right').sort()[1:-1]
    dirichlet = basis.get_dofs(facets={'top', 'bottom'})

    A = asm(laplace, basis)
    f = asm(LinearForm(lambda v, w: 1. * v), basis)

    # doubly periodic
    M = left
    S = np.concatenate((right, dirichlet))
    T = 2. * scipy.sparse.eye(len(S), len(M)).tocsr()

    y = solve(*mpc(A, f, M=M, S=S, T=T))

    assert_almost_equal(2 * y[basis.get_dofs('left')], y[basis.get_dofs('right')])



def test_mpc_doubly_periodic():

    m = MeshTri.init_sqsymmetric().refined(5).with_defaults()
    e = ElementTriP1()

    basis = Basis(m, e)

    left = basis.get_dofs(facets='left').sort()
    right = basis.get_dofs(facets='right').sort()
    top = basis.get_dofs(facets='top').sort()
    bottom = basis.get_dofs(facets='bottom').sort()

    topleft = basis.get_dofs(nodes=(0, 1))
    others = basis.get_dofs(nodes=[(1, 1), (1, 0), (0, 0)])
    A = asm(laplace, basis)
    f = asm(LinearForm(lambda v, w: (np.sin(2. * np.pi * w.x[0])
                                     + np.sin(2. * np.pi * w.x[1])) * v), basis)

    # doubly periodic
    M = np.concatenate((
        left[1:-1],
        top[1:-1],
        topleft,
    ))
    S = np.concatenate((
        right[1:-1],
        bottom[1:-1],
        others
    ))
    T = scipy.sparse.eye(len(S), len(M)).tocsr()
    T[-1, -1] = 1
    T[-2, -1] = 1
    T[-3, -1] = 1
    # periodic x2
    # M = right
    # S = np.concatenate((left, top, bottom))
    # T = 2 * scipy.sparse.eye(len(S), len(M))

    y = solve(*mpc(A, f, M=M, S=S, T=T))

    assert_almost_equal(y[basis.get_dofs('left')], y[basis.get_dofs('right')])
    assert_almost_equal(y[basis.get_dofs('bottom')], y[basis.get_dofs('top')])


def test_mpc_dirichlet():

    m = MeshTri().refined(3)
    basis = Basis(m, ElementTriP1())
    A = laplace.assemble(basis)
    b = unit_load.assemble(basis)
    D = basis.get_dofs()
    # these two should be equal
    y1 = solve(*mpc(A, b, S=D))
    y2 = solve(*condense(A, b, D=D))
    assert_almost_equal(y1, y2)
    # inhomogeneous
    y1 = solve(*mpc(A, b, S=D, g=m.p[0, D]))
    y2 = solve(*condense(A, b, D=D, x=m.p[0]))
    assert_almost_equal(y1, y2)

