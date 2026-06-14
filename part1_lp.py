"""
Part 1: The Kantorovich LP solved with three paradigms.
  - Simplex   : scipy.optimize.linprog(method='highs-ds')
  - IPM       : scipy.optimize.linprog(method='highs-ipm')
  - PDHG/PDLP : OR-Tools (Google PDLP)
Optional extra credit: own PDHG loop in NumPy.
"""
import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

from ot_utils import (sample_gaussian_clouds, cost_matrix, uniform_weights,
                      build_marginal_constraint_matrix, marginal_violation)


def solve_lp_scipy(C, a, b, method):
    """Solve the OT LP in standard form with scipy.optimize.linprog."""
    m, n = C.shape
    c = C.reshape(-1)
    A_eq = build_marginal_constraint_matrix(m, n)
    rhs = np.concatenate([a, b])
    bounds = [(0, None)] * (m * n)
    t0 = time.perf_counter()
    res = linprog(c, A_eq=A_eq, b_eq=rhs, bounds=bounds, method=method)
    elapsed = time.perf_counter() - t0
    if not res.success:
        raise RuntimeError(f"{method} failed: {res.message}")
    P = res.x.reshape(m, n)
    return P, res.fun, elapsed


def solve_lp_pdlp(C, a, b, tol=1e-8, time_limit_s=300.0):
    """Solve the OT LP via Google PDLP (OR-Tools)."""
    from ortools.pdlp import solve_log_pb2, solvers_pb2
    from ortools.pdlp.python import pdlp

    m, n = C.shape
    A_eq = build_marginal_constraint_matrix(m, n).tocsc()
    rhs = np.concatenate([a, b])
    nvar = m * n
    nrows = A_eq.shape[0]

    qp = pdlp.QuadraticProgram()
    qp.resize_and_initialize(nvar, nrows)
    qp.objective_vector = C.reshape(-1).astype(np.float64)
    qp.objective_offset = 0.0
    qp.variable_lower_bounds = np.zeros(nvar)
    qp.variable_upper_bounds = np.full(nvar, np.inf)
    qp.constraint_lower_bounds = rhs.astype(np.float64)
    qp.constraint_upper_bounds = rhs.astype(np.float64)
    qp.constraint_matrix = A_eq.astype(np.float64)

    params = solvers_pb2.PrimalDualHybridGradientParams()
    opt = params.termination_criteria.simple_optimality_criteria
    opt.eps_optimal_relative = tol
    opt.eps_optimal_absolute = tol
    params.termination_criteria.time_sec_limit = time_limit_s
    params.num_threads = 1
    params.verbosity_level = 0
    params.presolve_options.use_glop = False

    t0 = time.perf_counter()
    result = pdlp.primal_dual_hybrid_gradient(qp, params)
    elapsed = time.perf_counter() - t0
    P = np.asarray(result.primal_solution).reshape(m, n)
    obj = float(C.reshape(-1) @ result.primal_solution)
    return P, obj, elapsed, result


def solve_lp_pdhg_numpy(C, a, b, max_iters=20000, tol=1e-6, tau=None, sigma=None,
                         verbose=False):
    """Own PDHG loop in NumPy for the OT LP saddle-point form.

        min_{P>=0} max_{f,g}  <C,P> + f'(a - P1) + g'(b - P'1)

    Primal-first Chambolle-Pock updates:
        P^{k+1} = max(0,  P^k - tau*(C - f^k 1' - 1 g^k'))
        Pbar     = 2 P^{k+1} - P^k
        f^{k+1} = f^k + sigma * (a - Pbar*1)            <- gradient ASCENT on the saddle
        g^{k+1} = g^k + sigma * (b - Pbar'*1)

    NB: the project handout writes the dual step as +sigma*(Pbar*1 - a), which
    is gradient descent and does not converge from a zero initialization; the
    saddle gradient on (f, g) is (a - P1, b - P'1), so we use that sign.

    For the bilinear form (P,(f,g)) -> f'P1 + g'P'1 the operator norm of K is
    sqrt(m+n); we take tau*sigma * (m+n) <= 1 for stability.
    """
    m, n = C.shape
    if tau is None:
        tau = 0.9 / np.sqrt(m + n)
    if sigma is None:
        sigma = 0.9 / np.sqrt(m + n)
    P = np.zeros_like(C)
    f = np.zeros(m)
    g = np.zeros(n)
    history = []
    t0 = time.perf_counter()
    for k in range(max_iters):
        grad = C - f[:, None] - g[None, :]
        P_new = np.maximum(0.0, P - tau * grad)
        Pbar = 2 * P_new - P
        P = P_new
        f = f + sigma * (a - Pbar.sum(axis=1))
        g = g + sigma * (b - Pbar.sum(axis=0))
        if (k + 1) % 200 == 0:
            mv = marginal_violation(P, a, b)
            obj = float((C * P).sum())
            history.append((k + 1, obj, mv))
            if verbose and (k + 1) % 2000 == 0:
                print(f"  PDHG iter {k+1:>6}  obj={obj:.6f}  marg={mv:.2e}")
            if mv < tol:
                break
    elapsed = time.perf_counter() - t0
    obj = float((C * P).sum())
    return P, obj, elapsed, history


def run_part1():
    """Run all of Part 1, save timing/accuracy table."""
    print("=" * 70)
    print("Part 1: LP — three solver paradigms")
    print("=" * 70)

    # Correctness sanity check on tiny instance
    print("\n[sanity check on n=3]")
    X, Y, *_ = sample_gaussian_clouds(3, seed=0)
    C = cost_matrix(X, Y)
    a, b = uniform_weights(3, 3)
    P_s, obj_s, _ = solve_lp_scipy(C, a, b, "highs-ds")
    P_i, obj_i, _ = solve_lp_scipy(C, a, b, "highs-ipm")
    P_p, obj_p, _, _ = solve_lp_pdlp(C, a, b, tol=1e-9)
    P_h, obj_h, _, _ = solve_lp_pdhg_numpy(C, a, b, max_iters=50000, tol=1e-8)
    print(f"  simplex: {obj_s:.8f}  IPM: {obj_i:.8f}  PDLP: {obj_p:.8f}  own PDHG: {obj_h:.8f}")
    assert abs(obj_s - obj_i) < 1e-6 and abs(obj_s - obj_p) < 1e-5

    # Scaling experiment
    sizes = [50, 100, 200, 500]
    results = []
    for n in sizes:
        print(f"\n[n = {n}]")
        X, Y, *_ = sample_gaussian_clouds(n, seed=42)
        C = cost_matrix(X, Y)
        a, b = uniform_weights(n, n)

        P, obj, t = solve_lp_scipy(C, a, b, "highs-ds")
        mv = marginal_violation(P, a, b)
        print(f"  simplex : t={t:.3f}s   obj={obj:.6f}  marg={mv:.2e}")
        results.append({"n": n, "method": "simplex", "time": t, "obj": obj, "marg": mv})
        obj_ref = obj  # ground truth

        P, obj, t = solve_lp_scipy(C, a, b, "highs-ipm")
        mv = marginal_violation(P, a, b)
        print(f"  IPM     : t={t:.3f}s   obj={obj:.6f}  marg={mv:.2e}")
        results.append({"n": n, "method": "ipm", "time": t, "obj": obj, "marg": mv})

        P, obj, t, _ = solve_lp_pdlp(C, a, b, tol=1e-8)
        mv = marginal_violation(P, a, b)
        gap = (obj - obj_ref) / max(1.0, abs(obj_ref))
        print(f"  PDLP    : t={t:.3f}s   obj={obj:.6f}  marg={mv:.2e}  rel_gap={gap:+.2e}")
        results.append({"n": n, "method": "pdlp", "time": t, "obj": obj, "marg": mv})

        # own PDHG (extra credit) — only at small n for time
        if n <= 200:
            P, obj, t, hist = solve_lp_pdhg_numpy(C, a, b,
                                                  max_iters=80000, tol=5e-5)
            mv = marginal_violation(P, a, b)
            gap = (obj - obj_ref) / max(1.0, abs(obj_ref))
            print(f"  own PDHG: t={t:.3f}s   obj={obj:.6f}  marg={mv:.2e}  rel_gap={gap:+.2e}")
            results.append({"n": n, "method": "own_pdhg", "time": t, "obj": obj, "marg": mv})

    return results


if __name__ == "__main__":
    import json, os
    res = run_part1()
    os.makedirs("results", exist_ok=True)
    with open("results/part1.json", "w") as f:
        json.dump(res, f, indent=2)
    print("\nSaved results/part1.json")
