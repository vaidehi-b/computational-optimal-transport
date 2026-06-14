"""
Part 2: Quadratic-regularized OT.
    min_{P in U(a,b)}  <C,P> + (lambda/2) ||P||_F^2.
Solved via:
    - ADMM   : CVXPY + OSQP (which is itself an ADMM-based QP solver)
    - IPM    : CVXPY + CLARABEL (interior-point conic / QP solver)
"""
import time
import numpy as np
import cvxpy as cp

from ot_utils import (sample_gaussian_clouds, cost_matrix, uniform_weights,
                      marginal_violation)


def solve_qp_cvxpy(C, a, b, lam, solver="CLARABEL", **kwargs):
    """Solve the lambda-quadratic OT QP with the given CVXPY solver."""
    m, n = C.shape
    P = cp.Variable((m, n), nonneg=True)
    obj = cp.Minimize(cp.sum(cp.multiply(C, P)) + (lam / 2.0) * cp.sum_squares(P))
    cons = [cp.sum(P, axis=1) == a, cp.sum(P, axis=0) == b]
    prob = cp.Problem(obj, cons)
    t0 = time.perf_counter()
    prob.solve(solver=solver, **kwargs)
    elapsed = time.perf_counter() - t0
    return P.value, float(prob.value), elapsed


def run_part2():
    import json, os
    print("=" * 70)
    print("Part 2: Quadratic regularization")
    print("=" * 70)

    n = 200
    X, Y, *_ = sample_gaussian_clouds(n, seed=42)
    C = cost_matrix(X, Y)
    a, b = uniform_weights(n, n)
    lam_default = 0.1

    # (a) compare ADMM vs IPM at n=200
    print(f"\n[n={n}, lambda={lam_default}] comparing ADMM vs IPM")
    P_admm, obj_admm, t_admm = solve_qp_cvxpy(C, a, b, lam_default, solver="OSQP",
                                              max_iter=50000, eps_abs=1e-7, eps_rel=1e-7)
    mv_admm = marginal_violation(P_admm, a, b)
    transport_admm = float((C * P_admm).sum())
    print(f"  ADMM (OSQP)   : t={t_admm:.3f}s   obj={obj_admm:.6f}  "
          f"<C,P>={transport_admm:.6f}  marg={mv_admm:.2e}")

    P_ipm, obj_ipm, t_ipm = solve_qp_cvxpy(C, a, b, lam_default, solver="CLARABEL")
    mv_ipm = marginal_violation(P_ipm, a, b)
    transport_ipm = float((C * P_ipm).sum())
    print(f"  IPM (CLARABEL): t={t_ipm:.3f}s   obj={obj_ipm:.6f}  "
          f"<C,P>={transport_ipm:.6f}  marg={mv_ipm:.2e}")

    # (b) regularization path: <C, P*> vs lambda; check convergence to W_LP
    print("\n[Regularization path] lambda in log-spaced [10, 1e-3]")
    lams = np.logspace(np.log10(10.0), np.log10(1e-3), 14)
    path = []
    for lam in lams:
        P, obj, t = solve_qp_cvxpy(C, a, b, lam, solver="CLARABEL")
        trans = float((C * P).sum())
        # count "exactly zero" entries — quadratic should give sparse P
        sparsity = float((P < 1e-8).mean())
        path.append({"lambda": float(lam), "obj": obj, "transport": trans,
                     "time": t, "sparsity": sparsity})
        print(f"  lam={lam:9.4g}  transport=<C,P>={trans:.6f}  sparsity={sparsity:.3f}  "
              f"t={t:.2f}s")

    # ground truth from LP for comparison
    from scipy.optimize import linprog
    from ot_utils import build_marginal_constraint_matrix
    A_eq = build_marginal_constraint_matrix(n, n)
    rhs = np.concatenate([a, b])
    res = linprog(C.reshape(-1), A_eq=A_eq, b_eq=rhs,
                  bounds=[(0, None)] * (n * n), method="highs-ipm")
    w_lp = res.fun
    print(f"  W_LP (ground truth) = {w_lp:.6f}")

    # (c) effect of rho in ADMM (OSQP)
    print("\n[ADMM rho sweep] at lambda=0.1, n=200, eps=1e-6")
    rhos = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    rho_results = []
    for rho in rhos:
        try:
            P, obj, t = solve_qp_cvxpy(C, a, b, lam_default, solver="OSQP",
                                       rho=rho, max_iter=50000,
                                       eps_abs=1e-6, eps_rel=1e-6,
                                       adaptive_rho=False)
            mv = marginal_violation(P, a, b)
            rho_results.append({"rho": rho, "time": t, "obj": obj, "marg": mv})
            print(f"  rho={rho:8.4g}  t={t:.3f}s   obj={obj:.6f}  marg={mv:.2e}")
        except Exception as e:
            print(f"  rho={rho:8.4g}  FAILED: {e}")
            rho_results.append({"rho": rho, "time": None, "obj": None, "marg": None})

    # save
    os.makedirs("results", exist_ok=True)
    out = {
        "comparison": {
            "admm": {"time": t_admm, "obj": obj_admm, "transport": transport_admm,
                     "marg": mv_admm},
            "ipm":  {"time": t_ipm,  "obj": obj_ipm,  "transport": transport_ipm,
                     "marg": mv_ipm}},
        "path": path,
        "w_lp": float(w_lp),
        "rho_sweep": rho_results,
    }
    with open("results/part2.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved results/part2.json")
    return out


if __name__ == "__main__":
    run_part2()
