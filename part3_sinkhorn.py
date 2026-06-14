"""
Part 3: Entropic-regularized OT.
    min_{P in U(a,b)}  <C,P> + lambda * sum_ij P_ij log P_ij.
Solved via:
    - Sinkhorn : POT ot.sinkhorn (and own log-domain implementation for small lambda)
    - IPM      : CVXPY + CLARABEL using cp.entr
"""
import time
import numpy as np
import cvxpy as cp
import ot as pot

from ot_utils import (sample_gaussian_clouds, cost_matrix, uniform_weights,
                      marginal_violation)


def solve_sinkhorn_pot(C, a, b, lam, numItermax=10000, stopThr=1e-9, method="sinkhorn_log"):
    """Sinkhorn via POT. log-domain is numerically stable for small lambda."""
    t0 = time.perf_counter()
    P = pot.sinkhorn(a, b, C, reg=lam, numItermax=numItermax, stopThr=stopThr, method=method)
    elapsed = time.perf_counter() - t0
    obj = float((C * P).sum() + lam * np.sum(P * np.log(np.clip(P, 1e-300, None))))
    return P, obj, elapsed


def solve_sinkhorn_own(C, a, b, lam, max_iters=20000, tol=1e-9, return_history=False):
    """Own log-domain Sinkhorn (alternating row/col rescaling)."""
    m, n = C.shape
    K = -C / lam  # log kernel
    log_a = np.log(a); log_b = np.log(b)
    log_u = np.zeros(m); log_v = np.zeros(n)
    history = []
    t0 = time.perf_counter()
    for k in range(max_iters):
        # u <- a / (K v): in log-domain
        log_u = log_a - logsumexp_rows(K + log_v[None, :])
        log_v = log_b - logsumexp_rows((K + log_u[:, None]).T)
        if return_history and ((k + 1) % 1 == 0):
            logP = K + log_u[:, None] + log_v[None, :]
            P = np.exp(logP)
            mv = marginal_violation(P, a, b)
            history.append((k + 1, mv))
            if mv < tol:
                break
    elapsed = time.perf_counter() - t0
    logP = K + log_u[:, None] + log_v[None, :]
    P = np.exp(logP)
    obj = float((C * P).sum() + lam * np.sum(P * np.log(np.clip(P, 1e-300, None))))
    return P, obj, elapsed, history


def logsumexp_rows(X):
    """log-sum-exp along the second axis."""
    m = X.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(X - m).sum(axis=1, keepdims=True))).ravel()


def solve_ipm_entropic(C, a, b, lam, solver="CLARABEL"):
    """IPM/conic solve of entropic OT via CVXPY."""
    m, n = C.shape
    P = cp.Variable((m, n), nonneg=True)
    # -sum P log P = sum entr(P); objective <C,P> + lambda * sum P log P = <C,P> - lambda * sum entr(P)
    obj = cp.Minimize(cp.sum(cp.multiply(C, P)) - lam * cp.sum(cp.entr(P)))
    cons = [cp.sum(P, axis=1) == a, cp.sum(P, axis=0) == b]
    prob = cp.Problem(obj, cons)
    t0 = time.perf_counter()
    prob.solve(solver=solver)
    elapsed = time.perf_counter() - t0
    return P.value, float(prob.value), elapsed


def run_part3():
    import json, os
    print("=" * 70)
    print("Part 3: Entropic regularization (Sinkhorn)")
    print("=" * 70)

    n = 200
    X, Y, *_ = sample_gaussian_clouds(n, seed=42)
    C = cost_matrix(X, Y)
    a, b = uniform_weights(n, n)

    # Median cost normalisation so lambda values are interpretable
    C_med = np.median(C)
    print(f"[n={n}]  median cost = {C_med:.4f}")

    # (a) run Sinkhorn (POT) + own implementation across lambda
    print("\n[Sinkhorn (POT) vs own log-Sinkhorn]")
    lam_compare = 0.1
    P_pot, obj_pot, t_pot = solve_sinkhorn_pot(C, a, b, lam_compare)
    P_own, obj_own, t_own, _ = solve_sinkhorn_own(C, a, b, lam_compare,
                                                  max_iters=20000, tol=1e-9,
                                                  return_history=True)
    mv_pot = marginal_violation(P_pot, a, b)
    mv_own = marginal_violation(P_own, a, b)
    print(f"  POT log-Sinkhorn : t={t_pot:.3f}s  obj={obj_pot:.6f}  marg={mv_pot:.2e}")
    print(f"  own log-Sinkhorn : t={t_own:.3f}s  obj={obj_own:.6f}  marg={mv_own:.2e}")

    # (b) compare with IPM at small-ish n (n=200 may be too slow for cvxpy + entr;
    #     use smaller n if needed)
    print("\n[IPM via CVXPY + cp.entr]")
    n_ipm = 100  # cp.entr at n=200 is huge for conic solvers; use n=100
    X2, Y2, *_ = sample_gaussian_clouds(n_ipm, seed=42)
    C2 = cost_matrix(X2, Y2)
    a2, b2 = uniform_weights(n_ipm, n_ipm)
    lam_ipm = 0.1
    P_ipm, obj_ipm, t_ipm = solve_ipm_entropic(C2, a2, b2, lam_ipm)
    P_sk_ref, obj_sk_ref, t_sk_ref = solve_sinkhorn_pot(C2, a2, b2, lam_ipm)
    print(f"  n={n_ipm}, lambda={lam_ipm}")
    print(f"  Sinkhorn (POT) : t={t_sk_ref:.3f}s  obj={obj_sk_ref:.6f}")
    print(f"  IPM (CLARABEL) : t={t_ipm:.3f}s   obj={obj_ipm:.6f}")
    print(f"  ||P_sinkhorn - P_ipm||_F = {np.linalg.norm(P_sk_ref - P_ipm):.2e}")

    # (c) convergence: marginal violation vs iteration at n=200 for several lambda
    print("\n[Convergence at n=200 for lambda in {1, 0.1, 0.01}]")
    convergence = {}
    for lam in [1.0, 0.1, 0.01]:
        _, _, _, hist = solve_sinkhorn_own(C, a, b, lam,
                                            max_iters=2000, tol=1e-12,
                                            return_history=True)
        convergence[str(lam)] = hist
        print(f"  lambda={lam}: {len(hist)} iters,  final marg = {hist[-1][1]:.2e}")

    # (d) regularization path: 15 log-spaced lambda in [10, 0.01]
    print("\n[Regularization path]  15 lambdas log-spaced [10, 0.01]")
    lams = np.logspace(np.log10(10.0), np.log10(0.01), 15)
    path = []
    for lam in lams:
        try:
            P, obj, t = solve_sinkhorn_pot(C, a, b, lam, numItermax=20000)
            trans = float((C * P).sum())
            path.append({"lambda": float(lam), "transport": trans, "obj": obj, "time": t})
            print(f"  lam={lam:9.4g}  <C,P>={trans:.6f}  t={t:.2f}s")
        except Exception as e:
            print(f"  lam={lam:9.4g}  FAILED: {e}")

    # ground-truth W_LP
    from scipy.optimize import linprog
    from ot_utils import build_marginal_constraint_matrix
    A_eq = build_marginal_constraint_matrix(n, n)
    rhs = np.concatenate([a, b])
    res = linprog(C.reshape(-1), A_eq=A_eq, b_eq=rhs,
                  bounds=[(0, None)] * (n * n), method="highs-ipm")
    w_lp = res.fun
    print(f"  W_LP (ground truth) = {w_lp:.6f}")

    os.makedirs("results", exist_ok=True)
    out = {
        "sinkhorn_vs_own": {"t_pot": t_pot, "obj_pot": obj_pot, "marg_pot": mv_pot,
                             "t_own": t_own, "obj_own": obj_own, "marg_own": mv_own,
                             "lambda": lam_compare},
        "ipm_comparison": {"n": n_ipm, "lambda": lam_ipm,
                            "t_sinkhorn": t_sk_ref, "obj_sinkhorn": obj_sk_ref,
                            "t_ipm": t_ipm, "obj_ipm": obj_ipm,
                            "fro_diff": float(np.linalg.norm(P_sk_ref - P_ipm))},
        "convergence": convergence,
        "path": path,
        "w_lp": float(w_lp),
    }
    with open("results/part3.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved results/part3.json")
    return out


if __name__ == "__main__":
    run_part3()
