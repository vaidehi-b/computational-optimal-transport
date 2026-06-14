"""
Part 4: Gaussian OT. Bures-Wasserstein closed form + discrete convergence.

    W_2^2(N(mu1,S1), N(mu2,S2)) = ||mu1-mu2||^2 + tr(S1 + S2 - 2 (S1^{1/2} S2 S1^{1/2})^{1/2}).
"""
import time
import json, os
import numpy as np
from scipy.linalg import sqrtm
from scipy.optimize import linprog

from ot_utils import (sample_gaussian_clouds, cost_matrix, uniform_weights,
                      build_marginal_constraint_matrix)


def bures_wasserstein2(mu1, mu2, S1, S2):
    sq_S1 = sqrtm(S1)
    inner = sq_S1 @ S2 @ sq_S1
    inner_sqrt = sqrtm(inner)
    # Discard tiny imaginary parts from numerical sqrtm
    inner_sqrt = np.real_if_close(inner_sqrt, tol=1e6)
    mean_term = float(np.sum((mu1 - mu2) ** 2))
    cov_term = float(np.trace(S1 + S2 - 2.0 * inner_sqrt))
    return mean_term + cov_term


def solve_discrete_ot_lp(C, a, b):
    m, n = C.shape
    A_eq = build_marginal_constraint_matrix(m, n)
    rhs = np.concatenate([a, b])
    res = linprog(C.reshape(-1), A_eq=A_eq, b_eq=rhs,
                  bounds=[(0, None)] * (m * n), method="highs-ipm")
    return res.x.reshape(m, n), res.fun


def run_part4():
    print("=" * 70)
    print("Part 4: Gaussian OT — Bures-Wasserstein convergence")
    print("=" * 70)

    mu1 = np.array([0.0, 0.0])
    mu2 = np.array([3.0, 2.0])
    S1  = np.array([[1.0, 0.6], [0.6, 1.0]])
    S2  = np.array([[1.5, -0.7], [-0.7, 1.2]])

    w2sq_closed = bures_wasserstein2(mu1, mu2, S1, S2)
    print(f"Closed-form W_2^2 = {w2sq_closed:.6f}")

    sizes = [50, 100, 200, 500, 1000]
    seeds = list(range(15))
    convergence = []
    for n in sizes:
        costs = []
        for seed in seeds:
            X, Y, *_ = sample_gaussian_clouds(n, seed=seed,
                                              mu1=mu1, mu2=mu2,
                                              Sigma1=S1, Sigma2=S2)
            C = cost_matrix(X, Y)
            a, b = uniform_weights(n, n)
            _, obj = solve_discrete_ot_lp(C, a, b)
            costs.append(obj)
        costs = np.array(costs)
        mean = float(costs.mean())
        std = float(costs.std(ddof=1))
        convergence.append({"n": n, "mean": mean, "std": std,
                            "all": costs.tolist()})
        print(f"  n={n:>4}  discrete cost = {mean:.4f} ± {std:.4f}  "
              f"(gap to W_2^2 = {mean - w2sq_closed:+.4f})")

    # save
    os.makedirs("results", exist_ok=True)
    out = {"w2sq_closed": w2sq_closed,
           "mu1": mu1.tolist(), "mu2": mu2.tolist(),
           "S1": S1.tolist(), "S2": S2.tolist(),
           "convergence": convergence, "seeds": seeds}
    with open("results/part4.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved results/part4.json")
    return out


if __name__ == "__main__":
    run_part4()
