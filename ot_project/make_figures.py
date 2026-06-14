"""
Generate all figures from results/*.json. Saves to figures/.
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.2,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def load(name):
    with open(f"results/{name}.json") as f:
        return json.load(f)


# =========================================================================
# Figure 1: Part 1 — solver scaling (time vs n) and accuracy
# =========================================================================
def fig_part1():
    res = load("part1")
    methods = sorted(set(r["method"] for r in res))
    fig, ax = plt.subplots(1, 1, figsize=(4.0, 2.8))
    label_map = {"simplex": "Simplex (HiGHS-DS)", "ipm": "IPM (HiGHS-IPM)",
                 "pdlp": "PDLP (OR-Tools)", "own_pdhg": "Own PDHG (NumPy)"}
    color_map = {"simplex": "C0", "ipm": "C1", "pdlp": "C2", "own_pdhg": "C3"}
    marker_map = {"simplex": "o", "ipm": "s", "pdlp": "D", "own_pdhg": "^"}
    for m in ["simplex", "ipm", "pdlp", "own_pdhg"]:
        ns = [r["n"] for r in res if r["method"] == m]
        ts = [r["time"] for r in res if r["method"] == m]
        if not ns: continue
        ax.plot(ns, ts, label=label_map[m], color=color_map[m],
                marker=marker_map[m], markersize=4)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("$n$ (source = target size)")
    ax.set_ylabel("solve time (s)")
    ax.set_title("Part 1: LP solver scaling")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig("figures/part1_scaling.pdf")
    plt.close(fig)


# =========================================================================
# Figure 2: Regularization paths (Parts 2 & 3) and ADMM rho sensitivity
# =========================================================================
def fig_paths():
    p2 = load("part2"); p3 = load("part3")
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.6))

    # 2a: quadratic path
    ax = axes[0]
    lams = [p["lambda"] for p in p2["path"]]
    transp = [p["transport"] for p in p2["path"]]
    ax.semilogx(lams, transp, "o-", color="C0", markersize=3.5, label="quadratic")
    ax.axhline(p2["w_lp"], ls="--", color="k", lw=0.8, label=r"$W_{\rm LP}$")
    ax.invert_xaxis()
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\langle C, P^*_\lambda\rangle$")
    ax.set_title("(a) Quadratic-reg. path")
    ax.legend(loc="lower right")

    # 2b: entropic path
    ax = axes[1]
    lams = [p["lambda"] for p in p3["path"]]
    transp = [p["transport"] for p in p3["path"]]
    ax.semilogx(lams, transp, "s-", color="C1", markersize=3.5, label="entropic")
    ax.axhline(p3["w_lp"], ls="--", color="k", lw=0.8, label=r"$W_{\rm LP}$")
    ax.invert_xaxis()
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\langle C, P^*_\lambda\rangle$")
    ax.set_title("(b) Entropic-reg. path")
    ax.legend(loc="upper right")

    # 2c: ADMM rho sensitivity (time and marg violation)
    ax = axes[2]
    rhos = [r["rho"] for r in p2["rho_sweep"] if r["time"] is not None]
    times = [r["time"] for r in p2["rho_sweep"] if r["time"] is not None]
    margs = [r["marg"] for r in p2["rho_sweep"] if r["time"] is not None]
    ax.loglog(rhos, times, "o-", color="C2", markersize=4, label="time (s)")
    ax.set_xlabel(r"$\rho$ (ADMM penalty)")
    ax.set_ylabel("solve time (s)")
    ax2 = ax.twinx()
    ax2.loglog(rhos, margs, "x--", color="C3", markersize=4, label="marg. violation")
    ax2.set_ylabel("final marg. violation", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax2.grid(False)
    ax.set_title(r"(c) ADMM $\rho$ sensitivity")
    fig.tight_layout()
    fig.savefig("figures/paths_rho.pdf")
    plt.close(fig)


# =========================================================================
# Figure 3: Coupling visualizations — LP / quadratic / Sinkhorn at two lambdas
# =========================================================================
def fig_couplings():
    import numpy as np
    import cvxpy as cp
    import ot as pot
    from scipy.optimize import linprog
    from ot_utils import (sample_gaussian_clouds, cost_matrix, uniform_weights,
                          build_marginal_constraint_matrix)

    n = 80  # small for legible plot
    X, Y, *_ = sample_gaussian_clouds(n, seed=42)
    C = cost_matrix(X, Y)
    a, b = uniform_weights(n, n)

    # LP
    A_eq = build_marginal_constraint_matrix(n, n)
    rhs = np.concatenate([a, b])
    res = linprog(C.reshape(-1), A_eq=A_eq, b_eq=rhs,
                  bounds=[(0, None)] * (n * n), method="highs-ds")
    P_lp = res.x.reshape(n, n)

    # Quadratic at lambda = 0.1
    P = cp.Variable((n, n), nonneg=True)
    obj = cp.Minimize(cp.sum(cp.multiply(C, P)) + 0.05 * cp.sum_squares(P))
    prob = cp.Problem(obj, [cp.sum(P, axis=1) == a, cp.sum(P, axis=0) == b])
    prob.solve(solver="CLARABEL")
    P_quad = P.value

    # Sinkhorn at lambda = 1 (smooth) and lambda = 0.01 (near-LP)
    P_sk_hi = pot.sinkhorn(a, b, C, reg=1.0, numItermax=5000, method="sinkhorn_log")
    P_sk_lo = pot.sinkhorn(a, b, C, reg=0.05, numItermax=20000, method="sinkhorn_log")

    couplings = [(P_lp,      "LP (exact)"),
                 (P_quad,    r"Quadratic, $\lambda=0.05$"),
                 (P_sk_hi,   r"Sinkhorn, $\lambda=1$"),
                 (P_sk_lo,   r"Sinkhorn, $\lambda=0.05$")]
    fig, axes = plt.subplots(1, 4, figsize=(8.4, 2.4), sharex=True, sharey=True)
    for ax, (P, name) in zip(axes, couplings):
        # plot points
        ax.scatter(X[:, 0], X[:, 1], s=14, c="C0", alpha=0.7, edgecolors="white",
                   linewidths=0.4, label="source", zorder=3)
        ax.scatter(Y[:, 0], Y[:, 1], s=14, c="C1", alpha=0.7, edgecolors="white",
                   linewidths=0.4, label="target", zorder=3)
        # plot couplings as lines, alpha proportional to mass
        Pm = P / P.max()
        for i in range(n):
            for j in range(n):
                if Pm[i, j] > 0.05:  # only meaningful edges
                    ax.plot([X[i, 0], Y[j, 0]], [X[i, 1], Y[j, 1]],
                            color="k", alpha=0.3 * Pm[i, j], lw=0.5, zorder=1)
        ax.set_title(name)
        ax.set_aspect("equal")
    axes[0].legend(loc="upper left", framealpha=0.9, fontsize=7)
    fig.suptitle("Part 1-3: coupling structure (lines drawn for $P_{ij}/\\max P \\geq 0.05$, $n=80$)",
                 y=1.03)
    fig.tight_layout()
    fig.savefig("figures/couplings.pdf")
    plt.close(fig)
    return P_lp, P_quad, P_sk_hi, P_sk_lo


# =========================================================================
# Figure 4: Sinkhorn convergence at three lambdas
# =========================================================================
def fig_sinkhorn_convergence():
    p3 = load("part3")
    fig, ax = plt.subplots(1, 1, figsize=(4.0, 2.6))
    for lam_str, hist in p3["convergence"].items():
        iters = [h[0] for h in hist]
        margs = [h[1] for h in hist]
        ax.semilogy(iters, margs, "-", label=f"$\\lambda={lam_str}$")
    ax.set_xlabel("Sinkhorn iteration")
    ax.set_ylabel("marginal violation $\\|P\\mathbf{1}-a\\|_1 + \\|P^\\top\\mathbf{1}-b\\|_1$")
    ax.set_title("Part 3: Sinkhorn convergence ($n=200$)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/sinkhorn_convergence.pdf")
    plt.close(fig)


# =========================================================================
# Figure 5: Gaussian sample-OT convergence + arrows
# =========================================================================
def fig_gaussian():
    import numpy as np
    from scipy.optimize import linprog
    from ot_utils import (sample_gaussian_clouds, cost_matrix, uniform_weights,
                          build_marginal_constraint_matrix)
    from matplotlib.patches import Ellipse

    p4 = load("part4")
    w2sq = p4["w2sq_closed"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.8))

    # 5a: convergence
    ax = axes[0]
    conv = p4["convergence"]
    ns = np.array([r["n"] for r in conv])
    means = np.array([r["mean"] for r in conv])
    stds = np.array([r["std"] for r in conv])
    ax.errorbar(ns, means, yerr=stds, marker="o", capsize=3, label="discrete OT")
    ax.axhline(w2sq, ls="--", color="k", lw=1.0, label=r"$W_2^2$ closed form")
    ax.set_xscale("log")
    ax.set_xlabel("$n$ (sample size of each measure)")
    ax.set_ylabel(r"$\langle C, P^* \rangle$  (mean $\pm$ std over seeds)")
    ax.set_title("(a) Convergence of sample OT to Bures-Wasserstein")
    ax.legend()

    # 5b: arrow plot with covariance ellipses
    mu1 = np.array(p4["mu1"]); mu2 = np.array(p4["mu2"])
    S1 = np.array(p4["S1"]); S2 = np.array(p4["S2"])
    n_arrow = 150
    X, Y, *_ = sample_gaussian_clouds(n_arrow, seed=7, mu1=mu1, mu2=mu2,
                                      Sigma1=S1, Sigma2=S2)
    C = cost_matrix(X, Y); a, b = uniform_weights(n_arrow, n_arrow)
    A_eq = build_marginal_constraint_matrix(n_arrow, n_arrow)
    res = linprog(C.reshape(-1), A_eq=A_eq, b_eq=np.concatenate([a, b]),
                  bounds=[(0, None)] * (n_arrow * n_arrow), method="highs-ipm")
    P = res.x.reshape(n_arrow, n_arrow)

    ax = axes[1]
    ax.scatter(X[:, 0], X[:, 1], s=10, c="C0", alpha=0.55,
               edgecolors="white", linewidths=0.3, label="source", zorder=3)
    ax.scatter(Y[:, 0], Y[:, 1], s=10, c="C1", alpha=0.55,
               edgecolors="white", linewidths=0.3, label="target", zorder=3)
    Pn = P * n_arrow
    for i in range(n_arrow):
        for j in range(n_arrow):
            if Pn[i, j] > 1e-3:
                ax.plot([X[i, 0], Y[j, 0]], [X[i, 1], Y[j, 1]],
                        "k-", alpha=0.18 * Pn[i, j], lw=0.4, zorder=1)
    for mu, S, c in [(mu1, S1, "C0"), (mu2, S2, "C1")]:
        eigvals, eigvecs = np.linalg.eigh(S)
        angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
        e = Ellipse(xy=mu, width=2 * 2.0 * np.sqrt(eigvals[1]),
                    height=2 * 2.0 * np.sqrt(eigvals[0]),
                    angle=angle, facecolor="none", edgecolor=c,
                    linewidth=1.2, linestyle="--")
        ax.add_patch(e)
    ax.set_aspect("equal")
    ax.set_title(f"(b) LP transport, $n={n_arrow}$ (dashed: $2\\sigma$ ellipses)")
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig("figures/gaussian.pdf")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    print("Figure 1: Part 1 scaling")
    fig_part1()
    print("Figure 2: regularization paths and rho")
    fig_paths()
    print("Figure 3: coupling visualizations")
    fig_couplings()
    print("Figure 4: Sinkhorn convergence")
    fig_sinkhorn_convergence()
    print("Figure 5: Gaussian convergence + arrows")
    fig_gaussian()
    print("All figures saved to figures/")
