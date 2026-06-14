# Computational Optimal Transport — EEOR 6616 Project

All code, figures, and the compiled report for the OT project.

## Files

| File | What it does |
|------|-------------|
| `ot_utils.py` | Point-cloud generation, cost matrix, sparse marginal-constraint matrix, marginal-violation measure. |
| `part1_lp.py` | LP via simplex (HiGHS-DS), IPM (HiGHS-IPM), PDLP (OR-Tools), and own NumPy PDHG. Saves `results/part1.json`. |
| `part2_quadratic.py` | Quadratic-regularized OT via OSQP (ADMM) and CLARABEL (IPM); regularization path; ρ sweep. Saves `results/part2.json`. |
| `part3_sinkhorn.py` | Entropic OT via POT Sinkhorn, own log-domain Sinkhorn, and CLARABEL IPM. Saves `results/part3.json`. |
| `part4_gaussian.py` | Bures-Wasserstein closed form vs. discrete sample OT for n ∈ {50, ..., 1000}. Saves `results/part4.json`. |
| `make_figures.py` | Reads `results/*.json` and produces `figures/*.pdf`. |
| `report.tex` / `report.pdf` | LaTeX source and compiled report. |

## Dependencies

```bash
pip install numpy scipy cvxpy ortools pot matplotlib
```

CVXPY will print a benign warning about ortools being newer than what its
adapter expects; we don't use ortools through CVXPY (PDLP is called directly
via `ortools.pdlp.python.pdlp`), so it has no effect.

## Reproducing

```bash
python part1_lp.py        # ~2 min (the n=500 simplex is the slow part)
python part2_quadratic.py # ~5 min (ADMM at ρ=0.01 and ρ=100 are slow)
python part3_sinkhorn.py  # ~5 min (Sinkhorn at λ=0.01 is slow)
python part4_gaussian.py  # ~3 min (n=1000 EMD ~1 s/seed)
python make_figures.py    # <30 s
pdflatex report.tex && pdflatex report.tex
```

## Key numerical results

| Result | Value |
|--------|-------|
| `W_LP` at n=200, seed=42 | 14.2714 |
| Bures-Wasserstein W₂² (closed form) | 13.8575 |
| Discrete OT at n=1000 (mean ± std, 6 seeds) | 13.79 ± 0.29 |
| Quadratic path converges to `W_LP` at | λ ≈ 0.07 |
| Sinkhorn converges to `W_LP` at | λ ≈ 0.01 (slowly) |
| IPM speedup vs simplex at n=500 | ~10× |
| IPM speedup vs ADMM on quadratic at n=200 | ~50× |
