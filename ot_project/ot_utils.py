"""
Shared utilities for computational OT project.
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import vstack, csr_matrix, eye as speye, kron


def sample_gaussian_clouds(n, seed=0, mu1=None, mu2=None, Sigma1=None, Sigma2=None):
    """Sample n source and n target points from two 2D Gaussians with non-diagonal cov."""
    rng = np.random.default_rng(seed)
    if mu1 is None:
        mu1 = np.array([0.0, 0.0])
    if mu2 is None:
        mu2 = np.array([3.0, 2.0])
    if Sigma1 is None:
        Sigma1 = np.array([[1.0, 0.6], [0.6, 1.0]])
    if Sigma2 is None:
        Sigma2 = np.array([[1.5, -0.7], [-0.7, 1.2]])
    X = rng.multivariate_normal(mu1, Sigma1, size=n)
    Y = rng.multivariate_normal(mu2, Sigma2, size=n)
    return X, Y, mu1, mu2, Sigma1, Sigma2


def cost_matrix(X, Y):
    """Squared Euclidean cost matrix C_ij = ||x_i - y_j||^2."""
    return cdist(X, Y, "sqeuclidean")


def uniform_weights(m, n):
    a = np.ones(m) / m
    b = np.ones(n) / n
    return a, b


def build_marginal_constraint_matrix(m, n):
    """Build sparse A_eq for standard-form LP: vec(P) row-major (i*n + j).
    Row constraints: for each i, sum_j P_ij = a_i  (m rows)
    Col constraints: for each j, sum_i P_ij = b_j  (n rows)
    A_eq has shape (m+n, m*n).
    """
    # Row constraints: kron(I_m, 1_n^T)
    A_row = kron(speye(m, format="csr"), csr_matrix(np.ones((1, n))))
    # Col constraints: kron(1_m^T, I_n)
    A_col = kron(csr_matrix(np.ones((1, m))), speye(n, format="csr"))
    A = vstack([A_row, A_col], format="csr")
    return A


def marginal_violation(P, a, b):
    """L1 marginal violation."""
    return np.abs(P.sum(axis=1) - a).sum() + np.abs(P.sum(axis=0) - b).sum()
