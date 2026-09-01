"""
Synthetic data generation for sparse linear regression with heterogeneous Lipschitz constants.

Problem formulation:
    F(x) = (1/N) * Σ_{i=1}^N f_i(x)
    f_i(x) = (1/2) * (a_i^T x - y_i)^2

where:
    - x ∈ R^d, subject to ||x||_0 ≤ s
    - ∇f_i(x) = a_i(a_i^T x - y_i), with exact L_i = ||a_i||²
    - log L_i ~ N(0, σ²), where σ controls sample-wise smoothness heterogeneity
    - y_i = a_i^T x_true + ε_i, ε_i ~ N(0,1), so x_true approximates the s-sparse optimizer
"""

import numpy as np


def generate_synthetic_data(N=6000, d=200, s=20, sigma=1.0, seed=None):
    """
    Generate synthetic data for a sparse linear regression problem with heterogeneous smoothness.

    Parameters:
        N     : number of samples
        d     : ambient dimension
        s     : sparsity level of the ground-truth vector
        sigma : standard deviation of log L_i; controls heterogeneity
        seed  : random seed

    Returns:
        alphas  : (N,) sample-wise Lipschitz constants L_i = ||a_i||²
        centers : (N, d) sample rows a_i of the design matrix
        targets : (N,) response values y_i
        x_true  : (d,) known s-sparse ground-truth vector
    """
    rng = np.random.default_rng(seed)

    log_alphas = rng.normal(0.0, sigma, size=N)
    alphas = np.exp(log_alphas)

    centers = rng.normal(0.0, 1.0, size=(N, d))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    centers *= np.sqrt(alphas)[:, None]

    x_true = np.zeros(d)
    support = rng.choice(d, size=s, replace=False)
    x_true[support] = rng.normal(0.0, 1.0, size=s)
    targets = centers @ x_true + rng.normal(0.0, 1.0, size=N)

    return alphas, centers, targets, x_true


def compute_optimal_solution(x_true, s):
    """
    Return the s-sparse reference solution.

    Under the noiseless model, y_i = a_i^T x_true and x_true is s-sparse;
    hence, x_true is a feasible global minimizer with zero objective value.
    """
    return hard_threshold(x_true, s)


def hard_threshold(w, k):
    if k is None or k >= np.asarray(w).size:
        return np.asarray(w, dtype=float).copy()
    if k <= 0:
        return np.zeros_like(w, dtype=float)

    flat_w = np.asarray(w, dtype=float).ravel()
    abs_flat_w = np.abs(flat_w)
    threshold = np.partition(abs_flat_w, -k)[-k]
    greater_idx = np.flatnonzero(abs_flat_w > threshold)
    tie_idx = np.flatnonzero(abs_flat_w == threshold)
    top_k_idx = np.concatenate((greater_idx, tie_idx[:k - greater_idx.size]))
    out = np.zeros_like(flat_w)
    out[top_k_idx] = flat_w[top_k_idx]
    return out.reshape(np.shape(w))


def full_F_and_grad(x, centers, targets):
    residual = centers @ x - targets
    loss = 0.5 * np.mean(residual ** 2)
    grad = centers.T @ residual / centers.shape[0]
    return loss, grad


def batch_grad_diff(x, xp, centers_batch, weights=None):
    """
    Compute a (reweighted) mini-batch gradient difference.
    """
    dx = x - xp
    grad_diff = centers_batch * (centers_batch @ dx)[:, None]
    if weights is not None:
        grad_diff = weights[:, None] * grad_diff
    return np.mean(grad_diff, axis=0)


def compute_global_lipschitz(centers):
    """Compute the global Lipschitz constant L_F of F."""
    gram = centers.T @ centers / centers.shape[0]
    return float(np.linalg.eigvalsh(gram)[-1])


def describe_heterogeneity(alphas):
    """Summarize the distribution of the sample-wise constants L_i."""
    return {
        "min": float(np.min(alphas)),
        "max": float(np.max(alphas)),
        "ratio": float(np.max(alphas) / np.min(alphas)),
        "mean": float(np.mean(alphas)),
        "std": float(np.std(alphas)),
        "median": float(np.median(alphas)),
    }