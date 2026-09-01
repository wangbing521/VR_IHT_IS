import numpy as np
import time
from heterogeneous import (
    batch_grad_diff,
    compute_global_lipschitz,
    compute_optimal_solution,
    full_F_and_grad,
    hard_threshold,
)


class Importance_Sampling:
    """SARAH variance reduction with hard thresholding and importance sampling."""

    def __init__(self, x0=None, eta=4.0, max_iter=300, tol=None):
    
        self.x = x0
        self.eta = eta
        self.max_iter = max_iter
        self.tol = tol
        self.obj = []
        self.ngrad = []
        self.Time = []
        self.dist_to_opt = []
        self.f_opt = None
        self.x_opt = None

    def HT(self, w, k):
        return hard_threshold(w, k)

    def fit(self, centers, alphas, targets, x_true, s, batch_size=70, m=None, seed=None, verbose=False):
        """
            Parameters:
            centers, alphas : sample rows a_i and corresponding Lipschitz constants L_i
            targets         : response values y_i
            x_true          : ground-truth sparse vector used for data generation
            s               : sparsity level
            batch_size      : mini-batch size
            m               : full-gradient refresh parameter: P(full_grad) = 1/m
            seed            : random seed
            verbose         : whether to print progress
        """
        rng = np.random.default_rng(seed)
        N, d = centers.shape
        if m is None:
            m = max(1, 2 * batch_size)

        L_avg = compute_global_lipschitz(centers)
        step_size = 1.0 / (self.eta * L_avg)

        sum_alpha = np.sum(alphas)
        p_sample = alphas / sum_alpha
        is_weights_full = 1.0 / (N * p_sample)

        if self.x is None:
            self.x = np.zeros(d)
        self.x = self.HT(self.x, s)
        xp = self.x.copy()

        self.x_opt = compute_optimal_solution(x_true, s)
        self.f_opt, _ = full_F_and_grad(self.x_opt, centers, targets)

        f_val, v = full_F_and_grad(self.x, centers, targets)

        self.obj = [f_val]
        self.ngrad = [0.0]
        self.Time = [0.0]
        self.dist_to_opt = [np.linalg.norm(self.x - self.x_opt)]

        start_time = time.time()
        support_stable = 0
        full_batch_equiv = 0.0

        for it in range(self.max_iter):
            if it > 0 and rng.integers(m) != 0:
                idx = rng.choice(N, size=batch_size, replace=True, p=p_sample)
                Xb = centers[idx]
                w_b = is_weights_full[idx]

                grad_diff = batch_grad_diff(self.x, xp, Xb, weights=w_b)
                v = v + grad_diff
                full_batch_equiv += batch_size / N
            else:
                _, v = full_F_and_grad(self.x, centers, targets)
                full_batch_equiv += 1.0

            xp = self.x.copy()
            self.x = self.HT(self.x - step_size * v, s)

            step_err = np.linalg.norm(self.x - xp)
            rel_err = step_err / max(1.0, np.linalg.norm(xp))

            supp = set(np.nonzero(self.x.ravel())[0])
            supp_old = set(np.nonzero(xp.ravel())[0])
            if supp == supp_old:
                support_stable += 1
            else:
                support_stable = 0

            self.Time.append(time.time() - start_time)
            self.ngrad.append(self.ngrad[-1] + full_batch_equiv)
            full_batch_equiv = 0.0

            f_val, _ = full_F_and_grad(self.x, centers, targets)
            self.obj.append(f_val)
            self.dist_to_opt.append(np.linalg.norm(self.x - self.x_opt))

            if verbose and it % 20 == 0:
                print(f"  VR-IHT-IS iter {it:4d}: F={f_val:.6e}, "
                      f"dist={self.dist_to_opt[-1]:.6e}")

            if self.tol is not None and rel_err <= self.tol and support_stable >= 5:
                if verbose:
                    print(f"  VR-IHT-IS converged at iter {it} "
                          f"(rel_err={rel_err:.2e}, stable={support_stable})")
                break
        return self

    def get_results(self):
        """Return a dictionary compatible with the original interface."""
        return {
            "obj": np.array(self.obj),
            "ngrad": np.array(self.ngrad),
            "time": np.array(self.Time),
            "dist_to_opt": np.array(self.dist_to_opt),
            "f_opt": float(self.f_opt) if self.f_opt is not None else None,
        }