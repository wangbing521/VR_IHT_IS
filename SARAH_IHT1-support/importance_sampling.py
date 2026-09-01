import numpy as np
import time


class Importance_Sampling:
    """SARAH-IHT with Importance Sampling and support identification tracking."""

    def __init__(self, eta=3.0, lam=1e-3, maxIter=500, tol=1e-8, seed=None):
       
        self.eta = eta
        self.lam = lam
        self.maxIter = maxIter
        self.tol = tol
        self.rng = np.random.default_rng(seed)

        self.x = None

        self.obj = []
        self.Time = []
        self.iter_log = []
        self.converged = False

        self.N_switch = []
        self.M_nat = []
        self.gamma_hat = []
        self.T_stab = None

        self._supp_prev = None
        self._n_switch_acc = 0

    def HT(self, w, k):
        """Keep the k largest absolute entries, resolving ties by a single threshold value."""
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

    def full_grad(self, x, A, b):
        """Full gradient: ∇F(x) = (1/N) A^T(Ax - b) + λ x"""
        N = A.shape[0]
        residual = A @ x - b
        grad_data = A.T @ residual / N
        return grad_data + self.lam * x

    def full_obj(self, x, A, b):
        """F(x) = (1/(2N)) ||Ax - b||^2 + (λ/2) ||x||^2"""
        N = A.shape[0]
        residual = A @ x - b
        return 0.5 * np.sum(residual ** 2) / N + 0.5 * self.lam * np.sum(x ** 2)

    def compute_sample_L(self, A):
        """L_i ≈ ||a_i||^2 + λ  (Hessian of f_i is a_i a_i^T + λ I)"""
        row_norm2 = np.sum(A * A, axis=1)
        return row_norm2 + self.lam

    def _sample_indices_from_cdf(self, cdf, batch_size):
        u = self.rng.random(batch_size)
        return np.searchsorted(cdf, u, side="left")

    def _get_support(self, x):
        """Return sorted tuple of indices where x is non-zero."""
        return tuple(np.sort(np.nonzero(x)[0]))

    def _symmetric_diff_size(self, supp, S_nat):
        """|supp Δ S_nat|"""
        set_a = set(supp)
        set_b = set(S_nat)
        return len(set_a.symmetric_difference(set_b))

    def _compute_gamma_hat(self, v, s):
        """γ̂_k = |v|_{(s)} - |v|_{(s+1)}"""
        abs_v = np.abs(v)
        abs_v.sort()
        if s >= len(abs_v):
            return abs_v[-1]
        return abs_v[-s] - abs_v[-(s + 1)]

    def fit(self, A, b, S_nat, sparsity, batch_size=None, m=None, L=None, x0=None, pr=False):
        """
        Parameters
        ----------
        A, b        : data matrix (N, d) and target vector (N,)
        S_nat       : tuple / list of true support indices
        sparsity s  : target sparsity level
        batch_size  : mini-batch size; default sqrt(N)
        m           : inner loop length for SARAH; default 2*sqrt(N)
        L           : Lipschitz constant; if None, computed from A
        x0          : initial point; if None, use zeros
        pr          : if True, print progress
        """
        N, d = A.shape
        s = sparsity

        if batch_size is None:
            batch_size = max(1, int(np.sqrt(N)))
        batch_size = int(batch_size)

        if m is None:
            m = max(1, 2 * int(np.sqrt(N)))
        m = int(m)

        if L is None:
            L = np.linalg.norm(A.T @ A / N + self.lam * np.eye(d), ord=2)

        if x0 is not None:
            self.x = np.asarray(x0, dtype=float).copy()
        else:
            self.x = np.zeros(d)
        xp = self.x.copy()

        L_samples = self.compute_sample_L(A)
        sum_L = np.sum(L_samples)
        p_sample = L_samples / sum_L if sum_L > 0 else np.full(N, 1.0 / N)
        cdf_sample = np.cumsum(p_sample)
        cdf_sample[-1] = 1.0

        v = self.full_grad(self.x, A, b)

        S_cur = self._get_support(self.x)
        self._supp_prev = S_cur
        self._n_switch_acc = 0

        f0 = self.full_obj(self.x, A, b)
        self.obj = [f0]
        self.Time = [0.0]
        self.iter_log = [0]
        self.N_switch = [0]
        self.M_nat = [self._symmetric_diff_size(S_cur, S_nat)]

        v_before = self.x - v / (self.eta * L)
        self.gamma_hat = [self._compute_gamma_hat(v_before, s)]

        start_time = time.time()
        support_stable_count = 0

        for it in range(self.maxIter):
            if it > 0 and self.rng.integers(m) != 0:
                idx = self._sample_indices_from_cdf(cdf_sample, batch_size)
                A_b = A[idx]
                b_b = b[idx]

                residual_w = A_b @ self.x - b_b
                residual_xp = A_b @ xp - b_b
                weight = 1.0 / (N * p_sample[idx])

                grad_data_w = A_b.T @ (residual_w * weight) / batch_size
                grad_data_xp = A_b.T @ (residual_xp * weight) / batch_size

                v = v + (grad_data_w - grad_data_xp) + self.lam * (self.x - xp)

            else:
                v = self.full_grad(self.x, A, b)

            v_before_ht = self.x - v / (self.eta * L)

            gamma_k = self._compute_gamma_hat(v_before_ht, s)

            xp = self.x.copy()
            self.x = self.HT(v_before_ht, s)

            S_cur = self._get_support(self.x)
            support_changed = (S_cur != self._supp_prev)
            if support_changed:
                self._n_switch_acc += 1
            self._supp_prev = S_cur
            M_nat_k = self._symmetric_diff_size(S_cur, S_nat)

            self.N_switch.append(self._n_switch_acc)
            self.M_nat.append(M_nat_k)
            self.gamma_hat.append(gamma_k)

            self.Time.append(time.time() - start_time)
            self.iter_log.append(it + 1)
            f_val = self.full_obj(self.x, A, b)
            self.obj.append(f_val)

            if pr and (it + 1) % 50 == 0:
                print(f"  iter {it+1:4d}  obj={f_val:.6e}  "
                      f"N_switch={self._n_switch_acc:4d}  M_nat={M_nat_k:3d}  "
                      f"gamma={gamma_k:.6e}")

            rel_change = np.linalg.norm(self.x - xp) / max(1.0, np.linalg.norm(xp))
            supp_same = self._get_support(self.x) == self._get_support(xp)
            if supp_same:
                support_stable_count += 1
            else:
                support_stable_count = 0

            if support_stable_count >= 5 and rel_change <= self.tol:
                self.converged = True
                if pr:
                    print(f"  Converged at iter {it+1}: rel_change={rel_change:.2e} <= {self.tol}")
                break

        self.T_stab = self._compute_T_stab()

        if pr:
            print(f"  T_stab = {self.T_stab}")

        return self

    def _compute_T_stab(self):
        """T_stab = first iteration after which support never changes."""
        N_switch = np.array(self.N_switch)
        diffs = np.diff(N_switch, prepend=N_switch[0])
        change_indices = np.where(diffs[1:] == 1)[0]
        if len(change_indices) == 0:
            return 0
        return int(change_indices[-1]) + 1
