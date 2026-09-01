import numpy as np
import random
import warnings

warnings.filterwarnings('ignore')


class Importance_Sampling:

    def __init__(self, w0, eta=3.0, lam=1e-4, maxIter=30, maxTime=None, tol=None):
        self.w = w0
        self.eta = eta
        self.lam = lam
        self.maxIter = maxIter
        self.maxTime = maxTime
        self.tol = tol

        self.obj = []
        self.iter = []

    def HT(self, w, sparsity):
        if sparsity is None or sparsity >= np.asarray(w).size:
            return np.asarray(w, dtype=float).copy()
        if sparsity <= 0:
            return np.zeros_like(w, dtype=float)

        flat_w = np.asarray(w, dtype=float).ravel()
        abs_flat_w = np.abs(flat_w)
        threshold = np.partition(abs_flat_w, -sparsity)[-sparsity]
        greater_idx = np.flatnonzero(abs_flat_w > threshold)
        tie_idx = np.flatnonzero(abs_flat_w == threshold)
        top_k_idx = np.concatenate((greater_idx, tie_idx[:sparsity - greater_idx.size]))
        out = np.zeros_like(flat_w)
        out[top_k_idx] = flat_w[top_k_idx]
        return out.reshape(np.shape(w))

    def fit(self, X_train, y_train, batch_size=None, L=None, eval=True, sparsity=None, m=None, pr=False):
        """
        Parameters:
            X_train    : training data matrix
            y_train    : training targets
            batch_size : mini-batch size
            L          : global Lipschitz constant used for the step size
            eval       : whether to record objective values
            sparsity   : target number of nonzero entries
            m          : full-gradient refresh parameter: P(full_grad) = 1/m
            pr         : whether to print progress information
        """

        n, d = X_train.shape

        self.w = self.HT(self.w, sparsity)
        wp = self.w.copy()

        L_sample = self.compute_sample_L(X_train)

        p_sample = L_sample / np.sum(L_sample)
        self.p_sample = p_sample

        v = self.grad_loss(self.w, X_train, y_train) / n + self.lam * self.w

        self.obj = []
        self.iter = [0]

        if eval:
            f_val = self.obj_func(self.w, X_train, y_train)
            self.obj.append(f_val + self.obj_reg(self.w))
            if pr:
                print("obj:", self.obj[-1])

        consecutive_count = 0

        for iter in range(self.maxIter):
            if iter > 0 and random.sample(range(m), 1)[0] != 0:
                idx = np.random.choice(n, size=batch_size, p=p_sample)

                Xb = X_train[idx]
                yb = y_train[idx]

                grad_w = 2.0 * (Xb.dot(self.w) - yb)[:, None] * Xb
                grad_wp = 2.0 * (Xb.dot(wp) - yb)[:, None] * Xb

                scale = 1.0 / (n * p_sample[idx])

                v = v + np.mean((grad_w - grad_wp) * scale[:, None], axis=0) + self.lam * (self.w - wp)

            else:
                v = self.grad_loss(self.w, X_train, y_train) / n + self.lam * self.w

            wp = self.w.copy()
            self.w = self.HT(self.w - v / (self.eta * L), sparsity)

            step_err = np.linalg.norm(self.w - wp)
            rel_err = step_err / max(1.0, np.linalg.norm(wp))

            if eval:
                self.iter.append(1 + iter)

                f_val = self.obj_func(self.w, X_train, y_train)
                self.obj.append(f_val + self.obj_reg(self.w))
                if pr:
                    print("iter", iter, "obj:", self.obj[-1], "rel_err:", rel_err)

            if self.iter[-1] > self.maxIter:
                break
            if self.tol is not None:
                supp_curr = set(np.nonzero(self.w.ravel())[0])
                supp_prev = set(np.nonzero(wp.ravel())[0])
                if supp_curr == supp_prev:
                    consecutive_count += 1
                    if consecutive_count >= 5 and rel_err <= self.tol:
                        print('Stopped by convergence at ', iter, '-th iteration in VR_IHT_IS')
                        break
                else:
                    consecutive_count = 0


        return self

    def grad_loss(self, w, X, y):
        """
        Gradient of sum_i (x_i^T w - y_i)^2:
            grad = 2 X^T (Xw - y)
        """
        r = X.dot(w) - y
        return 2.0 * X.T.dot(r)

    def obj_func(self, w, X, y):
        n, d = X.shape
        r = X.dot(w) - y
        return np.dot(r, r) / n

    def obj_reg(self, w):
        return 0.5 * self.lam * np.sum(w * w)

    def compute_sample_L(self, X):
        row_norm_sq = np.sum(X * X, axis=1)
        L = 2.0 * row_norm_sq + self.lam
        return L


        