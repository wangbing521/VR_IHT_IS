import numpy as np
import time
import warnings
from scipy import sparse
from scipy.special import softmax, log_softmax

warnings.filterwarnings("ignore")


class Importance_Sampling:
    def __init__(self, w0, eta=3.0, lam=1e-4, maxIter=30, tol=None, c_mat=None, seed=None):
        
        self.w = np.asarray(w0, dtype=float).copy()
        self.eta = eta
        self.lam = lam
        self.c_mat = c_mat
        self.maxIter = maxIter
        self.tol = tol

        self.obj = []
        self.Time = []
        self.test_acc = []
        self.iter = []
        self.sampling_prob = None

        self.rng = np.random.default_rng(seed)

    def HT(self, w, k):
        if k is None or k >= w.size:
            return w.copy()
        if k <= 0:
            return np.zeros_like(w)

        flat_w = w.ravel()
        abs_flat_w = np.abs(flat_w)
        threshold = np.partition(abs_flat_w, -k)[-k]
        greater_idx = np.flatnonzero(abs_flat_w > threshold)
        tie_idx = np.flatnonzero(abs_flat_w == threshold)
        top_k_idx = np.concatenate((greater_idx, tie_idx[:k - greater_idx.size]))
        out = np.zeros_like(flat_w)
        out[top_k_idx] = flat_w[top_k_idx]
        return out.reshape(w.shape)

    def predict(self, X):
        scores = X @ self.w
        return np.argmax(scores, axis=1)

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        y_true = np.argmax(y, axis=1)
        return np.mean(y_pred == y_true)

    def compute_sample_L(self, X):
        """
        Return an array of shape (n,).
        Li ≈ c_norm * ||x_i||^2 + 2*lam
        """
        if self.c_mat is None:
            c_norm = 0.5
        elif np.isscalar(self.c_mat):
            c_norm = float(self.c_mat)
        else:
            c_norm = np.linalg.norm(self.c_mat, ord=2)

        if sparse.issparse(X):
            row_norm2 = np.asarray(X.multiply(X).sum(axis=1)).ravel()
        else:
            row_norm2 = np.sum(X * X, axis=1)

        L_sample = c_norm * row_norm2 + 2.0 * self.lam
        return np.asarray(L_sample, dtype=float).ravel()

    def full_batch_F_and_grad(self, W, X, y):
        """
        Returns:
            loss : scalar
            grad : array of shape (d, C)
        """
        n = X.shape[0]

        scores = X @ W
        probs = softmax(scores, axis=1)

        assert probs.shape == y.shape, f"probs.shape={probs.shape}, y.shape={y.shape}"

        loss_data = -np.sum(y * log_softmax(scores, axis=1)) / n
        loss_reg = self.lam * np.linalg.norm(W, ord="fro") ** 2
        loss = loss_data + loss_reg

        dscores = (probs - y) / n
        grad = X.T @ dscores + 2.0 * self.lam * W

        return loss, grad

    def _sample_indices_from_cdf(self, cdf, batch_size):
        u = self.rng.random(batch_size)
        idx = np.searchsorted(cdf, u, side="left")
        return idx

    def batch_grad_diff_data_fast(self, W, Wp, Xb, scale):
        """
        Compute:
            mean_i [ (1/(n p_i)) * x_i (softmax(x_iW)-softmax(x_iWp))^T ]
        Return an array of shape (d, C).
        """
        if sparse.issparse(Xb):
            scores_w = Xb @ W
            scores_wp = Xb @ Wp
            probs_w = softmax(scores_w, axis=1)
            probs_wp = softmax(scores_wp, axis=1)
            delta = (probs_w - probs_wp) * scale[:, None]
            grad_diff = (Xb.T @ delta) / Xb.shape[0]
        else:
            scores_w = Xb @ W
            scores_wp = Xb @ Wp
            probs_w = softmax(scores_w, axis=1)
            probs_wp = softmax(scores_wp, axis=1)
            delta = (probs_w - probs_wp) * scale[:, None]
            grad_diff = (Xb.T @ delta) / Xb.shape[0]

        return np.asarray(grad_diff, dtype=float)

    def fit(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        batch_size=None,
        L=None,
        eval=True,
        sparsity=None,
        m=None,
        pr=False
    ):
        """
        Parameters:
            X_train    : training data matrix
            y_train    : training targets
            X_test     : test data matrix
            y_test     : test targets
            batch_size : mini-batch size
            L          : global Lipschitz constant used for the step size
            eval       : whether to record objective values and evaluation metrics
            sparsity   : target number of nonzero entries
            m          : full-gradient refresh parameter: P(full_grad) = 1/m
            pr         : whether to print progress information
        """

        n, d = X_train.shape
        y_train = np.asarray(y_train, dtype=float)

        if y_train.ndim != 2:
            raise ValueError("y_train must be a one-hot matrix with shape (n, C).")

        if batch_size is None:
            batch_size = max(1, int(np.sqrt(n)))
        batch_size = int(batch_size)

        if m is None:
            m = max(1, int(np.sqrt(n)))
        m = int(m)

        self.w = self.HT(self.w, sparsity)
        wp = self.w.copy()

        L_sample = self.compute_sample_L(X_train)
        sum_L = np.sum(L_sample)
        if sum_L <= 0:
            p_sample = np.full(n, 1.0 / n, dtype=float)
        else:
            p_sample = L_sample / sum_L
        self.sampling_prob = p_sample

        cdf_sample = np.cumsum(p_sample)
        cdf_sample[-1] = 1.0

        f_val, v = self.full_batch_F_and_grad(self.w, X_train, y_train)

        self.Time = [0.0]
        self.obj = []
        self.test_acc = []
        self.iter = [0]

        if eval:
            acc_val = self.accuracy(X_train, y_train)
            self.obj.append(f_val)
            if pr:
                print("obj:", self.obj[-1])

        start_time = time.time()
        support_stable = 0

        for iter in range(self.maxIter):
            if iter > 0 and self.rng.integers(m) != 0:
                idx = self._sample_indices_from_cdf(cdf_sample, batch_size)

                if sparse.issparse(X_train):
                    Xb = X_train[idx, :]
                else:
                    Xb = X_train[idx]

                scale = 1.0 / (n * p_sample[idx])

                grad_diff_data = self.batch_grad_diff_data_fast(self.w, wp, Xb, scale)
                grad_diff_reg = 2.0 * self.lam * (self.w - wp)

                v = v + grad_diff_data + grad_diff_reg

            else:
                _, v = self.full_batch_F_and_grad(self.w, X_train, y_train)

            wp = self.w.copy()
            self.w = self.HT(self.w - v / (self.eta * L), sparsity)

            step_err = np.linalg.norm(self.w - wp)
            rel_err = step_err / max(1.0, np.linalg.norm(wp))

            if sparsity is not None:
                supp = set(np.nonzero(self.w.ravel())[0])
                supp_old = set(np.nonzero(wp.ravel())[0])
                if supp == supp_old:
                    support_stable += 1
                else:
                    support_stable = 0

            if eval:
                self.Time.append(time.time() - start_time)
                self.iter.append(iter + 1)

                f_val, _ = self.full_batch_F_and_grad(self.w, X_train, y_train)
                self.obj.append(f_val)

                if pr:
                    print("iter", iter + 1, "obj:", self.obj[-1], "rel_err:", rel_err, "stable:", support_stable)

            if self.iter[-1] >= self.maxIter:
                self.test_acc.append(self.accuracy(X_test, y_test))
                print('Accuracy of the test set in VR_IHT_IS', self.test_acc[-1])
                print('Training loss in VR_IHT_IS', self.obj[-1])
                print('Time in VR_IHT_IS', self.Time[-1])
                break

            if self.tol is not None:
                if rel_err <= self.tol and support_stable >= 5:
                    self.test_acc.append(self.accuracy(X_test, y_test))
                    print('Accuracy of the test set in VR_IHT_IS', self.test_acc[-1])
                    print('Stopped by rel_err and support stability at ', iter, '-th iteration')
                    print('Training loss in VR_IHT_IS', self.obj[-1])
                    print('Time in VR_IHT_IS', self.Time[-1])
                    break


        return self
