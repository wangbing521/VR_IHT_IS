from sklearn.metrics import accuracy_score
import numpy as np
import time
from scipy import sparse
from sklearn.utils import shuffle
from libsvm.svmutil import svm_read_problem
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
import random
import warnings

warnings.filterwarnings('ignore')


class Importance_Sampling:
    def __init__(self, w0, eta=4, lam=1e-4, maxIter=None, maxTime=None, tol=None):
       
        self.w = w0
        self.eta = eta
        self.lam = lam
        self.maxIter = maxIter
        self.maxTime = maxTime
        self.tol = tol

        self.Time = []
        self.obj = []
        self.acc_test = []
        self.iter = []
        self.importance_sampling_prob = None
        self.uniform_sampling_prob = None

    def HT(self, w, sparsity):
        if sparsity is None or sparsity >= w.size:
            return np.array(w, copy=True)
        if sparsity <= 0:
            return np.zeros_like(w)

        flat_w = np.asarray(w).ravel()
        abs_flat_w = np.abs(flat_w)
        threshold = np.partition(abs_flat_w, -sparsity)[-sparsity]
        greater_idx = np.flatnonzero(abs_flat_w > threshold)
        tie_idx = np.flatnonzero(abs_flat_w == threshold)
        top_k_idx = np.concatenate((greater_idx, tie_idx[:sparsity - greater_idx.size]))
        out = np.zeros_like(flat_w)
        out[top_k_idx] = flat_w[top_k_idx]
        return out.reshape(np.shape(w))

    def _loss_b(self, w, X, y, loss):
        y = np.asarray(y).reshape(-1, 1)
        aa = y * X.dot(w)

        if loss == 'sigmoid':
            prob = 1.0 / (1.0 + np.exp(aa))
            b = y * (1.0 - prob) * prob

        elif loss == 'logistic':
            prob = 1.0 / (1.0 + np.exp(-aa))
            b = y * (1.0 - prob)

        elif loss == 'NN2':
            prob = 1.0 / (1.0 + np.exp(aa))
            b = 2.0 * y * (1.0 - prob) * (prob ** 2)

        else:
            raise ValueError(f"Unknown loss: {loss}")

        return b

    def _sample_indices_from_cdf(self, cdf, batch_size):
        u = np.random.rand(batch_size)
        idx = np.searchsorted(cdf, u, side='left')
        return idx

    def fit(self, X_train_pri, X_train, y_train, X_test=None, y_test=None,
            batch_size=None, L=None, loss='sigmoid', eval=True,
            sparsity=None, m=None, pr=False):
        """
        Parameters:
            X_train_pri : data matrix used to compute sample-wise Lipschitz constants
            X_train     : training data matrix
            y_train     : training labels
            X_test      : test data matrix
            y_test      : test labels
            batch_size  : mini-batch size
            L           : global Lipschitz constant used for the step size
            loss        : loss function identifier
            eval        : whether to record objective values and evaluation metrics
            sparsity    : target number of nonzero entries
            m           : full-gradient refresh parameter: P(full_grad) = 1/m
            pr          : whether to print progress information
        """

        n, d = X_train.shape

        if batch_size is None:
            batch_size = max(1, int(np.sqrt(n)))
        batch_size = int(batch_size)

        if m is None:
            m = max(1, int(np.sqrt(n)))
        m = int(m)

        self.w = self.HT(self.w, sparsity)
        wp = self.w.copy()

        L_sample = self.compute_sample_L(X_train_pri, loss=loss)
        p_sample = L_sample / L_sample.sum()
        self.importance_sampling_prob = p_sample
        self.uniform_sampling_prob = None

        cdf_sample = np.cumsum(p_sample)
        cdf_sample[-1] = 1.0

        v = self.grad_loss(self.w, X_train, y_train, loss) / n + self.lam * self.w

        if eval:
            f_val = self.obj_func(self.w, X_train, y_train, loss)
            self.obj.append(f_val + self.obj_reg(self.w))
            if pr:
                print('obj: ', self.obj[-1])

        self.iter = [0]
        self.Time = [0.0]

        start_time = time.time()

        support_stable_count = 0
        for iter in range(self.maxIter):

            if iter > 0 and np.random.randint(m) != 0:
                idx = self._sample_indices_from_cdf(cdf_sample, batch_size)

                Xb = X_train[idx]
                yb = np.asarray(y_train)[idx]

                scale = (1.0 / (n * p_sample[idx])).reshape(-1, 1)

                bw = self._loss_b(self.w, Xb, yb, loss)
                bwp = self._loss_b(wp, Xb, yb, loss)

                delta_b = (bw - bwp) * scale

                grad_diff_loss = - Xb.T.dot(delta_b) / batch_size
                grad_diff_reg = self.lam * (self.w - wp)

                v = v + grad_diff_loss + grad_diff_reg

            else:
                v = self.grad_loss(self.w, X_train, y_train, loss) / n + self.lam * self.w

            wp = self.w.copy()
            self.w = self.HT(self.w - v / (self.eta * L), sparsity)

            step_err = np.linalg.norm(self.w - wp)

            if eval:
                self.Time.append(time.time() - start_time)
                self.iter.append(iter + 1)

                f_val = self.obj_func(self.w, X_train, y_train, loss)
                self.obj.append(f_val + self.obj_reg(self.w))

                if pr:
                    print('iter', iter, 'obj:', self.obj[-1], 'step_err:', step_err)

            if self.iter[-1] >= self.maxIter:
                self.acc_test.append(self.Comp_acc(X_test, y_test))
                print('Accuracy of the test set in VR-IHT-IS', self.acc_test[-1])
                break

            rel_err = step_err / max(1.0, np.linalg.norm(wp))
            supp_same = set(np.nonzero(self.w)[0]) == set(np.nonzero(wp)[0])
            if supp_same:
                support_stable_count += 1
            else:
                support_stable_count = 0

            if support_stable_count >= 5 and rel_err <= self.tol:
                print('Stopped by relative error and stable support at', iter, '-th iteration')
                self.acc_test.append(self.Comp_acc(X_test, y_test))
                print('Accuracy of the test set in VR-IHT-IS', self.acc_test[-1])
                break

        return self


    def Comp_acc(self, X, y):
        n, d = X.shape
        if d == self.w.shape[0] - 1:
            X = np.c_[np.ones((n, 1)), X]

        y = np.c_[y]
        y_pred = np.sign(X.dot(self.w))
        y_pred[y_pred == 0] = 1
        return np.mean(y == y_pred)

    def grad_loss(self, w, X, y, loss):
        b = self._loss_b(w, X, y, loss)
        grad = - X.T.dot(b)
        return grad

    def obj_func(self, w, X, y, loss):
        y = np.asarray(y).reshape(-1, 1)
        aa = y * X.dot(w)

        if loss == 'sigmoid':
            prob = 1.0 / (1.0 + np.exp(aa))
            f_val = prob.mean()

        elif loss == 'NN2':
            prob = 1.0 / (1.0 + np.exp(aa))
            f_val = (prob ** 2).mean()

        else:
            raise ValueError(f"Unknown loss: {loss}")

        return f_val

    def obj_reg(self, w):
        return 0.5 * self.lam * np.sum(w * w)

    def compute_sample_L(self, X, loss):
        if sparse.issparse(X):
            row_norm2 = np.asarray(X.multiply(X).sum(axis=1)).ravel()
        else:
            row_norm2 = np.sum(X * X, axis=1)

        if loss == 'sigmoid':
            return (1.0 / (6.0 * np.sqrt(3.0))) * row_norm2 + self.lam
        elif loss == 'NN2':
            return ((39.0 + 55.0 * np.sqrt(33.0)) / 2304.0) * row_norm2 + self.lam
        else:
            raise ValueError(f"Unknown loss: {loss}")