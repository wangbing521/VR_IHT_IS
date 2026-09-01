import numpy as np
import time 
import warnings

warnings.filterwarnings('ignore')

class Trust_Region:
    def __init__(self, w0, lam=1e-4, maxIter=30, maxTime=None, tol=None):

        self.w = w0
        self.lam = lam
        self.maxIter = maxIter
        self.maxTime = maxTime
        self.tol = tol

        self.obj = []
        self.iter = []

    def HT(self, w, sparsity):
        w = np.array(w, copy=True)

        if sparsity is None:
            return w
        if sparsity <= 0:
            return np.zeros_like(w)

        flat_abs = np.abs(w).ravel()
        if sparsity >= flat_abs.size:
            return w

        keep = np.argpartition(flat_abs, -sparsity)[-sparsity:]
        out = np.zeros_like(w)
        out_flat = out.ravel()
        w_flat = w.ravel()
        out_flat[keep] = w_flat[keep]
        return out

    def fit(self, X_train, y_train, batch_size=None,
        eval=True, sparsity=None, pr=False, eta1=1e-3, eta2=1e-3,
        delta0=1.0, deltamax=10.0, gamma=2.0):

        n, d = X_train.shape

        self.w = self.HT(self.w, sparsity)

        delta = delta0

        indices = np.arange(n)

        if eval:
            f_val = self.obj_func(self.w, X_train, y_train)
            self.obj.append(f_val + self.obj_reg(self.w))
            if pr:
                print('obj: ', self.obj[-1])

        consecutive_support = 0
        self.iter = [0]

        for iter in range(self.maxIter):

            batch_indices = np.random.choice(indices, size=batch_size, replace=False)
            X = X_train[batch_indices]
            y = y_train[batch_indices]

            w_old = self.w.copy()

            grad = self.grad_loss(w_old, X, y) / batch_size + self.lam * w_old
            grad_norm = np.linalg.norm(grad)

            step_scale = min(delta / grad_norm, 1) if grad_norm > 0 else 0.0
            wk = self.HT(w_old - step_scale * grad, sparsity)

            f1 = self.obj_func(w_old, X, y) + self.obj_reg(w_old)
            fk = self.obj_func(wk, X, y) + self.obj_reg(wk)

            accepted = False
            if ((f1 - fk) / (grad_norm * delta) > eta1) and (grad_norm > eta2 * delta):
                delta = min(delta * gamma, deltamax)
                self.w = wk
                accepted = True
            else:
                delta = delta / gamma
                self.w = w_old
                accepted = False

            step_err = np.linalg.norm(self.w - w_old)
            rel_err = step_err / max(1.0, np.linalg.norm(w_old))

            if eval:
                self.iter.append(1 + iter)

                f_val = self.obj_func(self.w, X_train, y_train)
                self.obj.append(f_val + self.obj_reg(self.w))
                if pr:
                    print('iter', iter, 'obj: ', self.obj[-1], 'rel_err:', rel_err,
                          'accepted:', accepted, 'delta:', delta)

            if self.iter[-1] > self.maxIter:
                break

            if self.tol is not None:
                supp_curr = set(np.nonzero(self.w.ravel())[0])
                supp_prev = set(np.nonzero(w_old.ravel())[0])

                if not accepted:
                    pass
                else:
                    if supp_curr == supp_prev:
                        consecutive_support += 1
                        if consecutive_support >= 5 and rel_err <= self.tol:
                            print('Stopped by convergence at ', iter, '-th iteration in PIHT')
                            break
                    else:
                        consecutive_support = 0

    
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
