from sklearn.metrics import accuracy_score
import numpy as np
import time 
from sklearn.utils import shuffle
from libsvm.svmutil import svm_read_problem
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
import warnings

warnings.filterwarnings('ignore')


class Trust_Region:
    def __init__(self, w0, lam=1e-4, maxIter=None, maxTime=None, tol=None):

        self.w = w0
        self.lam = lam
        self.maxIter = maxIter
        self.maxTime = maxTime
        self.tol = tol

        self.Time = []
        self.obj = []
        self.acc_test = []
        self.cardinality = []
        self.reg_val = None

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

    def fit(self, X_train, y_train, X_test=None, y_test=None, batch_size=None, loss='sigmoid', eval=True, sparsity=None, pr=False, eta1=1e-3, eta2=1e-3, delta0=1.0, deltamax=10.0, gamma=2.0):

        n, d = X_train.shape

        self.w = self.HT(self.w, sparsity)
        wp = self.w
        delta = delta0

        indices = np.arange(n)
            
            
        if eval:
            f_val = self.obj_func(wp, X_train, y_train, loss)
            
            self.obj.append(f_val + self.obj_reg(wp))
            if pr:
                print('obj: ',self.obj[-1])

        self.iter = [0]
        self.Time.append(0.0)
        self.cardinality = [0]

        start_time = time.time()

        support_stable_count = 0
        for iter in range(self.maxIter):

            batch_indices = np.random.choice(indices, size=batch_size, replace=False)
            X = X_train[batch_indices]
            y = y_train[batch_indices]

            w_old = self.w.copy()

            grad = self.grad_loss(w_old, X, y, loss) / batch_size + self.lam * w_old
            grad_norm = np.linalg.norm(grad)

            step = min(delta / grad_norm, 1)
            wk = self.HT(w_old - step * grad, sparsity)

            f1 = self.obj_func(w_old, X, y, loss) + self.obj_reg(w_old)
            fk = self.obj_func(wk, X, y, loss) + self.obj_reg(wk)

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
                self.Time.append(time.time() - start_time)
                self.iter.append(iter + 1)
                self.cardinality.append(int(np.count_nonzero(self.w)))

                f_val = self.obj_func(self.w, X_train, y_train, loss)
                self.obj.append(f_val + self.obj_reg(self.w))

                if pr:
                    print('iter', iter, 'obj:', self.obj[-1], 'step_err:', step_err)
            
            if self.iter[-1] >= self.maxIter:
                self.acc_test.append(self.Comp_acc(X_test, y_test))
                print('Accuracy of the test set in PIHT', self.acc_test[-1])
                break
            if self.tol is not None:
                supp_same = set(np.nonzero(self.w)[0]) == set(np.nonzero(w_old)[0])
                if not accepted:
                        pass
                else:
                    if supp_same:
                        support_stable_count += 1
                        if support_stable_count >= 5 and rel_err <= self.tol:
                                        print('Stopped by relative error and stable support at', iter, '-th iteration')
                                        self.acc_test.append(self.Comp_acc(X_test, y_test))
                                        print('Accuracy of the test set in PIHT', self.acc_test[-1])
                                        break
                    else:
                        support_stable_count = 0

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

        if loss == 'sigmoid':
            y = np.c_[y]
            aa = y * X.dot(w)
            prob = 1.0 / (1.0 + np.exp(aa))
            b = y * (1.0 - prob) * prob
            grad = - X.T.dot(b) 

        elif loss == 'logistic':
            y = np.c_[y]
            aa = y * X.dot(w)
            prob = 1.0 / (1.0 + np.exp(-aa)) 
            b = y * (1.0 - prob)
            grad = - X.T.dot(b) 

        elif loss == 'NN2':
            y = np.c_[y]
            aa = y * X.dot(w)
            prob = 1.0 / (1.0 + np.exp(aa))
            b = 2.0 * y * (1.0 - prob) * (prob ** 2)
            grad = - X.T.dot(b) 

        else:
            raise ValueError(f"Unknown loss: {loss}")

        return grad

    def obj_func(self, w, X, y, loss):
        if loss == 'sigmoid':
            y = np.c_[y]
            aa = y * X.dot(w)
            prob = 1.0 / (1.0 + np.exp(aa))
            f_val = prob.mean()

        elif loss == 'logistic':
            y = np.c_[y]
            aa = y * X.dot(w)
            f_val = np.logaddexp(0.0, -aa).mean()

        elif loss == 'NN2':
            y = np.c_[y]
            aa = y * X.dot(w)
            prob = 1.0 / (1.0 + np.exp(aa))
            f_val = (prob ** 2).mean()

        else:
            raise ValueError(f"Unknown loss: {loss}")

        return f_val

    def obj_reg(self, w):
        return 0.5 * self.lam * np.sum(w * w)

    
