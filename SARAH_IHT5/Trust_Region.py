import numpy as np
import time 
import warnings
from scipy.special import softmax, log_softmax

warnings.filterwarnings('ignore')

class Trust_Region:
    def __init__(self, w0, lam=1e-4, maxIter=30, tol=None):

        self.w = w0
        self.lam = lam
        self.maxIter = maxIter
        self.tol = tol

        self.Time = [0]
        self.obj = []
        self.test_acc = [] 
        self.iter = []
        

    def HT(self, w, k):
      
       flattened_w = w.ravel()
       top_k_indices = np.argpartition(np.abs(flattened_w), -k)[-k:]
       thresholded_w = np.zeros_like(flattened_w)
       thresholded_w[top_k_indices] = flattened_w[top_k_indices]
      
       reshaped_tensor = thresholded_w.reshape(w.shape)
    
       return reshaped_tensor
    
    def predict(self, X):
       scores = np.dot(X, self.w)
       return np.argmax(scores, axis=1)

    def accuracy(self, X, y):
       y_pred = self.predict(X)
       y_true = np.argmax(y, axis=1) 
       return np.mean(y_pred == y_true)


    def fit(self, X_train, y_train, X_test, y_test, batch_size=None, eval=True,
            sparsity=None, pr=False, eta1=1e-3, eta2=1e-3,
            delta0=1.0, deltamax=10.0, gamma=2.0):

        n, d = X_train.shape

        self.w = self.HT(self.w, sparsity)
        delta = delta0

        indices = np.arange(n)

        self.Time = [0]
        self.obj = []
        self.iter = [0]
        self.test_acc = []

        if eval:
            f_val, _ = self.full_batch_F_and_grad(self.w, X_train, y_train)
            self.obj.append(f_val)
            if pr:
                print("obj:", self.obj[-1])

        support_stable = 0
        start_time = time.time()

        for iter in range(self.maxIter):

            batch_indices = np.random.choice(indices, size=batch_size, replace=False)
            Xb = X_train[batch_indices]
            yb = y_train[batch_indices]

            w_old = self.w.copy()

            f1, grad = self.full_batch_F_and_grad(w_old, Xb, yb)

            step = min(delta / np.linalg.norm(grad), 1.0)
            wk = self.HT(w_old - step * grad, sparsity)

            fk, _ = self.full_batch_F_and_grad(wk, Xb, yb)

            accepted = False
            if ((f1 - fk) / (np.linalg.norm(grad) * delta) > eta1) and (np.linalg.norm(grad) > eta2 * delta):
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

                f_val, _ = self.full_batch_F_and_grad(self.w, X_train, y_train)
                self.obj.append(f_val)

                if pr:
                    print("iter", iter, "obj:", self.obj[-1], "rel_err:", rel_err, "stable:", support_stable)

            if self.iter[-1] == self.maxIter:
                self.test_acc.append(self.accuracy(X_test, y_test))
                print('Accuracy of the test set in PIHT', self.test_acc[-1])
                print('Training loss in PIHT', self.obj[-1])
                print('Time in PIHT', self.Time[-1])
                break

            if self.tol is not None:
                supp = set(np.nonzero(self.w.ravel())[0])
                supp_old = set(np.nonzero(w_old.ravel())[0])

                if not accepted:
                    pass
                else:
                    if supp == supp_old:
                        support_stable += 1
                        if rel_err <= self.tol and support_stable >= 5:
                            self.test_acc.append(self.accuracy(X_test, y_test))
                            print('Accuracy of the test set in PIHT', self.test_acc[-1])
                            print('Stopped by rel_err and support stability at', iter, '-th iteration')
                            print('Training loss in PIHT', self.obj[-1])
                            print('Time in PIHT', self.Time[-1])
                            break
                    else:
                        support_stable = 0

    def full_batch_F_and_grad(self, W, X, y):
        num_samples = X.shape[0]
        scores = np.dot(X, W)
        probs = softmax(scores, axis=1)
        assert probs.shape == y.shape
        loss =  -np.sum(y * log_softmax(scores, axis=1)) / num_samples  + self.lam * np.linalg.norm(W, ord='fro')**2 
        dscores = (probs - y) / num_samples
        grad = np.dot(X.T, dscores) + 2 * self.lam * W
        return loss, grad
