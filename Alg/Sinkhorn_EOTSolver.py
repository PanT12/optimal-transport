from Alg.Sinkhorn import *
from scipy.special import logsumexp
import timeit

class SinkhornSolver(optimal_transport):
    def comp_obj(self, X):
        X[X < 1e-300] = 1e-300
        return (self.C * X).sum().item() + (self.eta * (X * np.log(X))).sum().item()

    def internal_x(self, f, g):
        value = (f[:, None] + g[None, :] - self.C) / self.eta - 1.0
        # value = (np.expand_dims(f, axis=1) + np.expand_dims(g, axis=0) - self.C) / self.eta - 1.0
        return value

    def grad(self, X):
        grad_f = X.sum(axis=1) - self.a
        grad_g = X.sum(axis=0) - self.b
        grad = np.concatenate([grad_f, grad_g], axis=0)
        return np.linalg.norm(grad)

    def optimize(self, maxit=10000, tol=1e-4, time_max=np.inf, verbose=True):
        # initialization
        f = np.zeros(self.m)
        g = np.zeros(self.n)
        log_x = self.internal_x(f, g)
        X = np.exp(log_x)
        t_total = 0.0
        gnorm = self.grad(X)
        self.record(t_total=t_total, cg_iter=0, gnorm=gnorm, X=X)

        for iter in range(maxit):
            t_start = timeit.default_timer()

            f += self.eta * (self.log_a - logsumexp(log_x, 1))
            log_x = self.internal_x(f, g)

            g += self.eta * (self.log_b - logsumexp(log_x, 0))
            log_x = self.internal_x(f, g)

            t_end = timeit.default_timer()
            t_total += t_end - t_start

            # Diagnostics on X
            X = np.exp(log_x)
            gnorm = self.grad(X)
            self.record(t_total=t_total, cg_iter=0, gnorm=gnorm, X=X)
            if verbose and iter % 100 == 0:
                print(
                    f"iter = {iter:5d}, grad_norm = {gnorm:2.4e}, time_iter = {t_total:3.2f}"
                )

            if gnorm < tol or t_total > time_max:
                break

        return X