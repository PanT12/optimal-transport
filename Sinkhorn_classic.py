import numpy as np
from Sinkhorn import *
import math
from scipy.special import logsumexp
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import cg
from math import ceil, log
from scipy import sparse
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
        X = np.ones((self.m, self.n)) * self.a[:, None] / self.n
        log_x = np.log(X)
        f = np.zeros(self.m)
        g = np.zeros(self.n)
        # log_x = self.internal_x(f, g)
        # X = np.exp(log_x)
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
                    f"iter = {iter:5d}, rel_err = {self.history['rel_conv'][-1]:2.4e}, time_iter = {t_total:3.2f}"
                )

            if self.history['rel_conv'][-1] < tol or t_total > time_max:
                break

        return X

    # def comp_obj(self, X):
    #     X_safe = np.clip(X, 1e-300, None)
    #     # 如果想和常见定义一致，用: self.eta * (X_safe * (np.log(X_safe) - 1.0)).sum()
    #     return float((self.C * X_safe).sum() + self.eta * (X_safe * np.log(X_safe)).sum())
    #
    # def internal_x(self, f, g):
    #     # 去掉 -1.0
    #     return (np.expand_dims(f, 1) + np.expand_dims(g, 0) - self.C) / self.eta
    #
    # def sinkhorn(self, maxit=10000, tol=1e-4, verbose=False):
    #     X = np.ones((self.m, self.n)) * self.a[:, None] / self.n
    #     log_x = np.log(np.clip(X, 1e-300, None))
    #     f = np.zeros(self.m)
    #     g = np.zeros(self.n)
    #     t_total = 0.0
    #
    #     obj_old = float((self.C * X).sum())
    #     col_err = np.linalg.norm(X.sum(axis=0) - self.b)
    #     history = {"col_resid": [col_err], "row_resid": [0.0], "cost": [obj_old],
    #                "ent_cost": [self.comp_obj(X)], "time": [0.0], "rel_err": [], "abs_err": []}
    #
    #     if self.obj_truth is None:
    #         rel_err = 1.0;
    #         abs_err = 1.0
    #     else:
    #         rel_err = abs(self.obj_truth - obj_old) / abs(self.obj_truth)
    #         abs_err = abs(self.obj_truth - obj_old)
    #     history["rel_err"].append(rel_err);
    #     history["abs_err"].append(abs_err)
    #
    #     # 预先保存 loga, logb 以减少重复
    #     loga = self.log_a if hasattr(self, "log_a") else np.log(np.clip(self.a, 1e-300, None))
    #     logb = self.log_b if hasattr(self, "log_b") else np.log(np.clip(self.b, 1e-300, None))
    #
    #     for it in range(maxit):
    #         t_start = timeit.default_timer()
    #
    #         # ---- 建议：闭式赋值（更干净稳定）----
    #         f = self.eta * (loga - logsumexp(((-self.C + g[None, :]) / self.eta), axis=1))
    #         g = self.eta * (logb - logsumexp((((-self.C).T + f[None, :]) / self.eta), axis=1))
    #         log_x = self.internal_x(f, g)
    #
    #         t_total += timeit.default_timer() - t_start
    #
    #         # Diagnostics
    #         X = np.exp(log_x)
    #         col_err = np.linalg.norm(X.sum(axis=0) - self.b)
    #         row_err = np.linalg.norm(X.sum(axis=1) - self.a)
    #         cost = float((self.C * X).sum())
    #         entropy_cost = self.comp_obj(X)
    #         history["ent_cost"].append(entropy_cost)
    #         history["col_resid"].append(col_err)
    #         history["row_resid"].append(row_err)
    #         history["cost"].append(cost)
    #         history["time"].append(t_total)
    #
    #         if self.obj_truth is not None:
    #             relative_error = abs(cost - self.obj_truth) / abs(self.obj_truth)
    #             absolute_error = abs(cost - self.obj_truth)
    #         else:
    #             relative_error = abs(cost - obj_old) / max(1e-300, abs(obj_old))
    #             absolute_error = abs(cost - obj_old)
    #             obj_old = cost
    #
    #         history["rel_err"].append(relative_error)
    #         history["abs_err"].append(absolute_error)
    #
    #         if verbose and it % 100 == 0:
    #             print(f"iter={it:5d}, feas={row_err + col_err:8.2e}, rel_err={relative_error:8.2e}, t={t_total:6.2f}s")
    #
    #         if (row_err + col_err) < tol:
    #             break
    #
    #     return X, history
