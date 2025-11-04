import numpy as np
from .base import *
import timeit


class IPOTSolver(optimal_transport):
    def __init__(self, *args, L=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.L = int(L)
        self.log_gap = 100

    def _safe_div(self, num, den):
        return num / np.maximum(den, 1e-300)

    def optimize(self, maxit=10000, tol=1e-4, time_max=np.inf, verbose=False):
        # initialization
        X = np.ones((self.m, self.n)) * self.a[:, None] / self.n
        G = np.exp(-self.C / self.eta)
        c = np.ones(self.m)
        d = np.ones(self.n) / self.n

        t_total = 0.0
        self.record(t_total=t_total, cg_iter=0, gnorm=None, X=X)

        for it in range(maxit):
            t_start = timeit.default_timer()

            Q = G * X

            for _ in range(self.L):
                # a = μ / (Q b)
                Qb = Q @ d
                c = self._safe_div(self.a, Qb)

                # b = ν / (Q^T a)
                Qt_a = Q.T @ c
                d = self._safe_div(self.b, Qt_a)

            X = (c[:, None]) * Q * (d[None, :])

            t_end = timeit.default_timer()
            t_total += (t_end - t_start)

            self.record(t_total=t_total, cg_iter=0, gnorm=None, X=X)

            if verbose and it % 100 == 0:
                print(f"iter = {it:5d}, rel_conv = {self.history['rel_conv'][-1]:2.4e}, "
                      f"row_resid = {self.history['row_resid'][-1]:2.2e}, "
                      f"col_resid = {self.history['col_resid'][-1]:2.2e}, "
                      f"time = {t_total:3.2f}")

            if self.history['rel_conv'][-1] < tol or t_total > time_max:
                break
        return X

#
# # ===== 使用示意 =====
# from matplotlib import pyplot as plt
# m, n = 3, 4
# np.random.seed(10)
# a = np.random.rand(m)
# a = a / np.sum(a) # [0.54100398 0.01455541 0.44444061]
# # print("a is ", a)
# b = np.random.rand(n)
# b = b / np.sum(b) # [0.44833981 0.29847674 0.13459504 0.11858842]
# # print("b is ", b)
# C = np.random.rand(m, n)
#
# solver = IPOTSolver(C, eta=1.0, a=a, b=b, obj_truth=0.4164344012190675)
# X, history = solver.optimize(maxit=20000, tol=1e-5, verbose=True)
# print(X)
# plt.plot(history["rel_err"])
# plt.yscale("log")
# plt.show()
#
