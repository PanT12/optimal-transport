import numpy as np
from Sinkhorn import *
import timeit


class ExtraGrad(optimal_transport):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.C1 = 124.0
        self.C2 = 0.024
        self.C3 = 1.0
        self.C_use = 1.0
        self.C3_prac = 1e-2
        self.B_prac = 1.0
        self.eta_prac = 0.0

    def _normalize_rows(self, x, axis=1):
        """Normalize along rows to the simplex (Algorithm 2)."""
        s = x.sum(axis=axis, keepdims=True)
        return x / s
        # s = np.where(s > 0, s, 1.0)
        # y = x / s
        # avoid exact zeros to keep logs safe
        # return np.clip(y, eps, None) / np.clip(y, eps, None).sum(axis=axis, keepdims=True)

    def _softmax_from_log(self, logw, axis=1):
        """Row-wise softmax from log-weights, numerically stable."""
        z = logw - logw.max(axis=axis, keepdims=True)
        w = np.exp(z)
        return w / w.sum(axis=axis, keepdims=True)

    def _adjust_mu(self, mu, B):
        """
        (16c) adjustment: mu_{j,s} <- max( mu_{j,s}, e^{-B} * max(mu_{j,+}, mu_{j,-}) ), then renormalize per j.
        Keeps ratio <= e^B.
        """
        m = mu.max(axis=1, keepdims=True)
        floor = np.exp(-B) * m
        mu_adj = np.maximum(mu, floor)
        return self._normalize_rows(mu_adj, axis=1)

    # main algorithm 1
    def optimize(self, tol, max_iter=1000, mode="practical", time_max=np.inf, verbose=True):
        C_norm = np.max(np.abs(self.C))
        epsilon = 1e-4
        if C_norm > 0:
            self.C = self.C / C_norm
            epsilon = epsilon / C_norm

        # Initialization
        p = np.full((self.m, self.n), 1.0 / self.n)
        mu = np.full((self.n, 2), 0.5)
        mu_adjust = mu.copy()
        halfC = 0.5 * self.C

        P_hat = self.a[:, None] * p

        t_total = 0.0
        self.record(t_total=t_total, cg_iter=0, gnorm=None, X=P_hat)

        # design the stepsize
        if mode == "theory":
            B = self.C1 * np.log(self.n / max(epsilon, 1e-300))
            eta = self.C2 / (2.0 * epsilon * np.sqrt(B * np.log(self.n)))
            eta_mu = 15.0 * self.C2 * np.sqrt(B) / (self.b + self.C3 / self.n)          # η_{μ,j}
            gamma = (self.C2 / np.sqrt(B)) * np.ones_like(self.a)             # γ_i = η_{p,i} r_i (只用到这个乘积)
            C3_used = self.C3
        elif mode == "practical":
            B = self.B_prac
            eta = self.eta_prac
            eta_mu = self.C_use * np.sqrt(B) / (self.b + self.C3_prac / self.n)
            gamma = (self.C_use / np.sqrt(B)) * np.ones_like(self.a)              # γ_i = η_{p,i} r_i
            C3_used = self.C3_prac
        else:
            raise ValueError("mode must be 'theory' or 'practical'")

        for _ in range(max_iter):
            t_start = timeit.default_timer()
            # -------- Step 1: midpoints (15a)(15b) --------
            col_mass = (self.a[:, None] * p).sum(axis=0)  # Σ_i r_i p_{i,j}
            diff = col_mass - self.b
            # μ mid
            signs = np.array([+1.0, -1.0])          # s ∈ {+,-}
            log_mu_mid = (1.0 - eta) * np.log(mu_adjust) + (eta_mu[:, None] * diff[:, None] * signs[None, :])
            mu_mid = self._softmax_from_log(log_mu_mid, axis=1)

            # p mid
            dmu_adjust = (mu_adjust[:, 0] - mu_adjust[:, 1])  # μ_{j,+} - μ_{j,-}
            G = halfC + dmu_adjust[None, :]
            log_p_mid = (1.0 - eta) * np.log(np.maximum(p, 1e-300)) - (gamma[:, None] * G)
            p_mid = self._softmax_from_log(log_p_mid, axis=1)

            # -------- Step 2: main sequence (16a)(16b) -----
            col_mass_mid = (self.a[:, None] * p_mid).sum(axis=0)
            diff_mid = col_mass_mid - self.b

            log_mu_next = (1.0 - eta) * np.log(mu_adjust) + (eta_mu[:, None] * diff_mid[:, None] * signs[None, :])
            mu_next = self._softmax_from_log(log_mu_next, axis=1)

            dmu_next = (mu_mid[:, 0] - mu_mid[:, 1])
            G_next = halfC + dmu_next[None, :]
            log_p_next = (1.0 - eta) * np.log(np.maximum(p, 1e-300)) - (gamma[:, None] * G_next)
            p_next = self._softmax_from_log(log_p_next, axis=1)

            # -------- Step 3: adjust μ (16c) ---------------
            mu_adjust_next = self._adjust_mu(mu_next, B)

            p = p_next
            mu_adjust = mu_adjust_next

            t_end = timeit.default_timer()
            t_total += t_end - t_start

            # diagnostics of P_hat
            P_hat = self.a[:, None] * p
            self.record(t_total=t_total, cg_iter=0, gnorm=None, X=P_hat)

            if verbose and _ % 100 == 0:
                print(
                    f"iter = {_:5d}, rel_conv = {self.history['rel_conv'][-1]:2.4e}, "
                    f"row_resid = {self.history['row_resid'][-1]:2.4e}, "
                    f"col_resid = {self.history['col_resid'][-1]:2.4e}, "
                    f"time = {t_total:3.2f}"
                )

            if self.history['rel_conv'][-1] < tol or t_total > time_max:
                break

        # make it feasible
        t_start = timeit.default_timer()
        P_hat = self.a[:, None] * p
        P_tilde = self._round_to_marginals(P_hat, self.a, self.b)
        t_end = timeit.default_timer()
        t_total += t_end - t_start

        self.record(t_total=t_total, cg_iter=0, gnorm=None, X=P_tilde)

        return P_tilde

# ------------------- 使用示例 -------------------
# if __name__ == "__main__":
#     from GurobiSolver import *
#
#     m, n = 400, 400
#     np.random.seed(10)
#     a = np.random.rand(m)
#     a = a / np.sum(a)  # [0.54100398 0.01455541 0.44444061]
#     # print("a is ", a)
#     b = np.random.rand(n)
#     b = b / np.sum(b)  # [0.44833981 0.29847674 0.13459504 0.11858842]
#     # print("b is ", b)
#     C = np.random.rand(m, n)
#
#     guro = Gurobisolver(C, 0.1, a, b, 0.0)
#     gt = guro.Gurobi_Solver_original()
#     opt = np.sum(C * gt)
#
#     solver = ExtraGrad(C, 1e-3, a, b)
#     P, history = solver.optimize(tol=1e-7, tmax=10000)
#     # 验证边缘
#     print("row err L1 =", np.linalg.norm(P.sum(axis=1) - a, 1))
#     print("col err L1 =", np.linalg.norm(P.sum(axis=0) - b, 1))
#     print("OT cost     =", float((C * P).sum()))
#     print("value diff =", float((C * P).sum() - opt))
