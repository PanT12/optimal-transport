import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any
from .base import *
import timeit
from scipy.sparse.linalg import LinearOperator, cg

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


class SSNSSolver(optimal_transport):
    def __init__(self, *args, cg_tol=1e-11, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_sparse = True

        # CG params
        self.cg_tol = cg_tol
        self.cg_maxit = max(self.m, self.n) * 100

        # Algorithm 2 default params
        self.mu0 = 1.0
        self.nu0 = 1e-2
        self.cl = 0.1
        self.cu = 1.0
        self.kappa = 1e-3
        self.gamma = 1.0
        self.rho0 = 0.25            # ρ0 ∈ (0, 1/2)
        self.step_candidates = (1.0, 0.5, 0.25, 0.1)

    def _tau(self, alpha, beta):
        # T_ij = exp((alpha_i + beta_j - M_ij)/eta)
        return np.exp((alpha[:, None] + beta[None, :] - self.C) / self.eta)

    def _select_small_mask(self, v, delta):
        """
        select_small(v, δ): select some smallest elements such that the summation <= δ
        return mask
        """
        # if delta <= 0:
        #     return np.zeros_like(v, dtype=np.int8)
        idx = np.argsort(v)
        vs = v[idx]
        csum = np.cumsum(vs)
        # Find the largest S such that sum_{s=1..S} <= δ
        S = np.searchsorted(csum, delta, side='right')
        mask = np.zeros_like(v, dtype=np.int8)
        if S > 0:
            pick = idx[:S]
            mask[pick] = 1
        return mask

    def _apply_mask(self, v, mask):
        return v * mask

    def pack(self, alpha, beta_tilde):
        return np.concatenate([alpha, beta_tilde], axis=0)

    def unpack(self, x):
        alpha = x[:self.m]
        beta_tilde = x[self.m:]
        beta = np.zeros(self.n, dtype=float)
        beta[:-1] = beta_tilde
        return alpha, beta_tilde, beta

    def recover_plan(self, x):
        alpha, _, beta = self.unpack(x)
        T = self._tau(alpha, beta)
        return alpha, beta, T

    def f(self, alpha, beta, T):
        # f(x) = η * 1ᵀ T 1 - αᵀ a - βᵀ b
        return self.eta * T.sum() - alpha @ self.a - beta @ self.b

    def orginal_problem_cost(self, T):
        return np.sum(self.C * T)

    def grad(self, T):
        # alpha, beta_tilde, beta = self.unpack(x)
        # T = self._tau(alpha, beta)
        # alpha, beta, T = self.recover_plan(x)
        g_alpha = T.sum(axis=1) - self.a                    # (m,)
        g_beta  = T.sum(axis=0) - self.b                    # (n,)
        g_beta_tilde = g_beta[:-1]                          # remove last component (β_m fixed)
        return np.concatenate([g_alpha, g_beta_tilde], axis=0)  # (n+m-1,)

    # ===== Algorithm 1: Hessian sparsification =====

    def sparsify_hessian(self, x, delta):
        """
        Return LinearOperator，calculate (H_Ω) y
        H_δ = (1/η) [ diag(T1)   T̃_δ
                      T̃_δᵀ      diag(T̃ᵀ1) ]
        """
        alpha, beta_tilde, beta = self.unpack(x)
        T = self._tau(alpha, beta)      # (n, m)

        Delta = np.zeros_like(T)
        for j in range(self.n - 1):
            phi_col = self._select_small_mask(T[:, j], delta)
            Delta[:, j] = self._apply_mask(T[:, j], phi_col)

        for i in range(self.m):
            phi_row = self._select_small_mask(Delta[i, :], delta)
            Delta[i, :] = self._apply_mask(Delta[i, :], phi_row)

        T_delta = T - Delta
        # Block pieces
        r = T.sum(axis=1)                # (m,)
        c = T.sum(axis=0)[:-1]           # (n-1,)
        T_tilde_delta = T_delta[:, :-1]  # (m, n-1)

        scale = 1.0 / self.eta

        if self.use_sparse:
            D_alpha = r * scale                      # diag block (n,)
            D_beta  = c * scale                      # diag block (m-1,)

            T_block = T_tilde_delta * scale          # (n, m-1)
            T_block = sp.csr_matrix(T_block)         # sparse

            def Hdelta_mv(v):
                # v = [v_a (n,); v_b (m-1,)]
                va = v[:self.m]
                vb = v[self.m:]
                out_a = D_alpha * va + T_block @ vb                  # (n,)
                out_b = T_block.T @ va + D_beta * vb                 # (m-1,)
                return np.concatenate([out_a, out_b], axis=0)

            return Hdelta_mv, None  # 不返回显式矩阵
        else:
            A = np.diag(r) * scale
            B = T_tilde_delta * scale
            C = B.T
            D = np.diag(c) * scale
            H = np.block([[A, B],
                          [C, D]])
            return None, H

    # ===== Solve (H_δ + λI) p = -g =====
    def _solve_direction(self, Hdelta_mv, g, lam):
        n_tot = self.m + self.n - 1
        assert Hdelta_mv is not None

        def shift_mv(v):
            return Hdelta_mv(v) + lam * v

        # use callback to record the cg iterations
        it_counter = {"n": 0}

        def cb(xk):
            it_counter["n"] += 1

        Aop = LinearOperator((n_tot, n_tot), matvec=shift_mv, dtype=np.float64)
        p, info = cg(Aop, -g, atol=self.cg_tol, maxiter=self.cg_maxit, callback=cb)
        if info != 0:
            raise RuntimeError(f"CG did not converge, info={info}")
        cg_iter = it_counter["n"]
        return p, cg_iter

    def _model_drop(self, g, Hdelta_mv, p, xi):
        """
        mk(0) - mk(ξp) = - (gᵀ ξp + 0.5 ξ^2 pᵀ H_δ p)
        """
        gp = g @ (xi * p)
        Hp = Hdelta_mv(p)
        pHp = p @ Hp
        return - (gp + 0.5 * (xi ** 2) * pHp)

    def _pick_step(self, x, p):
        """
        Algorithm 3：Given fixed candidate stepsize, choose one
        return (xi*, f(x+xi*p))
        """
        f_best = np.inf
        xi_best = self.step_candidates[0]
        alpha, beta, T = self.recover_plan(x)
        f_x = self.f(alpha, beta, T)
        for xi in self.step_candidates:
            xt = x + xi * p
            alpha_t, beta_t, Tt = self.recover_plan(xt)
            ft = self.f(alpha_t, beta_t, Tt)
            if ft < f_best:
                f_best, xi_best = ft, xi
            if ft < f_x:
                return xi, ft
        return xi_best, f_best

    def optimize(self, tol, max_iter, time_max=np.inf, verbose=True):
        # ===== main（Algorithm 2） =====

        # initialization
        alpha = np.zeros(self.m)
        beta_tilde = np.zeros(self.n - 1)
        x = self.pack(alpha, beta_tilde)
        mu = self.mu0

        t_total = 0.0

        alpha, beta, T = self.recover_plan(x)
        gk = self.grad(T)
        gnorm = np.linalg.norm(gk)
        self.record(t_total=t_total, cg_iter=0, gnorm=gnorm, X=T, cg_time=0.0)

        for k in range(max_iter):
            t_start = timeit.default_timer()
            # history["cost"].append(self.f(x))
            # history["mu"].append(mu)

            if gnorm < tol or t_total > time_max:
                break

            delta_k = self.nu0 * (gnorm ** self.gamma)
            lam_k = mu * gnorm
            # history["delta"].append(delta_k)

            # construct H_δk
            Hdelta_mv, H_dense = self.sparsify_hessian(x, delta_k)

            cg_start_time = timeit.default_timer()
            pk, cg_iter = self._solve_direction(Hdelta_mv, gk, lam_k)
            cg_end_time = timeit.default_timer()
            cg_time = cg_end_time - cg_start_time

            # stepsize（Algorithm 3）
            xi_k, f_trial = self._pick_step(x, pk)

            denom = self._model_drop(gk, Hdelta_mv, pk, xi_k)
            numer = self.f(alpha, beta, T) - f_trial
            rho_k = numer / denom if denom > 0 else -np.inf
            # history["rho"].append(rho_k)

            if rho_k < self.rho0:
                mu = 4.0 * mu
                accepted = False
                x_next = x
            elif rho_k >= 1 - self.rho0:
                mu = max(mu / 2.0, self.kappa)
                accepted = True
                x_next = x + xi_k * pk
            else:
                accepted = rho_k > 0
                x_next = x + xi_k * pk if accepted else x

            # history["accepted"].append(accepted)
            x = x_next

            alpha, beta, T = self.recover_plan(x)
            gk = self.grad(T)
            gnorm = np.linalg.norm(gk)

            t_end = timeit.default_timer()
            t_total += t_end - t_start

            self.record(t_total=t_total, cg_iter=cg_iter, gnorm=gnorm, X=T, cg_time=cg_time)

            if verbose and k % 10 == 0:
                print(f"iter = {k:5d}, grad_norm = {gnorm:2.4e}, "
                      f"row_resid = {self.history['row_resid'][-1]:2.2e}, "
                      f"col_resid = {self.history['col_resid'][-1]:2.2e}, "
                      f"time = {t_total:3.2f}")

        return T


# ========== 使用示例 ==========
if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    m = n = 1000
    seed = 41
    RESULT_ROOT = f"Results/alg_compare_m{m}_n{n}_rng{seed}"
    OPT_CACHE_DIR = os.path.join(RESULT_ROOT, "_cache_opt")  # 专门放 a,b,opt
    os.makedirs(OPT_CACHE_DIR, exist_ok=True)
    opt_cache_path = os.path.join(OPT_CACHE_DIR, f"trial{1:03d}_opt_ab.npz")
    cache = np.load(opt_cache_path, allow_pickle=False)
    a = cache["a"]
    b = cache["b"]
    opt = float(cache["opt"])
    J = np.arange(n)
    C = (J[None, :] - J[:, None]) ** 2
    C = C / np.max(C)

    solver = SSNSSolver(C=C, eta=1e-5, a=a, b=b, obj_truth=opt)
    X = solver.optimize(tol=1e-7, max_iter=3000)

    plt.plot(solver.history["abs_err"], label="abs_err")
    plt.plot(solver.history["grad_norm"], label="gnorm")
    plt.yscale('log')
    plt.legend()
    plt.show()


    # print("final grad norm:", log["grad_norm"][-1])
    # print("final f:", log["f"][-1])
