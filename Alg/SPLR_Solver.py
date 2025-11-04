import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, cg
from scipy.optimize import line_search
from .base import *
import timeit


class SPLRSolver(optimal_transport):
    """
    Sparse-Plus-Low-Rank Quasi-Newton (Algorithm 1) for Entropic-regularized OT dual.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rho_max = 1e-2
        self.rho_min = 0.01 * self.rho_max
        self.rho = 0.1 * self.rho_max

        self.tau_max = 1e-3

        # CG parameters
        self.cg_maxit = max(self.m, self.n) * 100
        self.cg_tol = 1e-11

        # Wolfe line-search parameters
        self.c1, self.c2 = 1e-4, 0.9

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

    def _tau(self, alpha, beta):
        """T_ij = exp{ (α_i + β_j - M_ij) / eta }."""
        return np.exp((alpha[:, None] + beta[None, :] - self.C) / self.eta)

    def f(self, alpha, beta, T):
        """f(alpha, beta, T) = η * sum_ij T_{ij} - α^T a - β^T b"""
        return self.eta * T.sum() - alpha @ self.a - beta @ self.b

    def grad(self, T):
        """g(x) = [ T 1_m - a ; T̃^T 1_n - b̃ ]"""
        g_alpha = T.sum(axis=1) - self.a
        g_beta_t = T[:, :-1].sum(axis=0) - self.b[:-1]
        return np.concatenate([g_alpha, g_beta_t])

    def _omega_star_union(self, T_tilde, rho):
        """
        Ω*(ρ) = Ω(ρ) ∪ Ω*, where Ω* keeps a whole row and a column
        """
        n, m_1 = T_tilde.shape
        k = int(np.floor(rho * n * m_1))
        # k largest entries of T_tilde
        if k > 0:
            flat = T_tilde.ravel()
            idx = np.argpartition(flat, -k)[-k:]
            rows, cols = np.unravel_index(idx, T_tilde.shape)
            mask = np.zeros_like(T_tilde, dtype=bool)
            mask[rows, cols] = True
        else:
            mask = np.zeros_like(T_tilde, dtype=bool)

        # Ω*(ρ) = Ω(ρ) ∪ Ω*
        mask[0, :] = True
        mask[:, 0] = True
        return mask

    def _make_HOmega_shifted_op(self, T, mask, tau):
        """
        Return LinearOperator，calculate (H_Ω + τ I) y
        """
        # T = self._tau(alpha, beta)
        n, m = T.shape
        T_tilde = T[:, :-1]  # (n, m-1)

        D_alpha = (T.sum(axis=1) / self.eta)            # (n,)
        D_beta  = (T_tilde.sum(axis=0) / self.eta)      # (m-1,)

        T_block = (T_tilde / self.eta) * mask           # (n, m-1)
        T_block = sp.csr_matrix(T_block)

        def mv(v):
            va = v[:n]
            vb = v[n:]
            out_a = D_alpha * va + T_block @ vb
            out_b = T_block.T @ va + D_beta * vb
            return np.concatenate([out_a, out_b]) + tau * v

        size = n + (m - 1)
        return LinearOperator((size, size), matvec=mv, dtype=float)

    def _solve_B_rhs1(self, S_op, rhs, u, v, a, b, cg_tol=1e-8, cg_maxit=200):
        """
        Woodbury：设 S = H_Ω + τI，B = S + a uuᵀ + b vvᵀ.
        B^{-1} = S^{-1} - S^{-1} U (C^{-1} + Uᵀ S^{-1} U)^{-1} Uᵀ S^{-1}, U=[u v], C=diag(a,b).
        注意 b 可能为负（见公式 (7) 的构造），但 Woodbury 仍可用。
        """
        # 先求 S^{-1} * {rhs, u, v} （用 CG）
        y,  info1 = cg(S_op, rhs, tol=cg_tol, maxiter=cg_maxit)
        if info1 != 0:
            print("Warning: CG did not converge in _solve_B_rhs.")
        Su, info2 = cg(S_op, u,   tol=cg_tol, maxiter=cg_maxit)
        if info2 != 0:
            print("Warning: CG did not converge in _solve_B_rhs.")
        Sv, info3 = cg(S_op, v,   tol=cg_tol, maxiter=cg_maxit)
        if info3 != 0:
            print("Warning: CG did not converge in _solve_B_rhs.")

        # 2x2 小系统 K = C^{-1} + Uᵀ S^{-1} U
        Cinv = np.array([[1.0/a, 0.0],
                         [0.0,   1.0/b]])
        UT_Sinv_U = np.array([[u @ Su, u @ Sv],
                              [v @ Su, v @ Sv]])
        K = Cinv + UT_Sinv_U

        # 右端项 Uᵀ S^{-1} rhs
        UT_Sinv_rhs = np.array([u @ y, v @ y])

        w = np.linalg.solve(K, UT_Sinv_rhs)
        # y - [Su Sv] w
        return y - (Su * w[0] + Sv * w[1])

    def _solve_B_rhs2(self, S_op, rhs, y, s):
        ys = np.sum(y * s)
        xi = 1.0 / ys
        srhs = np.sum(s * rhs)
        r1 = rhs - xi * y * srhs

        it_counter = {"n": 0}
        def cb(xk):
            it_counter["n"] += 1

        z, info = cg(S_op, r1, atol=self.cg_tol, maxiter=self.cg_maxit, callback=cb)
        if info != 0:
            raise RuntimeError(f"CG did not converge, info={info}")

        yz = np.sum(y * z)
        first_term = z - xi * s * yz
        second_term = xi * s * srhs
        return first_term + second_term, it_counter["n"]

    def _solve_B_rhs_no_rank(self, S_op, rhs):
        it_counter = {"n": 0}
        def cb(xk):
            it_counter["n"] += 1

        p, info = cg(S_op, rhs, atol=self.cg_tol, maxiter=self.cg_maxit, callback=cb)
        if info != 0:
            raise RuntimeError(f"CG did not converge, info={info}")
        return p, it_counter["n"]

    # ---------- Line search (Armijo + strong Wolfe) ----------
    def wolfe_line_search(self, x, d, fval, gval, k):
        gTd = gval @ d
        test_cur = self.c2 * gTd
        assert gTd < 0, "direction must be descent."

        stepsize = 1.0
        f_new = fval
        while True:
            x_new = x + stepsize * d
            alpha, _, beta = self.unpack(x_new)
            T = self._tau(alpha, beta)
            f_new = self.f(alpha, beta, T)
            g_new = self.grad(T)
            # f_new, g_new, _ = problem.dual_obj_grad(x_new)
            # Armijo
            if f_new <= fval + self.c1 * stepsize * gTd:
                # 近似曲率：g(x+αd)^T d >= c2 g(x)^T d
                gnewd = g_new @ d
                diff = gnewd - test_cur
                if k >= 10 and gnewd >= test_cur:
                    return stepsize
                if k < 10:
                    return stepsize
            stepsize *= 0.5
            if stepsize < 1e-12:
                break
        return stepsize

    def _line_search(self, x, d, fval, gval):
        # 先尝试 SciPy 的强 Wolfe
        try:
            alpha, fc, gc, f_new, g_new, _ = line_search(
                f=self.f, fprime=self.g, xk=x, pk=d, gfk=gval,
                c1=self.c1, c2=self.c2)
            if alpha is not None and alpha > 0:
                return alpha
        except Exception:
            pass

        # 回退到简单 Armijo 回溯
        stepsize = 1.0
        phi0 = fval
        gtd = gval @ d
        while True:
            alpha, _, beta = self.unpack(x + stepsize * d)
            T = self._tau(alpha, beta)
            if self.f(alpha, beta, T) <= phi0 + self.c1 * stepsize * gtd:
                break
            stepsize *= 0.5
            if stepsize < 1e-12:
                break
        return stepsize

    # ---------- Main（Algorithm 1） ----------
    def optimize(self, tol, max_iter, time_max=np.inf, verbose=True):
        # initialization
        x = np.zeros(self.n + self.m - 1, dtype=float)
        alpha, beta, T = self.recover_plan(x)
        f0 = self.f(alpha, beta, T)
        g0 = self.grad(T)
        g0norm = np.linalg.norm(g0)
        tau = min(self.tau_max, g0norm)  # (Algorithm 1, line 4)

        t_total = 0.0
        # record initial
        self.record(t_total=t_total, cg_iter=0, gnorm=g0norm, X=T, cg_time=0.0)

        # first iteration
        t_start = timeit.default_timer()
        # construct Ω0 and HΩ0（Algorithm 1, lines 1–3）
        mask = self._omega_star_union(T[:, :-1], self.rho)
        S_op = self._make_HOmega_shifted_op(T, mask, tau)

        # first d0 = -(HΩ0 + τ0 I)^{-1} g0（Algorithm 1, line 5）
        # d, _ = cg(S_op, -g0, atol=1e-10, maxiter=400)
        cg_start_time = timeit.default_timer()
        d, cg_iter = self._solve_B_rhs_no_rank(S_op, -g0)
        cg_end_time = timeit.default_timer()
        cg_time = cg_end_time - cg_start_time

        # update x1（Algorithm 1, line 6）
        # a0 = self._line_search(x, d, f0, g0)
        a0 = self.wolfe_line_search(x, d, f0, g0, 0)
        x_prev = x.copy()
        x = x + a0 * d
        g_prev = g0.copy()

        alpha, beta, T = self.recover_plan(x)
        fval = self.f(alpha, beta, T)
        gval = self.grad(T)
        gvalnorm = np.linalg.norm(gval)

        t_end = timeit.default_timer()
        t_total += (t_end - t_start)

        # first iteration record
        self.record(t_total=t_total, cg_iter=cg_iter, gnorm=gvalnorm, X=T, cg_time=cg_time)

        if verbose:
            print(f"iter = 0, grad_norm = {gvalnorm:2.4e}, "
                  f"row_resid = {self.history['row_resid'][-1]:2.2e}, col_resid = {self.history['col_resid'][-1]:2.2e}, "
                  f"time = {t_total:3.2f}")
        # main loop（Algorithm 1, lines 7–20）
        for k in range(max_iter):

            t_start = timeit.default_timer()
            if gvalnorm < tol or t_total > time_max:  # (line 9–11)
                break

            # (Algorithm 1, line 12）
            if np.linalg.norm(gval) < np.linalg.norm(g_prev):
                self.rho = max(self.rho_min, 0.99 * self.rho)
            else:
                self.rho = min(self.rho_max, 1.10 * self.rho)

            # update HΩk（Algorithm 1, line 13）
            alpha, beta, T = self.recover_plan(x)
            mask = self._omega_star_union(T[:, :-1], self.rho)
            tau = min(self.tau_max, np.linalg.norm(gval))          # (line 17)
            S_op = self._make_HOmega_shifted_op(T, mask, tau)

            s = x - x_prev
            y = gval - g_prev

            use_lr = (y @ s) > 1e-6 * (np.linalg.norm(y) ** 2)
            cg_start_time = timeit.default_timer()
            if use_lr:
                # u = y
                # v = (HΩ + τ I) s
                # v = S_op @ s
                # a = 1.0 / (y @ s)
                # b = -1.0 / (s @ v)
                # Solve：d_k = -B^{-1} g_k，where B = S + a uuᵀ + b vvᵀ
                # d = - self._solve_B_rhs1(S_op, gval, u, v, a, b, cg_tol=1e-7, cg_maxit=400)
                d, cg_iter = self._solve_B_rhs2(S_op, -gval, y, s)
            else:
                # no low rank
                d, cg_iter = self._solve_B_rhs_no_rank(S_op, -gval)
                # d, _ = cg(S_op, -gval, tol=1e-10, maxiter=400)
            cg_end_time = timeit.default_timer()
            cg_time = cg_end_time - cg_start_time

            # line search and update
            # a_k = self._line_search(x, d, fval, gval)
            a_k = self.wolfe_line_search(x, d, fval, gval, k)
            # print(a_k)
            x_next = x + a_k * d
            x_prev = x
            g_prev = gval
            x = x_next

            alpha, beta, T = self.recover_plan(x)
            fval = self.f(alpha, beta, T)
            gval = self.grad(T)
            gvalnorm = np.linalg.norm(gval)

            t_end = timeit.default_timer()
            t_total += (t_end - t_start)
            self.record(t_total=t_total, cg_iter=cg_iter, gnorm=gvalnorm, X=T, cg_time=cg_time)

            if verbose and (k % 10 == 0):
                print(f"iter = {k}, grad_norm = {gvalnorm:2.4e}, "
                      f"row_resid = {self.history['row_resid'][-1]:2.2e}, col_resid = {self.history['col_resid'][-1]:2.2e}, "
                      f"time = {t_total:3.2f}")
        return T


# # # ========== 使用示例 ==========
if __name__ == "__main__":
    from GurobiSolver import *
    import matplotlib.pyplot as plt
    import os

    m, n = 4, 5
    eta = 1e-3
    opt = None
    # np.random.seed(10)
    a = np.random.rand(m)
    a = a / np.sum(a)  # [0.54100398 0.01455541 0.44444061]
    # print("a is ", a)
    b = np.random.rand(n)
    b = b / np.sum(b)  # [0.44833981 0.29847674 0.13459504 0.11858842]
    # print("b is ", b)
    C = np.random.rand(m, n)
    C /= np.max(C)

    # guro = Gurobisolver(C, a, b)
    # gt = guro.Gurobi_Solver_original()
    # opt = np.sum(C * gt)

    # m = n = 10000
    # cost_matrix_norm = "Uniform"  # "Square", "Uniform", "Absolute"
    # seed = 41
    # RESULT_ROOT = f"Results/alg_compare_m{m}_n{n}_rng{seed}"
    # RESULT_ROOT = os.path.join(RESULT_ROOT, cost_matrix_norm)  # 专门放 a,b,opt
    # opt_cache_path = os.path.join(RESULT_ROOT, "opt_ab.npz")
    # cache = np.load(opt_cache_path, allow_pickle=False)
    # a = cache["a"]
    # b = cache["b"]
    # opt = float(cache["opt"])
    #
    # # ===== cost matrix =====
    # if cost_matrix_norm == "Square":
    #     J = np.arange(n)
    #     C = (J[None, :] - J[:, None]) ** 2
    #     C = C / np.max(C)
    # elif cost_matrix_norm == "Uniform":
    #     np.random.seed(seed)
    #     C = np.random.uniform(0, 1, size=(m, n))
    #     C = C / np.max(C)
    # elif cost_matrix_norm == "Absolute":
    #     J = np.arange(n)
    #     C = np.abs(J[None, :] - J[:, None])
    #     C = C / np.max(C)

    solver = SPLRSolver(C=C, eta=1e-3, a=a, b=b, obj_truth=opt)
    X = solver.optimize(tol=1e-10, max_iter=1000, verbose=True)

    plt.plot(solver.history["abs_err"], label="abs_err")
    plt.plot(solver.history["grad_norm"], label="gnorm")
    plt.yscale('log')
    plt.show()

#     # 可恢复对应的传输矩阵（正则化解）
#     T_star = np.exp((alpha[:, None] + beta[None, :] - C) / eta)
#     # 验证边缘约束（近似）
#     print("row sums err:", np.max(np.abs(T_star.sum(axis=1) - a)))
#     print("col sums err:", np.max(np.abs(T_star.sum(axis=0) - b)))
#     print("cost err:", np.abs(np.sum(C * T_star) - opt))
#
