import numpy as np
from .base import *
from scipy.special import logsumexp
from math import ceil, log
import timeit
from scipy.sparse.linalg import LinearOperator, cg
import scipy.sparse as sp


class PINSsolver(optimal_transport):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cg_tol = 1e-11
        self.cg_maxit = max(self.m, self.n) * 100

        self.tol_init = 1e-4

        self.alpha_list = []
        self.loss_list = []

    def internal_x(self, f, g, c_iter):
        value = (np.expand_dims(f, axis=1) + np.expand_dims(g, axis=0) - c_iter) / self.eta - 1.0
        return value

    def sinkhorn_step(self, x, f, g, c_iter, maxit=1000, tol=1e-4):
        X = np.exp(x)
        # x = self.internal_x(f, g, c_iter)
        iter1 = 0
        for iter in range(maxit):
            iter1 += 1
            f += self.eta * (self.log_a - logsumexp(x, 1))
            x = self.internal_x(f, g, c_iter)

            g += self.eta * (self.log_b - logsumexp(x, 0))
            x = self.internal_x(f, g, c_iter)

            X = np.exp(x)
            err_a = self.a - np.sum(X, axis=1)
            err_b = self.b - np.sum(X, axis=0)
            gradient = np.concatenate([err_a, err_b])
            if np.linalg.norm(gradient) < tol:
            # if np.linalg.norm(err_a) < tol and np.linalg.norm(err_b) < tol:
                # print("err_a is ", err_a, " err_b is ", err_b, "tol is ", tol)
                break

        return X, f, g, iter1

    def sparsify(self, X, rho):
        x = np.reshape(X, (X.size,))
        idx = np.argsort(x)
        num_zeros = ceil(x.size * (1 - rho))
        x[idx[0:num_zeros]] = 0.0
        X_sparse = np.reshape(x, X.shape)
        X_sparse = sp.csr_matrix(X_sparse)
        return X_sparse

        # thresholds = np.exp(1e-9 / (self.eta * self.a)) - 1
        # thre = self.a * thresholds[:, None]
        # mask = X > thre
        # X = X * mask
        # X_sparse = sp.csr_matrix(X)
        # return X_sparse

        row_sum = np.sum(X, axis=1, keepdims=True)
        scale_X = X / row_sum
        mask = scale_X > rho
        X = X * mask
        X_sparse = sp.csr_matrix(X)
        return X_sparse

    def sparsify_hessian(self, row_sum, column_sum, X, rho):
        X_sparse = self.sparsify(X, rho)
        scale = 1.0 / self.eta
        # 构造线性算子 H_δ * v（避免显式拼块）
        D_alpha = row_sum * scale                      # diag block (n,)
        D_beta  = column_sum * scale                      # diag block (m-1,)

        X_block = X_sparse * scale          # (n, m-1)
        # 用闭包提供 matvec
        def Hdelta_mv(v):
            # v = [v_a (n,); v_b (m-1,)]
            va = v[:self.m]
            vb = v[self.m:]
            out_a = D_alpha * va + X_block @ vb                  # (n,)
            out_b = X_block.T @ va + D_beta * vb                 # (m-1,)
            return np.concatenate([out_a, out_b], axis=0)

        return Hdelta_mv

    def _solve_direction(self, Hdelta_mv, g, lam):
        n_tot = g.size
        # 用 LinearOperator + CG
        assert Hdelta_mv is not None

        def shift_mv(v):
            return Hdelta_mv(v) + lam * v

        # 3) 用 callback 统计迭代次数
        it_counter = {"n": 0}

        def cb(xk):
            it_counter["n"] += 1

        Aop = LinearOperator((n_tot, n_tot), matvec=shift_mv, dtype=np.float64)
        p, info = cg(Aop, -g, atol=self.cg_tol, maxiter=self.cg_maxit, callback=cb)
        if info != 0:
            raise RuntimeError(f"CG did not converge, info={info}")
        cg_iter = it_counter["n"]
        return p, cg_iter

    def sparse_newton(self, X, f, g, c_iter, maxit=20, tol=1e-7, rho=None):
        m = f.size
        n = g.size
        if rho is None:
            rho = 2.0 / n
        Xen = np.sum(X, axis=1)
        Xtem = np.sum(X, axis=0)
        grad_f = Xen - self.a
        grad_g = Xtem - self.b

        alpha_list = []
        loss_list = []
        # print("zero element in X", np.count_nonzero(X == 0))
        cg_iter = []
        cg_time = []
        for i in range(maxit):
            grad = np.hstack((grad_f, grad_g))

            # rho = self.eta / (self.m * self.n) * np.linalg.norm(grad)
            Hdelta_mv = self.sparsify_hessian(Xen, Xtem, X, rho)
            cg_start_time = timeit.default_timer()
            d_fg, iter = self._solve_direction(Hdelta_mv, grad, np.linalg.norm(grad))
            cg_end_time = timeit.default_timer()
            cg_time.append(cg_end_time - cg_start_time)

            cg_iter.append(iter)
            d_f = d_fg[0:m]
            d_g = d_fg[m: m + n]

            if i == 0:
                internalx = self.internal_x(f, g, c_iter)
            else:
                internalx = x

            lya_current = self.lyapunov(f, g, internalx)
            alpha = 1.0
            beta = 0.5
            tol_ls = 1e-4
            inprod = (grad_f * d_f).sum().item() + (grad_g * d_g).sum().item()
            if inprod > 0:
                print("Inner product is positive with value ", inprod)
                assert inprod < 0, "Inner product should be negative"
            while True:
                new_f = f + alpha * d_f
                new_g = g + alpha * d_g
                new_internalx = self.internal_x(new_f, new_g, c_iter)
                lya_new = self.lyapunov(new_f, new_g, new_internalx)
                if lya_new <= lya_current + tol_ls * alpha * inprod:
                    break
                alpha *= beta

            step_size = alpha
            alpha_list.append(alpha)
            # print("alpha is ", alpha, "d is ", np.linalg.norm(d_fg), "gradient is ", np.linalg.norm(grad))
            # print(f"Line search time: {t2 - t1:.4f} seconds, step size = {step_size:.4f}")

            f += step_size * d_f
            g += step_size * d_g

            x = new_internalx
            X = np.exp(x)
            Xen = np.sum(X, axis=1)
            Xtem = np.sum(X, axis=0)
            grad_f = Xen - self.a
            grad_g = Xtem - self.b

            # err_a = np.linalg.norm(grad_f)
            # err_b = np.linalg.norm(grad_g)
            gradient = np.concatenate([grad_f, grad_g])

            if np.linalg.norm(gradient) < tol:
            # if err_a < tol and err_b < tol:
                KL_div = self.KL_divergence(X+1e-300)
                # print("Inner iteration stops at ", i, "with gnorm", gnorm, "and KL divergence", KL_div)
                if KL_div < min(self.n, self.m) * tol:
                    return X, f, g, i+1, cg_iter, cg_time
                # return X, f, g, iter+1, cg_total_iter

        self.alpha_list.append(alpha_list)
        self.loss_list.append(loss_list)

        return X, f, g, maxit, cg_iter, cg_time

    def optimize(self, maxit=100, tol=1e-10, rho=1e-1, time_max=np.inf, verbose=True):
        # initializaion
        # X = np.ones((self.m, self.n)) * self.a[:, None] / self.n
        X = np.outer(self.a, self.b)
        f = np.zeros(self.m)
        g = np.zeros(self.n)
        eta_log_x = self.eta * np.log(X)

        t_total = 0.0
        self.record(t_total=t_total, cg_iter=[], gnorm=None, X=X, newton_iter=0, cg_time=[])
        kkt_err = self.kkt_err(X, g, f)

        for iter in range(maxit):
            t_start = timeit.default_timer()

            tol_inner = max(self.tol_init / (iter + 1)**2, tol)
            c_iter = self.C - eta_log_x
            # first phase
            X, f, g, iter1 = self.sinkhorn_step(eta_log_x, f, g, c_iter, maxit=50000, tol=1e-3)
            # second phase
            X, f, g, iter2, itercg, timecg = self.sparse_newton(X, f, g, c_iter, maxit=20, tol=tol_inner, rho=rho)

            log_x = self.internal_x(f, g, c_iter)
            eta_log_x = self.eta * log_x

            t_end = timeit.default_timer()
            t_total += t_end - t_start

            self.record(t_total=t_total, cg_iter=itercg, gnorm=None, X=X, newton_iter=iter2, cg_time=timecg)
            kkt_err = self.kkt_err(X, g, f)

            if verbose and iter % 1 == 0:
                print(
                    f"iter = {iter:5d}, first iter = {iter1:5d}, second iter = {iter2:5d}, "
                    f"inner required tol = {tol_inner:2.1e}, kkt_err = {kkt_err:2.1e}, time_iter = {t_total:3.2f}"
                )

            if kkt_err < tol or t_total > time_max:
                break

        return X

    def lyapunov(self, f, g, internalx):
        term1 = - self.eta * (np.exp(internalx)).sum().item()
        term2 = (self.a * f).sum().item() + (self.b * g).sum().item()
        return - term1 - term2


# ========== 使用示例 ==========
if __name__ == "__main__":
    from GurobiSolver import *
    from BISN_Solver import BISNsolver
    import matplotlib.pyplot as plt
    # m, n = 300, 400
    # np.random.seed(10)
    # a = np.random.rand(m)
    # a = a / np.sum(a)  # [0.54100398 0.01455541 0.44444061]
    # # print("a is ", a)
    # b = np.random.rand(n)
    # b = b / np.sum(b)  # [0.44833981 0.29847674 0.13459504 0.11858842]
    # # print("b is ", b)
    # C = np.random.rand(m, n)
    #
    # guro = Gurobisolver(C, 0.1, a, b, 0.0)
    # gt = guro.Gurobi_Solver_original()
    # opt = np.sum(C * gt)

    m = n = 1000
    cost_matrix_norm = "Uniform"  # "Square", "Uniform", "Absolute"
    seed = 41
    RESULT_ROOT = f"Results/alg_compare_m{m}_n{n}_rng{seed}"
    RESULT_ROOT = os.path.join(RESULT_ROOT, cost_matrix_norm)  # 专门放 a,b,opt
    opt_cache_path = os.path.join(RESULT_ROOT, "opt_ab.npz")
    cache = np.load(opt_cache_path, allow_pickle=False)
    a = cache["a"]
    b = cache["b"]
    opt = float(cache["opt"])

    # ===== cost matrix =====
    if cost_matrix_norm == "Square":
        J = np.arange(n)
        C = (J[None, :] - J[:, None]) ** 2
        C = C / np.max(C)
    elif cost_matrix_norm == "Uniform":
        np.random.seed(seed)
        C = np.random.uniform(0, 1, size=(m, n))
        C = C / np.max(C)
    elif cost_matrix_norm == "Absolute":
        J = np.arange(n)
        C = np.abs(J[None, :] - J[:, None])
        C = C / np.max(C)

    solver = PINSsolver(C=C, eta=1e-4, a=a, b=b, obj_truth=opt)
    X = solver.optimize(tol=1e-11, maxit=2)

    plt.plot(solver.history["abs_err"])
    plt.yscale('log')
    plt.show()

    plt.plot(solver.history["Delta_p"], label='Delta_p')
    plt.plot(solver.history["Delta_d"], label='Delta_d')
    plt.plot(solver.history["Delta_c"], label='Delta_c')
    plt.legend()
    plt.yscale('log')
    plt.show()

    # 恢复传输计划 T*
    cost = np.sum(C * X)

    # print("difference between gurobi and ssns", np.linalg.norm(X - gt))
    print("difference between cost is ", np.abs(cost - opt))


