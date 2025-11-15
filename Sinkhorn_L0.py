from Alg.base import *
import math
import timeit
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, cg
import matplotlib.pyplot as plt
from scipy.special import logsumexp
import warnings

class BISNsolver(optimal_transport):
    def __init__(self, *args, skip_first_stage=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.cg_tol = 1e-11
        self.cg_maxit = max(self.m, self.n) * 100

        self.tol_init = 1e-4
        self.argmax = np.argmax(self.a)

        self.skip_first_stage = skip_first_stage
        # record
        # self.condition_num = []
        # self.alpha_list = []
        # self.loss_list = []
        self.gnorm = []
        self.KL = []

    def update_P_X(self, g, fixed_term):
        # normalized_g = g - np.max(g)
        # g1 = normalized_g * np.ones((self.m, 1))
        # Y = logX + (g - self.C) / self.eta
        # Y = Y - np.max(Y, axis=1, keepdims=True)
        # numerator = np.exp(Y)
        # row_sum = numerator.sum(axis=1, keepdims=True)

        exp_term = np.exp((g - np.max(g)) / self.eta)
        # numerator = np.maximum(fixed_term * exp_term, 1e-300)
        numerator = fixed_term * exp_term
        row_sum = numerator.sum(axis=1, keepdims=True)
        if np.any(row_sum == 0.0):
            warnings.warn("Updating P: A row in P has all zero entries. It is better to increase regularization parameter eta.")
            zero_indices = np.where(row_sum == 0)[0]
            numerator[zero_indices, :] = np.random.uniform(low=0.0, high=1.0, size=(len(zero_indices), self.n))
            row_sum = numerator.sum(axis=1, keepdims=True)

        P = numerator / row_sum
        newX = P * self.a[:, None]
        return P, newX

    def accelerated_gradient(self, g, fixed_term, first_stage_tol, inner_iter, stepsize=1e-1):
        # stepsize = 1e-3
        beta = 0.0
        lam, lam_past = 0.0, 0.0
        g_past = g.copy()
        iter = 0
        numerator = np.zeros((self.m, self.n))
        row_sum = np.zeros((self.m, 1))
        loss_list = []
        gnorm_list = []
        for i in range(inner_iter):
            iter += 1
            y = g + beta * (g - g_past)
            g_past = g.copy()

            exp_term = np.exp((y - np.max(y)) / self.eta)
            numerator = fixed_term * exp_term
            row_sum = numerator.sum(axis=1, keepdims=True)
            if np.any(row_sum == 0.0):
                warnings.warn("First phase: A row in P has all zero entries. It is better to increase regularization parameter eta.")
                zero_indices = np.where(row_sum == 0)[0]
                numerator[zero_indices, :] = np.random.uniform(low=0.0, high=1.0, size=(len(zero_indices), self.n))
                row_sum = numerator.sum(axis=1, keepdims=True)
            w = (self.a / row_sum.squeeze(1))[:, None]  # (m,1)

            column_sum = (w * numerator).sum(axis=0)  # (n,)
            gradient = column_sum - self.b
            g = y - stepsize * gradient
            # loss_list.append(self.dual_obj_cal_normalize(g, log_fixed_term))
            # loss_list.append(self.dual_obj_cal_normalize(g, fixed_term))
            lam_past = lam
            lam = (1.0 + math.sqrt(1.0 + 4.0 * lam_past ** 2)) / 2.0
            beta = (lam_past - 1.0) / lam
            # gnorm_list.append(np.linalg.norm(gradient))
            if np.linalg.norm(gradient) < first_stage_tol:
                break
        # P, newX = self.update_P_X(g, fixed_term)
        P = numerator / row_sum
        newX = P * self.a[:, None]
        # plt.plot(np.abs(np.array(loss_list)))
        # plt.plot(np.array(gnorm_list))
        # plt.yscale('log')
        # plt.show()
        return g, P, newX, iter

    def optimize(self, max_iter, tol, rho=None, time_max=np.inf, verbose=True, first_stage_tol=1e-1):
        # initialization
        # X = np.ones((self.m, self.n)) * self.a[:, None] / self.n
        X = np.outer(self.a, self.b)
        if self.skip_first_stage:
            P = X / self.a[:, None]
        else:
            P = np.zeros((self.m, self.n))
        g = np.zeros(self.n)
        t_total = 0.0
        self.record(t_total=t_total, cg_iter=[], gnorm=None, X=X, newton_iter=0, cg_time=[])

        for k in range(max_iter):

            t_start = timeit.default_timer()

            tol_inner = max(self.tol_init / (k + 1)**2, 1e-11)
            fixed_term = X * self.K
            log_fixed_term = np.log(X) + self.logK

            # log_fixed_term = np.full_like(fixed_term, -np.inf)
            # mask = fixed_term > 0
            # log_fixed_term[mask] = np.log(fixed_term[mask])
            # first-order method
            if self.skip_first_stage:
                iter1 = 0
            else:
                g, P, X, iter1 = self.accelerated_gradient(g, fixed_term, first_stage_tol, inner_iter=5000, stepsize=2*self.eta)
            # second-order method
            g, X, P, cg_iter, cg_time, iter2 = self.Newton_sort(g, X, P, log_fixed_term, fixed_term, tol_inner, rho_=rho, inner_max_iter=30)

            t_end = timeit.default_timer()
            t_total += t_end - t_start

            self.record(t_total=t_total, cg_iter=cg_iter, gnorm=None, X=X, newton_iter=iter2, cg_time=cg_time)
            kkt_err = self.kkt_err(X, g)
            if verbose and k % 1 == 0:
                print(
                    f"iter = {k:5d}, first iter = {iter1:5d}, second iter = {iter2:5d}, "
                    f"inner required tol = {tol_inner:2.1e}, kkt_err = {kkt_err:2.1e}, time_iter = {t_total:3.2f}"
                )
            if kkt_err < tol or t_total > time_max:
                break

        return X

    def Newton_sort(self, g, X, P, log_fixed_term, fixed_term, tol, rho_, inner_max_iter=20):
        "Second-phase of Sinkhorn-KL algorithm with Newton's method and sparsity introduction in each inner-iteration."
        check = False
        cg_iter = []
        cg_time = []
        # alpha_list = []
        # condition_num = []
        loss_list = []
        gnorm_list = []
        for i in range(inner_max_iter):

            # gradient used to calculate hessian
            column_sum = X.sum(axis=0)
            gradient = column_sum - self.b
            gnorm = np.linalg.norm(gradient)
            # gnorm_list.append(gnorm)
            # KL_div = self.KL_divergence(X)

            # print("gradient norm is", np.linalg.norm(gradient))

            if gnorm < tol and i > 0:
                KL_div = self.KL_divergence(X)
                # print("Inner iteration stops at ", i, "with gnorm", gnorm, "and KL divergence", KL_div)
                if KL_div < min(self.n, self.m) * tol:
                    return g, X, P, cg_iter, cg_time, i

            # sparsity introduce
            # t1 = timeit.default_timer()
            # rho = 100 * gnorm / max(self.m, self.n) if rho_ is None else 0.0
            rho = self.eta / (self.m * self.n) * gnorm if rho_ is None else 0.0
            _, P_sparse = self.Sparsity_introduce(P, rho)
            # calculate percentage of sparsity of P_sparse
            # sparsity_percentage = np.count_nonzero(P_sparse) / P_sparse.size

            X_sparse = P_sparse * self.a[:, None]

            # hessian
            if check:
                hessian1 = (np.diag(X_sparse.sum(axis=0)) - np.dot(X_sparse.T, P_sparse)) / self.eta
                hessian1 += np.linalg.norm(gradient) * np.eye(self.n)
                # condition_num.append(np.linalg.cond(hessian1))

            H_sv = self.sparsify_hessian(P_sparse, X_sparse)
            cg_start_time = timeit.default_timer()
            d, iter = self._solve_direction(H_sv, gradient, gnorm)
            cg_end_time = timeit.default_timer()
            cg_time.append(cg_end_time - cg_start_time)
            cg_iter.append(iter)

            # if check:
            #     print("diff is ", np.linalg.norm(np.dot(hessian1, d) + X_sparse.sum(axis=0) - self.b))
            # print("iteration numer is ", iter)
            # print("d is ", d)
            alpha, dual_obj = self.line_search_Armijo_(g, log_fixed_term, gradient, d)
            g += alpha * d
            # loss_list.append(dual_obj)
            P, X = self.update_P_X(g, fixed_term)
            # alpha_list.append(alpha)
        # self.condition_num.append(condition_num)
        # self.alpha_list.append(alpha_list)
        # self.loss_list.append(loss_list)
        # print("alpha mean is ", np.mean(alpha), "alpha std is ", np.std(alpha))
        warnings.warn("Has reached maximum inner iterations in Newton's method. "
                      "Perhaps increase inner_max_iter or increase regularization parameter eta.")
        return g, X, P, cg_iter, cg_time, inner_max_iter

    def check_matrix(self, mat):
        col_mask = (mat & ~mat.all(axis=1, keepdims=True)).any(axis=0)
        return col_mask.all()

    def Sparsity_introduce(self, P, rho):
        # thresholds = np.exp(lam / (np.maximum(self.a, 0.0))) - 1
        # mask = P > thresholds[:, None]
        # P_sparse = P * mask

        mask = P > rho
        mask[self.argmax, :] = True
        P_sparse = P * mask
        # P_sparse[self.argmax, :] += 1e-30
        row_sum = P_sparse.sum(axis=1, keepdims=True)
        if np.any(row_sum == 0.0):
            warnings.warn("Updating P: A row in P has all zero entries. It is better to increase regularization parameter eta.")
            zero_indices = np.where(row_sum == 0)[0]
            P[zero_indices, :] = np.random.uniform(low=0.0, high=1.0, size=(len(zero_indices), self.n))
            row_sum = P.sum(axis=1, keepdims=True)
        P_sparse = P_sparse / row_sum
        return mask, P_sparse

    def dual_obj_cal_normalize(self, g, log_fixed_term):
        # minimize -g^T b + η Σ_i a_i log(Σ_j K_ij exp(g_j/η))
        u = g / self.eta
        s = logsumexp(log_fixed_term + u[None, :], axis=1)  # (m,)
        obj = -np.sum(g * self.b) + self.eta * np.sum(self.a * s)
        return obj


        max_g = np.max(g)
        e = np.exp((g - max_g) / self.eta)  # (n,)
        numerator = fixed_term * e  # (m,n)
        row_sum = numerator.sum(axis=1)  # (m,)
        # obj = -g^T b + η Σ_i a_i log(row_sum_i) + max_g Σ_i a_i

        return -np.dot(g, self.b) + self.eta * np.sum(self.a * np.log(row_sum)) + max_g

    def line_search_Armijo_(self, g, log_fixed_term, cur_grad_g, direction, sigma=0.8, gamma=1e-4):
        alpha = 1.0
        iter_num = 0
        cur_obj = self.dual_obj_cal_normalize(g, log_fixed_term)
        # print("current objective value is ", cur_obj)
        inner_prod = np.dot(cur_grad_g, direction)
        new_dual_obj = 0.0
        # print("inner product is ", inner_prod)
        if inner_prod > 0:
            print("Inner product is positive with value ", inner_prod)
            assert inner_prod < 0, "Inner product should be negative"
        # cur_grad_g = self.b - np.sum(self.X, axis=0, keepdims=False)
        while iter_num <= 100:
            new_g = g + alpha * direction
            new_dual_obj = self.dual_obj_cal_normalize(new_g, log_fixed_term)
            # print("new objective value is ", new_dual_obj)
            if new_dual_obj - cur_obj > gamma * alpha * inner_prod:
                alpha *= sigma
            else:
                break
            iter_num += 1
        return alpha, new_dual_obj

    def sparsify_hessian(self, P_sparse, X_sparse):
        column_sum = X_sparse.sum(axis=0)
        X_sparse = sp.csr_matrix(X_sparse)
        P_sparse = sp.csr_matrix(P_sparse)

        def H_sv(v):
            # term1 = diag(column_sum) @ v
            term1 = column_sum * v  # (n,)

            # Pv = P @ v
            Pv = P_sparse @ v  # (m,)

            # XtPv = X^T @ (P v)
            XtPv = X_sparse.T @ Pv

            Hv = (term1 - XtPv) / self.eta
            return Hv
        return H_sv

    def _solve_direction(self, H_sv, g, lam):
        n_tot = g.size
        # 用 LinearOperator + CG
        assert H_sv is not None

        def shift_mv(v):
            return H_sv(v) + lam * v

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

# default
#         self.cg_tol = 1e-11
# tol_inner = max(self.tol_init / (k + 1) ** 2, 1e-11)
# lam = 1e-5
# now I introduce modification to lam

# ========== 使用示例 ==========
if __name__ == "__main__":
    from Alg.GurobiSolver import *
    import matplotlib.pyplot as plt
    import os
    m, n = 3, 3
    eta = 1e-2
    opt = None
    # np.random.seed(1)
    a = np.random.rand(m)
    a = a / np.sum(a)  # [0.54100398 0.01455541 0.44444061]
    # print("a is ", a)
    b = np.random.rand(n)
    b = b / np.sum(b)  # [0.44833981 0.29847674 0.13459504 0.11858842]
    # print("b is ", b)
    # C = np.random.rand(m, n)

    J = np.arange(n)
    C = (J[None, :] - J[:, None]) ** 2
    C = C / np.max(C)

    # guro = Gurobisolver(C, a, b)
    # gt = guro.Gurobi_Solver_original()
    # opt = np.sum(C * gt)


    # m = n = 5000
    # cost_matrix_norm = "Square"  # "Square", "Uniform", "Absolute"
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

    solver = BISNsolver(C=C, eta=eta, a=a, b=b, obj_truth=opt)
    X = solver.optimize(max_iter=300, tol=1e-11)
    X = solver._round_to_marginals(X, a, b)

    print("row difference is ", np.linalg.norm(X.sum(axis=1) - a))
    print("col difference is ", np.linalg.norm(X.sum(axis=0) - b))
    print("objective value is ", np.sum(C * X))

    plt.plot(solver.history["abs_err"])
    plt.yscale('log')
    plt.show()

    plt.plot(solver.history["Delta_p"], label='Delta_p')
    plt.plot(solver.history["Delta_d"], label='Delta_d')
    plt.plot(solver.history["Delta_c"], label='Delta_c')
    # plt.plot(solver.KL, label='KL')
    # plt.plot(solver.gnorm, label='gnorm')
    plt.legend()
    plt.yscale('log')
    plt.show()

    # 恢复传输计划 T*
    cost = np.sum(C * X)

    # print("column sum err is ", np.linalg.norm(X.sum(axis=0) - b))

    # print("difference between gurobi and ssns", np.linalg.norm(X - gt))
    print("difference between cost is ", np.abs(cost - opt))
