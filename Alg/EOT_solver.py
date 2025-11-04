from base import *
import timeit
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, cg
from scipy.special import logsumexp
from math import ceil


class BISNsolver_EOT(optimal_transport):
    def __init__(self, *args, skip_first_stage=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.cg_tol = 1e-11
        self.cg_maxit = max(self.m, self.n) * 100

        self.argmax = np.argmax(self.a)

        self.skip_first_stage = skip_first_stage

    def update_P_X(self, g, fixed_term):
        exp_term = np.exp((g - np.max(g)) / self.eta)
        numerator = fixed_term * exp_term
        row_sum = np.maximum(numerator.sum(axis=1, keepdims=True), 1e-300)
        P = numerator / row_sum
        newX = P * self.a[:, None]

        # Z = log_fixed_term + g[None, :]
        # lse = logsumexp(Z, axis=1, keepdims=True)  # (m,1)
        # P = np.exp(Z - lse)  # softmax rows
        # newX = P * self.a[:, None]
        return P, newX

    def accelerated_gradient(self, g, log_fixed_term, first_stage_tol, inner_iter, outer_iter, stepsize=1e-1):
        # stepsize = 1e-3
        beta = 0.0
        lam, lam_past = 0.0, 0.0
        g_past = g.copy()
        iter = 0
        P = np.zeros((self.m, self.n))
        # row_sum = np.zeros((self.m, 1))
        loss_list = []
        gnorm_list = []
        for i in range(inner_iter):
            iter += 1
            y = g + beta * (g - g_past)
            g_past = g.copy()

            u = y / self.eta
            Z = log_fixed_term + u[None, :]
            lse = logsumexp(Z, axis=1, keepdims=True)  # (m,1)
            P = np.exp(Z - lse)  # softmax rows
            column_sum = (P * self.a[:, None]).sum(axis=0)

            gradient = column_sum - self.b
            g = y - stepsize * gradient
            # loss_list.append(self.dual_obj_cal_normalize(g, log_fixed_term))
            # loss_list.append(self.dual_obj_cal_normalize(g, fixed_term))
            lam_past = lam
            lam = (1.0 + math.sqrt(1.0 + 4.0 * lam_past ** 2)) / 2.0
            beta = (lam_past - 1.0) / lam
            # gnorm_list.append(np.linalg.norm(gradient))
            gnorm = np.linalg.norm(gradient)
            if outer_iter == 0:
                if i >= 2 and gnorm < first_stage_tol:
                    break
            else:
                if gnorm < first_stage_tol:
                    break
        newX = P * self.a[:, None]
        return g, P, newX, iter

    def optimize(self, tol, rho=None, first_stage_tol=1e-1):
        # initialization
        X = np.ones((self.m, self.n))
        row_sum = np.maximum(self.K.sum(axis=1, keepdims=True), 1e-300)
        g = np.zeros(self.n)
        P = self.K / row_sum
        t_total = 0.0
        self.record(t_total=t_total, cg_iter=[], gnorm=None, X=X, newton_iter=0, cg_time=[])
        # _ = self.kkt_err(X, g)

        t_start = timeit.default_timer()

        fixed_term = self.K
        log_fixed_term = self.logK

        # first-order method
        if not self.skip_first_stage:
            g, P, X, iter1 = self.accelerated_gradient(
                g, log_fixed_term, first_stage_tol, inner_iter=5000, outer_iter=0, stepsize=2*self.eta
            )
        else:
            X = P * self.a[:, None]
        # second-order method
        g, X, P, cg_iter, cg_time, iter2 = self.Newton_sort(g, X, P, log_fixed_term, fixed_term, tol, rho_=rho, inner_max_iter=1000)

        t_end = timeit.default_timer()
        t_total += t_end - t_start

        gnorm = np.linalg.norm(self.b - X.sum(axis=0, keepdims=False))
        self.record(t_total=t_total, cg_iter=cg_iter, gnorm=gnorm, X=X, newton_iter=iter2, cg_time=cg_time)
        # _ = self.kkt_err(X, g)

        print(f"iter = {iter2}, gnorm = {gnorm:2.1e}, time = {t_total:.2f}")

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

            # print(f"iter = {i + 1:5d}, gnorm = {gnorm:2.1e}")

            if gnorm < tol:
                return g, X, P, cg_iter, cg_time, i

            # sparsity introduce
            # t1 = timeit.default_timer()
            # rho = 100 * gnorm / max(self.m, self.n) if rho_ is None else 0.0
            rho = self.eta / (self.m * self.n) * gnorm if rho_ is None else 0.0
            _, P_sparse = self.Sparsity_introduce(P, rho)
            # sparsity_percentage = np.count_nonzero(P_sparse) / P_sparse.size

            X_sparse = P_sparse * self.a[:, None]
            column_sum = X_sparse.sum(axis=0)
            X_sparse = sp.csr_matrix(X_sparse)
            P_sparse = sp.csr_matrix(P_sparse)

            # hessian
            if check:
                hessian1 = (np.diag(X_sparse.sum(axis=0)) - np.dot(X_sparse.T, P_sparse)) / self.eta
                hessian1 += np.linalg.norm(gradient) * np.eye(self.n)
                # condition_num.append(np.linalg.cond(hessian1))

            H_sv = self.sparsify_hessian(P_sparse, X_sparse, column_sum)
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
        # plt.plot(np.abs(np.array(loss_list)))
        # plt.plot(np.array(gnorm_list))
        # plt.yscale('log')
        # plt.show()
        return g, X, P, cg_iter, cg_time, inner_max_iter

    def Sparsity_introduce(self, P, rho):
        # thresholds = np.exp(lam / (np.maximum(self.a, 0.0))) - 1
        # mask = P > thresholds[:, None]
        # P_sparse = P * mask

        mask = P >= rho
        mask[self.argmax, :] = True
        P_sparse = P * mask
        # P_sparse[self.argmax, :] += 1e-30
        row_sum = np.maximum(P_sparse.sum(axis=1, keepdims=True), 1e-300) # !!!!!
        P_sparse = P_sparse / row_sum
        # positive = self.check_matrix(np.array(mask))
        # if not positive:
        #     mask[0, :] = True
        #     P_sparse[0, :] = P_sparse[0, :] * 0.95
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

    def sparsify_hessian(self, P_sparse, X_sparse, column_sum):

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


class PINSsolver_EOT(optimal_transport):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cg_tol = 1e-11
        self.cg_maxit = max(self.m, self.n) * 100

        self.alpha_list = []
        self.loss_list = []

    def internal_x(self, f, g, c_iter):
        value = (np.expand_dims(f, axis=1) + np.expand_dims(g, axis=0) - c_iter) / self.eta - 1.0
        return value

    def sparsify(self, X, rho):
        x = np.reshape(X, (X.size,))
        idx = np.argsort(x)
        num_zeros = ceil(x.size * (1 - rho))
        x[idx[0:num_zeros]] = 0.0
        X_sparse = np.reshape(x, X.shape)
        X_sparse = sp.csr_matrix(X_sparse)
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

    def sparse_newton(self, f, g, c_iter, maxit=20, tol=1e-7, rho=None):
        m = f.size
        n = g.size
        if rho is None:
            rho = 2.0 / n
        X = np.exp(self.internal_x(f, g, c_iter))
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
            gnorm = np.linalg.norm(gradient)

            # print(f"iter = {i + 1:5d}, gnorm = {gnorm:2.1e}")

            if gnorm < tol:
                return X, f, g, i+1, cg_iter, cg_time

        self.alpha_list.append(alpha_list)
        self.loss_list.append(loss_list)

        return X, f, g, maxit, cg_iter, cg_time

    def optimize(self, tol, rho=1e-1):
        # initializaion
        X = np.ones((self.m, self.n))
        f = np.zeros(self.m)
        g = np.zeros(self.n)

        t_total = 0.0
        self.record(t_total=t_total, cg_iter=[], gnorm=None, X=X, newton_iter=0, cg_time=[])
        # kkt_err = self.kkt_err(X, g, f)

        t_start = timeit.default_timer()

        # second phase
        X, f, g, iter2, itercg, timecg = self.sparse_newton(f, g, self.C, maxit=1000, tol=tol, rho=rho)

        t_end = timeit.default_timer()
        t_total += t_end - t_start

        gnorm = np.linalg.norm(self.b - X.sum(axis=0, keepdims=False))
        self.record(t_total=t_total, cg_iter=itercg, gnorm=gnorm, X=X, newton_iter=iter2, cg_time=timecg)
        # kkt_err = self.kkt_err(X, g, f)

        print(f"iter = {iter2}, gnorm = {gnorm:2.1e}, time = {t_total:.2f}")

        return X

    def lyapunov(self, f, g, internalx):
        term1 = - self.eta * (np.exp(internalx)).sum().item()
        term2 = (self.a * f).sum().item() + (self.b * g).sum().item()
        return - term1 - term2

# ========== 使用示例 ==========
if __name__ == "__main__":
    from Alg.GurobiSolver import *

    # m, n = 500, 500
    # opt = None
    # # np.random.seed(10)
    # a = np.random.rand(m)
    # a = a / np.sum(a)  # [0.54100398 0.01455541 0.44444061]
    # # print("a is ", a)
    # b = np.random.rand(n)
    # b = b / np.sum(b)  # [0.44833981 0.29847674 0.13459504 0.11858842]
    # # print("b is ", b)
    # C = np.random.rand(m, n)
    # C /= np.max(C)
    #
    # guro = Gurobisolver(C, a, b)
    # gt = guro.Gurobi_Solver_original()
    # opt = np.sum(C * gt)
    # print(gt)



    m = n = 1000
    cost_matrix_norm = "Uniform"  # "Square", "Uniform", "Absolute"
    eta = 1e-4 if cost_matrix_norm == "Square" else 1e-3
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

    # solver = SSNSSolver(C=C, eta=1e-3, a=a, b=b, obj_truth=opt)
    # X1 = solver.optimize(tol=1e-7, max_iter=3000)

    solver = PINSsolver_EOT(C=C, eta=1e-3, a=a, b=b, obj_truth=opt)
    X1 = solver.optimize(tol=1e-9)

    solver = BISNsolver_EOT(C=C, eta=1e-3, a=a, b=b, obj_truth=opt, skip_first_stage=False)
    X = solver.optimize(tol=1e-9)
    print(np.linalg.norm(X - X1))
    # X = solver._round_to_marginals(X, a, b)

