from tkinter.font import names

from Alg.base import *
from scipy.special import logsumexp
import warnings
import timeit

# inexact version to solve original OT, inner solver: Sinkhorn
class SinkhornSolver(optimal_transport):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tol_init = 1e-4

    def optimize(self, tol, max_iter):
        # initialize
        X = np.outer(self.a, self.b)
        u = np.ones(self.m) / self.m
        v = np.ones(self.n) / self.n
        t_total = 0.0
        self.record(t_total=t_total, cg_iter=0, gnorm=None, X=X, first_stage_iter=0)
        self.kkt_err(X, self.eta * np.log(v), self.eta * np.log(u))

        for k in range(max_iter):

            t_start = timeit.default_timer()

            tol_inner = max(self.tol_init / (k + 1)**2, 1e-11)
            C_iter = self.C - self.eta * np.log(X)
            X, u, v, i = self.sinkhorn_knopp(u, v, C_iter, numItermax=10000, stopThr=tol_inner)

            t_end = timeit.default_timer()
            t_total += t_end - t_start

            row_sum, col_sum = X.sum(axis=1), X.sum(axis=0)
            row_diff, col_diff = row_sum - self.a, col_sum - self.b
            diff = np.concatenate([row_diff, col_diff])
            gnorm = np.linalg.norm(diff)
            self.record(t_total=t_total, cg_iter=0, gnorm=gnorm, X=X, first_stage_iter=i)
            kkt_err = self.kkt_err(X, self.eta * np.log(v), self.eta * np.log(u))
            if k % 1 == 0:
                print(
                    f"iter = {k:5d}, inner_iter = {i:5d}, gnorm = {gnorm: 2.1e}, "
                    f"inner required tol = {tol_inner:2.1e}, kkt_err = {kkt_err:2.1e}, time_iter = {t_total:3.2f}"
                )
            if kkt_err < tol:
                break

        return X


    def sinkhorn_knopp(self, u, v, M, numItermax=1000, stopThr=1e-9, verbose=False, log=False, warn=True):

        # init data
        if log:
            log = {"err": []}

        K = np.exp(M / (-self.eta))

        Kp = (1 / self.a).reshape(-1, 1) * K

        err = 1
        iter = 0
        for ii in range(numItermax):
            iter += 1
            uprev = u
            vprev = v
            KtransposeU = np.dot(K.T, u)
            v = self.b / KtransposeU
            u = 1.0 / np.dot(Kp, v)

            if (
                np.any(KtransposeU == 0)
                or np.any(np.isnan(u))
                or np.any(np.isnan(v))
                or np.any(np.isinf(u))
                or np.any(np.isinf(v))
            ):
                # we have reached the machine precision
                # come back to previous solution and quit loop
                warnings.warn("Warning: numerical errors at iteration %d" % ii)
                u = uprev
                v = vprev
                break
            if ii % 10 == 0:
                # we can speed up the process by checking for the error only all
                # the 10th iterations
                # compute right marginal tmp2= (diag(u)Kdiag(v))^T1
                tmp2 = np.einsum("i,ij,j->j", u, K, v)
                err = np.linalg.norm(tmp2 - self.b)  # violation of marginal
                if log:
                    log["err"].append(err)

                if err < stopThr:
                    X = u.reshape((-1, 1)) * K * v.reshape((1, -1))
                    KL_div = self.KL_divergence(X)
                    # print("Inner iteration stops at ", i, "KL divergence", KL_div, "and requird tol ", min(self.n, self.m) * tol)
                    if KL_div < min(self.n, self.m) * stopThr:
                        break
                if verbose:
                    if ii % 200 == 0:
                        print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
                    print("{:5d}|{:8e}|".format(ii, err))
        else:
            if warn:
                warnings.warn(
                    "Sinkhorn did not converge. You might want to "
                    "increase the number of iterations `numItermax` "
                    "or the regularization parameter `reg`."
                )
        if log:
            log["niter"] = ii
            log["u"] = u
            log["v"] = v

        if log:
            return u.reshape((-1, 1)) * K * v.reshape((1, -1)), u, v, iter, log
        else:
            return u.reshape((-1, 1)) * K * v.reshape((1, -1)), u, v, iter


# ========== 使用示例 ==========
if __name__ == "__main__":
    from GurobiSolver import *
    import matplotlib.pyplot as plt
    import os
    m, n = 500, 500
    eta = 1e-1
    opt = None
    # np.random.seed(4)
    a = np.random.rand(m)
    a = a / np.sum(a)  # [0.54100398 0.01455541 0.44444061]
    # print("a is ", a)
    b = np.random.rand(n)
    b = b / np.sum(b)  # [0.44833981 0.29847674 0.13459504 0.11858842]
    # print("b is ", b)
    C = np.random.rand(m, n)

    # J = np.arange(n)
    # C = (J[None, :] - J[:, None]) ** 2
    # C = C / np.max(C)

    guro = Gurobisolver(C, a, b)
    gt = guro.Gurobi_Solver_original()
    opt = np.sum(C * gt)

    # m = n = 1000
    # cost_matrix_norm = "Uniform"  # "Square", "Uniform", "Absolute"
    # eta = 1e-4 if cost_matrix_norm == "Square" else 1e-3
    # seed = 41
    # RESULT_ROOT = f"../Results/alg_compare_m{m}_n{n}_rng{seed}"
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

    solver = SinkhornSolver(C=C, eta=1e-2, a=a, b=b, obj_truth=opt)
    X = solver.optimize(max_iter=300, tol=1e-11)
    X = solver._round_to_marginals(X, a, b)

    print("row difference is ", np.linalg.norm(X.sum(axis=1) - a))
    print("col difference is ", np.linalg.norm(X.sum(axis=0) - b))
    print("objective value is ", np.sum(C * X))

    plt.plot(solver.history["abs_err"])
    plt.yscale('log')
    # plt.show()

    plt.plot(solver.history["Delta_p"], label='Delta_p')
    plt.plot(solver.history["Delta_d"], label='Delta_d')
    plt.plot(solver.history["Delta_c"], label='Delta_c')
    # plt.plot(solver.KL, label='KL')
    # plt.plot(solver.gnorm, label='gnorm')
    plt.legend()
    plt.yscale('log')
    # plt.show()

    # 恢复传输计划 T*
    cost = np.sum(C * X)

    # print("column sum err is ", np.linalg.norm(X.sum(axis=0) - b))

    # print("difference between gurobi and ssns", np.linalg.norm(X - gt))
    print("difference between cost is ", np.abs(cost - opt))
