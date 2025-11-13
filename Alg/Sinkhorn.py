from Alg.base import *
from scipy.special import logsumexp
import warnings

class Sinkhorn(optimal_transport):
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

    def Sinkhorn_stage(self, max_iter):
        K = np.exp(-self.C / self.eta)
        Kg = np.dot(K, self.g)
        for i in range(max_iter):
            self.X = self.f[:, np.newaxis] * K * self.g
            self.f = self.a / Kg
            self.g = self.b / (np.dot(K.T, self.f))
            Kg = np.dot(K, self.g)
            self.loss.append(np.sum(self.C * self.X))

    def Sinkhorn_log_stage(self, max_iter):
        for i in range(max_iter):
            f1 = self.f[:, np.newaxis]
            self.X = np.exp((-self.C + f1 + self.g) / self.eta - np.ones((self.m, self.n)))
            row_sum = np.sum(self.X, axis=1)
            self.f += self.eta * (np.log(self.a) - np.log(row_sum))

            f1 = self.f[:, np.newaxis]
            self.X = np.exp((-self.C + f1 + self.g) / self.eta - np.ones((self.m, self.n)))
            column_sum = np.sum(self.X, axis=0)
            self.g += self.eta * (np.log(self.b) - np.log(column_sum))
            # if np.linalg.norm(f - f_prev, ord=1) < tol and np.linalg.norm(g - g_prev, ord=1) < tol:
            #     break
            self.loss.append(np.sum(self.C * self.X))

    def dual_obj(self, f, g):
        term1 = np.sum(self.a * f)
        term2 = np.sum(self.b * g)
        f1 = f[:, np.newaxis] * np.ones((1, self.n))
        g1 = g * np.ones((self.m, 1))
        X = np.exp((-self.C + f1 + g1) / self.eta - np.ones((self.m, self.n)))
        term3 = np.sum(X) * self.eta
        return term1 + term2 - term3

    def line_search_Armijo(self, direction, sigma=0.7, gamma=0.5):
        alpha = 2.0
        iter_num = 0
        cur_obj = self.dual_obj(self.f, self.g)
        cur_grad_f = self.a - np.sum(self.X, axis=1, keepdims=False)
        cur_grad_g = self.b - np.sum(self.X, axis=0, keepdims=False)
        while iter_num <= 10:
            new_f = self.f + alpha * direction[:self.m]
            new_g = self.g + alpha * direction[self.m:]
            new_dual_obj = self.dual_obj(new_f, new_g)
            inner_prod = np.sum(cur_grad_f * direction[:self.m]) + np.sum(cur_grad_g * direction[self.m:])
            if new_dual_obj - cur_obj < gamma * alpha * inner_prod:
                alpha *= sigma
            else:
                return alpha

            iter_num += 1
        return alpha


def sinkhorn_knopp(
    a,
    b,
    M,
    reg,
    numItermax=1000,
    stopThr=1e-9,
    verbose=False,
    log=False,
    warn=True,
    warmstart=None,
    **kwargs,
):

    if len(a) == 0:
        a = np.full((M.shape[0],), 1.0 / M.shape[0])
    if len(b) == 0:
        b = np.full((M.shape[1],), 1.0 / M.shape[1])

    # init data
    dim_a = len(a)
    dim_b = b.shape[0]

    if len(b.shape) > 1:
        n_hists = b.shape[1]
    else:
        n_hists = 0

    if log:
        log = {"err": []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if warmstart is None:
        if n_hists:
            u = np.ones((dim_a, n_hists)) / dim_a
            v = np.ones((dim_b, n_hists)) / dim_b
        else:
            u = np.ones(dim_a) / dim_a
            v = np.ones(dim_b) / dim_b
    else:
        u, v = np.exp(warmstart[0]), np.exp(warmstart[1])

    K = np.exp(M / (-reg))

    Kp = (1 / a).reshape(-1, 1) * K

    err = 1
    for ii in range(numItermax):
        uprev = u
        vprev = v
        KtransposeU = np.dot(K.T, u)
        v = b / KtransposeU
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
            if n_hists:
                tmp2 = np.einsum("ik,ij,jk->jk", u, K, v)
            else:
                # compute right marginal tmp2= (diag(u)Kdiag(v))^T1
                tmp2 = np.einsum("i,ij,j->j", u, K, v)
            err = np.linalg.norm(tmp2 - b)  # violation of marginal
            if log:
                log["err"].append(err)

            if err < stopThr:
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

    if n_hists:  # return only loss
        res = np.einsum("ik,ij,jk,ij->k", u, K, v, M)
        if log:
            return res, log
        else:
            return res

    else:  # return OT matrix
        if log:
            return u.reshape((-1, 1)) * K * v.reshape((1, -1)), log
        else:
            return u.reshape((-1, 1)) * K * v.reshape((1, -1))


if __name__ == "__main__":
    import os
    import timeit
    m = n = 5000
    cost_matrix_norm = "Square"  # "Square" or "Uniform"

    seed = 41
    rng = np.random.default_rng(seed)
    RESULT_ROOT = f"Results/cg/Synthetic/cg_compare_m{m}_n{n}"
    print(f"Result root is {RESULT_ROOT}")
    RESULT_ROOT = os.path.join(RESULT_ROOT, str(cost_matrix_norm))
    os.makedirs(RESULT_ROOT, exist_ok=True)

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
    # ===== 1) 读取或生成 a,b，并读取或计算 opt =====
    opt_cache_path = os.path.join(RESULT_ROOT, "opt_ab.npz")

    if os.path.exists(opt_cache_path):
        cache = np.load(opt_cache_path, allow_pickle=False)
        a = cache["a"]
        b = cache["b"]
        opt = float(cache["opt"])
        print(f"  [cache] loaded a,b,opt from {opt_cache_path}")

        start = timeit.default_timer()
        X = sinkhorn_knopp(a, b, C, 1e-4, verbose=True, numItermax=5000)
        stop = timeit.default_timer()
        print(f"time: {stop - start}")
        print("Difference of row is ", np.linalg.norm(a - X.sum(axis=1)))
        print("Difference of column is ", np.linalg.norm(b - X.sum(axis=0)))
