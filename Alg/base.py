import numpy as np
# import torch
from scipy import sparse
import scipy.stats as ss
from scipy.special import xlogy
from scipy.special import rel_entr
# from gurobipy import *


class optimal_transport:
    """
    Parameters:
    - C: Cost matrix (m x n)
    - a: Source distribution (m,)
    - b: Target distribution (n,)
    - eta: Regularization parameter
    """
    def __init__(self, C, eta, a, b, obj_truth=None):
        self.C = C
        self.Cnorm = np.linalg.norm(C, ord='fro')
        self.K = np.exp(-C / eta)
        self.logK = -C / eta
        self.eta = eta
        self.a = a
        self.m = len(a)
        self.b = b
        self.n = len(b)
        self.anorm = np.linalg.norm(a)
        self.bnorm = np.linalg.norm(b)
        self.log_a = np.log(a)
        self.log_b = np.log(b)
        self.obj_truth = obj_truth
        self.cost_past = None
        self.history = {
            "row_resid": [],
            "col_resid": [],
            "cost": [],
            "time": [],
            "rel_err": [],
            "abs_err": [],
            "rel_conv": [],
            "grad_norm": [],
            "cg_iter": [],
            "cg_time":[],
            "kkt_err": [],
            "Delta_p": [],
            "Delta_d": [],
            "Delta_c": [],
            "newton_iter":[],
        }

    def _residuals(self, X):
        # residuals of constraints
        row_resid = np.linalg.norm(X.sum(axis=1) - self.a)
        col_resid = np.linalg.norm(X.sum(axis=0) - self.b)
        return row_resid, col_resid

    def record(self, t_total, cg_iter, gnorm, X, newton_iter=None, cg_time=None):
        self.history["time"].append(t_total)

        row_resid, col_resid = self._residuals(X)
        self.history["row_resid"].append(row_resid)
        self.history["col_resid"].append(col_resid)

        X = self._round_to_marginals(X, self.a, self.b)
        cost = np.sum(self.C * X)
        self.history["cost"].append(cost)
        self.history["cg_iter"].append(cg_iter)

        if self.obj_truth is not None:
            rel_err = abs(cost - self.obj_truth) / self.obj_truth
            abs_err = abs(cost - self.obj_truth)
            self.history["rel_err"].append(rel_err)
            self.history["abs_err"].append(abs_err)

        self.history["grad_norm"].append(gnorm)
        if self.cost_past is None:
            self.history["rel_conv"].append(1.0)
        else:
            self.history["rel_conv"].append(abs(cost - self.cost_past) / abs(self.cost_past))
        self.cost_past = cost

        if newton_iter is not None:
            self.history["newton_iter"].append(newton_iter)

        if cg_time is not None:
            self.history["cg_time"].append(cg_time)

    def _round_to_marginals(self, X, r, c):
        """
        Altschuler et al. 2017, Alg. 2
        """
        # row fix
        row = X.sum(axis=1)
        scale_r = np.divide(np.minimum(r, row), np.where(row > 0, row, 1.0))
        X1 = (X.T * scale_r).T

        # col fix
        col1 = X1.sum(axis=0)
        scale_c = np.divide(np.minimum(c, col1), np.where(col1 > 0, col1, 1.0))
        X2 = X1 * scale_c

        # residuals
        row_diff = np.maximum(r - X2.sum(axis=1), 0.0)
        col_diff = np.maximum(c - X2.sum(axis=0), 0.0)
        mass = row_diff.sum()
        if mass > 0:
            X2 = X2 + np.outer(row_diff, col_diff) / mass
        return X2

    def kkt_err(self, X, g, f=None):
        if f is None:
            f = np.min(self.C - g[None, :], axis=1)
        Delta_p1 = (np.linalg.norm(X.sum(axis=0) - self.b)) / (1 + self.bnorm)
        Delta_p2 = (np.linalg.norm(X.sum(axis=1) - self.a)) / (1 + self.anorm)
        dual = self.C - np.expand_dims(f, axis=1) - np.expand_dims(g, axis=0)
        Delta_d = np.linalg.norm(np.minimum(dual, 0.0), ord='fro') / (1 + self.Cnorm)
        Delta_p = max(Delta_p1, Delta_p2)
        Delta_c = abs(np.sum(X * dual)) / (1 + self.Cnorm)
        # gap = abs(np.sum(f * self.a) + np.sum(g * self.b) - np.sum(X * self.C))/ (1 + self.Cnorm)
        kkt_err = max(Delta_p, Delta_c, Delta_d)
        self.history["kkt_err"].append(kkt_err)
        self.history["Delta_p"].append(Delta_p)
        self.history["Delta_d"].append(Delta_d)
        self.history["Delta_c"].append(Delta_c)
        return kkt_err

    def KL_divergence(self, X):
        X_feasible = self._round_to_marginals(X, self.a, self.b)
        return np.sum(rel_entr(X_feasible, X))


def congujate_gradient(X, P, c, eta, cg_iter=1000, lam=0.0):
    X_sp, P_sp = True, True
    cs = X.sum(axis=0)
    X = sparse.coo_matrix(X).tocsr()
    P = sparse.coo_matrix(P).tocsr()

    # 稀疏：建议用 CSR 提升乘法性能；稠密：保持 ndarray
    # if X_sp and not sparse.isspmatrix_csr(X):
    #     X = X.tocsr()
    # if P_sp and not sparse.isspmatrix_csr(P):
    #     P = P.tocsr()

    # X_sp, P_sp = False, False
    XT = X.T if not X_sp else X.transpose()  # 稀疏转置开销低

    def H_mv(v: np.ndarray) -> np.ndarray:
        # term1 = diag(column_sum) @ v
        term1 = cs * v  # (n,)

        # Pv = P @ v
        Pv = P.dot(v) if P_sp else np.dot(P, v)  # (m,)

        # XtPv = X^T @ (P v)
        XtPv = XT.dot(Pv) if sparse.issparse(XT) else np.dot(XT, Pv)  # (n,)

        Hv = (term1 - XtPv) / eta
        Hv = Hv + lam * v
        return Hv

    r = c.copy()
    p = r.copy()
    x = np.zeros_like(c)
    rdotr = np.dot(r, r)
    residual_tol = 1e-6
    iter = 0
    for i in range(cg_iter):
        z = H_mv(p)              # A p
        pz = np.dot(p, z)
        if pz <= 0.0:
            print("decrease", i)
            return -c

        v = rdotr / pz
        x += v * p
        r -= v * z

        new_rdotr = np.dot(r, r)
        if np.sqrt(new_rdotr) < residual_tol * lam:
            break

        mu = new_rdotr / rdotr
        p = r + mu * p
        rdotr = new_rdotr
        iter += 1

    return x, iter

def Steihaug_congujate_gradient(A, c, delta, cg_iter=4):
    # solving Ax = c
    r = np.copy(c)
    p = np.copy(c)
    x = np.zeros_like(c)
    rdotr = np.dot(r, r)
    residual_tol = min(0.5, np.sqrt(np.linalg.norm(c)))
    residual_tol = 1e-6
    for i in range(cg_iter):
        z = np.dot(A, p)
        if np.dot(p, z) <= 0:
            print("decrease", i)
            d = np.linalg.norm(delta)**2 - np.linalg.norm(x)**2 - np.linalg.norm(p)**2
            tau = d / (2 * np.dot(x, p))
            return x + tau * p
            # return -c
        v = rdotr / np.dot(p, z)
        print("v is ", v)
        x += v * p
        if np.linalg.norm(x) >= delta:
            d = np.linalg.norm(delta) ** 2 - np.linalg.norm(x) ** 2 - np.linalg.norm(p) ** 2
            tau = d / (2 * np.dot(x, p))
            return x + tau * p
        r -= v * z
        newrdotr = np.dot(r, r)
        mu = newrdotr / rdotr
        p = r + mu * p
        rdotr = newrdotr
        if np.sqrt(rdotr) < residual_tol * np.linalg.norm(c):
            # print("iter is ", i)
            break
    return x
