import ot
from Alg.EOT_solver import *
import timeit
import os


def _round_to_marginals(X, r, c):
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


# ===== 固定超参 =====
m = n = 1000
cost_matrix_norm = "Uniform"  # "Square" or "Uniform"
time_max = 10000.0
eta = 1e-4 if cost_matrix_norm == "Square" else 1e-3
tol = 1e-7
seed = 41
rng = np.random.default_rng(seed)
RESULT_ROOT = f"Results/first_stage_effect_m{m}_n{n}"
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

opt_cache_path = os.path.join(RESULT_ROOT, "opt_ab.npz")

if os.path.exists(opt_cache_path):
    cache = np.load(opt_cache_path, allow_pickle=False)
    a = cache["a"]
    b = cache["b"]
    opt = float(cache["opt"])
    print(f"  [cache] loaded a,b,opt from {opt_cache_path}")
else:
    # 生成一次随机 a,b
    a = rng.random(m); a /= a.sum()
    b = rng.random(n); b /= b.sum()

    # 用 Gurobi 只为计算 opt（统一误差口径）
    guro = Gurobisolver(C, a, b)
    gt = guro.Gurobi_Solver_original()

    print("row difference is ", np.linalg.norm(a - gt.sum(axis=1)))
    print("col difference is ", np.linalg.norm(b - gt.sum(axis=0)))
    opt = float(np.sum(C * gt))
    print("  [Gurobi] computed opt =", opt)

    # 缓存 a,b,opt，方便复现与跳过 Gurobi
    np.savez(opt_cache_path, a=a, b=b, opt=opt)
    print(f"  [cache] saved a,b,opt to {opt_cache_path}")


# m, n = 40, 50
# # np.random.seed(10)
# a = np.random.rand(m)
# a = a / np.sum(a)  # [0.54100398 0.01455541 0.44444061]
# # print("a is ", a)
# b = np.random.rand(n)
# b = b / np.sum(b)  # [0.44833981 0.29847674 0.13459504 0.11858842]
# # print("b is ", b)
# C = np.random.rand(m, n)
#
# guro = Gurobisolver(C, a, b)
# gt = guro.Gurobi_Solver_original()
# opt = np.sum(C * gt)
#
# print("Gurobi optimal value is ", opt)
# # print("solution is \n", gt)

# our
solver = BISNsolver_EOT(C, 1e-3, a, b, skip_first_stage=True)
X = solver.optimize(tol=1e-9)
rowdiff = np.linalg.norm(a - X.sum(axis=1))
coldiff = np.linalg.norm(b - X.sum(axis=0))
print("row difference before rounding is ", rowdiff)
print("col difference before rounding is ", coldiff)

# KL-regularized OT
start1 = timeit.default_timer()
X1 = ot.smooth.smooth_ot_dual(a, b, C, 1e-3, reg_type='kl', stopThr=1e-9, numItermax=3000)
end1 = timeit.default_timer()
t1 = end1 - start1
rowdiff1 = np.linalg.norm(a - X1.sum(axis=1))
coldiff1 = np.linalg.norm(b - X1.sum(axis=0))
print("row difference before rounding is ", rowdiff)
print("col difference before rounding is ", coldiff)
print("KL-regularized OT time is ", t1)

# Quadratically regularized OT
start2 = timeit.default_timer()
X2 = ot.smooth.smooth_ot_dual(a, b, C, 1e-3, reg_type='l2', stopThr=1e-9, numItermax=3000)
end2 = timeit.default_timer()
t2 = end2 - start2
rowdiff2 = np.linalg.norm(a - X2.sum(axis=1))
coldiff2 = np.linalg.norm(b - X2.sum(axis=0))
print("row difference before rounding is ", np.linalg.norm(a - X2.sum(axis=1)))
print("col difference before rounding is ", np.linalg.norm(b - X2.sum(axis=0)))
print("Quadratically regularized OT time is ", t2)


# Sparsity-constrained optimal transport
start3 = timeit.default_timer()
X3 = ot.smooth.smooth_ot_dual(a, b, C, 1e-3, reg_type='sparsity_constrained', max_nz=2, stopThr=1e-9, numItermax=2000)
end3 = timeit.default_timer()
t3 = end3 - start3
rowdiff3 = np.linalg.norm(a - X3.sum(axis=1))
coldiff3 = np.linalg.norm(b - X3.sum(axis=0))
print("row difference before rounding is ", np.linalg.norm(a - X3.sum(axis=1)))
print("col difference before rounding is ", np.linalg.norm(b - X3.sum(axis=0)))
print("Sparsity-constrained OT time is ", t3)

# N. Courty; R. Flamary; D. Tuia; A. Rakotomamonjy, “Optimal Transport for Domain Adaptation,”
# in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol.PP, no.99, pp.1-1
# Rakotomamonjy, A., Flamary, R., & Courty, N. (2015).
# Generalized conditional gradient: analysis of convergence and applications. arXiv preprint arXiv:1510.06567.
def f(x):
    return 0.5 * np.sum(x**2)
def df(x):
    return x
start4 = timeit.default_timer()
X4 = ot.optim.gcg(a, b, C, 1e-3, 1e-3, f, df, G0=None, numItermax=100, numInnerItermax=2000, stopThr=1e-09, stopThr2=1e-09)
end4 = timeit.default_timer()
t4 = end4 - start4
rowdiff4 = np.linalg.norm(a - X4.sum(axis=1))
coldiff4 = np.linalg.norm(b - X4.sum(axis=0))
print("row difference before rounding is ", np.linalg.norm(a - X4.sum(axis=1)))
print("col difference before rounding is ", np.linalg.norm(b - X4.sum(axis=0)))
print("GCG OT time is ", t4)
