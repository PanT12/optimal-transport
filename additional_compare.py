import ot
from Alg.EOT_solver import *
import timeit
import os
from Alg.Algorithm_Import import Gurobisolver
from Data_Sample_Import import *


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


exp = "real"
eta = 1e-3

# experiment settings
if exp == "synthetic":
    m = n = 10000
    cost_matrix_norm = "Square"  # "Square" or "Uniform"
elif exp == "real":
    experiment_id = 2  # 1,2,3
    size = 64  # only for DOTmark
    ID = {"DOTmark": {1: "Shapes", 2: "ClassicImages", 3: "MicroscopyImages"}}
else:
    raise ValueError(f"Unknown")


if exp == "synthetic":
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
elif exp == "real":
    # choose data and experiment_id
    dataset_name = "DOTmark"  # MNIST, FashionMNIST, DOTmark

    this_dir = os.path.dirname(__file__)
    default_dir = os.path.join(this_dir, "data", dataset_name)

    dataset_name == "DOTmark"
    category = ID[dataset_name][experiment_id]
    pid, qid = 2, 4
    m = n = size ** 2
    C, a, b = load_dotmark(default_dir, category, size, p_id=pid, q_id=qid, norm_type='euclidean', eps=0.01)
    RESULT_ROOT = f"Results/cg/Real/DOTmark_{category}_size{size}"
    print(f"Result root is {RESULT_ROOT}")
    os.makedirs(RESULT_ROOT, exist_ok=True)

    # ===== 1) 读取或生成 a,b，并读取或计算 opt =====
    opt_cache_path = os.path.join(RESULT_ROOT, "opt_ab.npz")

    if os.path.exists(opt_cache_path):
        cache = np.load(opt_cache_path, allow_pickle=False)
        opt = float(cache["opt"])
        print(f"  [cache] loaded a,b,opt from {opt_cache_path}")
    else:
        guro = Gurobisolver(C, a, b)
        gt = guro.Gurobi_Solver_original()

        print("row difference is ", np.linalg.norm(a - gt.sum(axis=1)))
        print("col difference is ", np.linalg.norm(b - gt.sum(axis=0)))
        opt = float(np.sum(C * gt))
        print("  [Gurobi] computed opt =", opt)

        # 缓存 a,b,opt，方便复现与跳过 Gurobi
        np.savez(opt_cache_path, opt=opt)
        print(f"  [cache] saved opt to {opt_cache_path}")

alg = [
    0,  #BISN
    # 1,  # KL-regularized
    # 2,  # L2-regularized
    # 3,  # sparsity regularized
    4,  # sinkhorn stabilized
]

# our
if 0 in alg:
    solver = BISNsolver_EOT(C, eta, a, b, skip_first_stage=True)
    X = solver.optimize(tol=1e-15)
    t = solver.history["time"][-1]
    rowdiff = np.linalg.norm(a - X.sum(axis=1))
    coldiff = np.linalg.norm(b - X.sum(axis=0))
    print("row difference before rounding is ", rowdiff)
    print("col difference before rounding is ", coldiff)
    print("BISN EOT time is ", t)


# KL-regularized OT
if 1 in alg:
    start1 = timeit.default_timer()
    X1 = ot.smooth.smooth_ot_dual(a, b, C, eta, reg_type='kl', stopThr=1e-9)
    end1 = timeit.default_timer()
    t1 = end1 - start1
    rowdiff1 = np.linalg.norm(a - X1.sum(axis=1))
    coldiff1 = np.linalg.norm(b - X1.sum(axis=0))
    print("row difference before rounding is ", rowdiff1)
    print("col difference before rounding is ", coldiff1)
    print("KL-regularized OT time is ", t1)

# Quadratically regularized OT
if 2 in alg:
    start2 = timeit.default_timer()
    X2 = ot.smooth.smooth_ot_dual(a, b, C, eta, reg_type='l2', stopThr=1e-9)
    end2 = timeit.default_timer()
    t2 = end2 - start2
    rowdiff2 = np.linalg.norm(a - X2.sum(axis=1))
    coldiff2 = np.linalg.norm(b - X2.sum(axis=0))
    print("row difference before rounding is ", rowdiff2)
    print("col difference before rounding is ", coldiff2)
    print("Quadratically regularized OT time is ", t2)


# Sparsity-constrained optimal transport
if 3 in alg:
    start3 = timeit.default_timer()
    X3 = ot.smooth.smooth_ot_dual(a, b, C, eta, reg_type='sparsity_constrained', max_nz=2)
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

if 4 in alg:
    # def f(x):
    #     return 0.5 * np.sum(x**2)
    # def df(x):
    #     return x
    # start4 = timeit.default_timer()
    # X4 = ot.optim.gcg(a, b, C, 1e-3, 1e-3, f, df, numItermax=100,  numInnerItermax=500)
    # end4 = timeit.default_timer()
    # t4 = end4 - start4
    start4 = timeit.default_timer()
    X4 = ot.sinkhorn(a, b, C, eta, method='sinkhorn_stabilized', verbose=True, numItermax=10000, stopThr=1e-15)
    end4 = timeit.default_timer()
    end4 = timeit.default_timer()
    t4 = end4 - start4
    rowdiff4 = np.linalg.norm(a - X4.sum(axis=1))
    coldiff4 = np.linalg.norm(b - X4.sum(axis=0))
    print("row difference before rounding is ", np.linalg.norm(a - X4.sum(axis=1)))
    print("col difference before rounding is ", np.linalg.norm(b - X4.sum(axis=0)))
    print("Sinkhorn stabilized OT time is ", t4)


result = {
    "KL": [rowdiff1, coldiff1, t1],
    "L2": [rowdiff2, coldiff2, t2],
    "sparsity": [rowdiff3, coldiff3, t3],
    "Sinkhorn": [rowdiff4, coldiff4, t4],
    "(E)BISN(N)": [rowdiff, coldiff, t],
}

# for latex
for key in result:
    rowdiff, coldiff, t = result[key]
    print(f"{key} & {rowdiff:.1e} & {coldiff:.1e} & {t:.2f}")
