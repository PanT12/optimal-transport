from Alg.Algorithm_Import import *
from Data_Sample_Import import *
from Alg.EOT_solver import BISNsolver_EOT, PINSsolver_EOT
import os
import pandas as pd
import numpy as np
from itertools import chain
import ast


def run_PINS_EOT(C, a, b, eta, tol, time_max, opt):
    solver = PINSsolver_EOT(C, eta/10, a=a, b=b, obj_truth=opt)
    X = solver.optimize(tol=tol)
    return X, solver.history, solver.eta


def run_BISN_EOT_skip1(C, a, b, eta, tol, time_max, opt):
    solver = BISNsolver_EOT(C, eta/10, a=a, b=b, obj_truth=opt, skip_first_stage=True)
    X = solver.optimize(tol=tol)
    return X, solver.history, solver.eta

def run_BISN_EOT_noskip(C, a, b, eta, tol, time_max, opt):
    solver = BISNsolver_EOT(C, eta/10, a=a, b=b, obj_truth=opt, skip_first_stage=False)
    X = solver.optimize(tol=tol)
    return X, solver.history, solver.eta


# ===== 固定超参 =====
time_max = 10000.0
eta = 1e-2
tol = 1e-9

exp = "synthetic"

# experiment settings
if exp == "synthetic":
    m = n = 10000
    cost_matrix_norm = "Uniform"  # "Square" or "Uniform"
elif exp == "real":
    experiment_id = 1  # 1,2,3
    size = 32  # only for DOTmark
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


# ===== 算法注册表（先只启用你实现的；其它先注释或保持占位）=====
ALGORITHMS = {
    "SSNS": run_ssns,
    "SPLR": run_splr,
    "PINS_EOT": run_PINS_EOT,
    "BISN_EOT_skip1": run_BISN_EOT_skip1,
    "BISN_EOT_noskip": run_BISN_EOT_noskip,
}

for alg_name, fn in ALGORITHMS.items():
    DATA_path = os.path.join(RESULT_ROOT, f"{alg_name}.csv")
    if not os.path.exists(DATA_path):
        print(f" Running {alg_name}...")
        X, history, eta_val = fn(C, a, b, eta=eta, tol=tol, time_max=time_max, opt=opt)
        df = pd.DataFrame({
            "algo": alg_name,
            "m": int(m), "n": int(n),
            "iter": np.arange(len(history['time']), dtype=int),
            "opt": [opt] * len(history.get("time", [])),
            "eta": [eta_val] * len(history.get("time", [])),
        })
        for key in history:
            if not history[key]:
                history[key] = [None] * len(history['time'])
        df = pd.concat([df, pd.DataFrame(history)], axis=1)

        out_path = os.path.join(RESULT_ROOT, f"{alg_name}.csv")
        df.to_csv(out_path, index=False)
        print(f"  saved -> {out_path}")


# ==== 用法：后续任何时候直接读取并画图 ====
loaded = load_results(RESULT_ROOT)
print(f"Loaded results for algorithms: {list(loaded.keys())}")

def cg_iter_stats_ent(algs):
    alg = loaded[algs][0]
    cg_iter = alg["cg_iter"][1:].to_numpy()
    # cg_time = alg["cg_time"][1:].to_numpy()
    total_time = alg["time"].to_numpy()[-1]
    # mean and std
    return cg_iter.mean(), cg_iter.sum(), cg_iter.std(), total_time

def cg_iter_stats_ori(algs):
    alg = loaded[algs][0]
    cg_iter = np.array(list(chain.from_iterable(ast.literal_eval(s) for s in alg["cg_iter"][1:])))
    # cg_time = np.array(list(chain.from_iterable(ast.literal_eval(s) for s in alg["cg_time"][1:])))
    # total_iter = len(alg["time"].to_numpy())
    total_time = alg["time"].to_numpy()[-1]
    # newton_iter = alg["newton_iter"][1:].to_numpy()
    # abs_err = alg["abs_err"].to_numpy()[-1]
    # return cg_time.sum(), total_iter, newton_iter.sum(), total_time, abs_err
    return cg_iter.mean(), cg_iter.sum(), cg_iter.std(), total_time


for a in ALGORITHMS.keys():
    if a in loaded:
        if a == "SSNS" or a == "SPLR":
            iter_mean, iter_sum, iter_std, time = cg_iter_stats_ent(a)
        else:
            iter_mean, iter_sum, iter_std, time = cg_iter_stats_ori(a)
        print(f"{a}: per CG iters iter = {iter_mean:.2f} ± {iter_std:.2f} with total iters = {iter_sum} and time = {time:.2f}")


# for latex
for a in ALGORITHMS.keys():
    if a in loaded:
        if a == "SSNS" or a == "SPLR":
            iter_mean, iter_sum, iter_std, time = cg_iter_stats_ent(a)
        else:
            iter_mean, iter_sum, iter_std, time = cg_iter_stats_ori(a)
        if a == "PINS_EOT":
            a = "PINS"
        elif a == "BISN_EOT_skip1":
            a = "BISN(N)"
        elif a == "BISN_EOT_noskip":
            a = "BISN(W)"
        print(f"{a} & {iter_mean:.1f}({iter_std:.0f}) & {iter_sum} & {time:.2f}")

