import numpy as np
import os, glob, inspect
import pandas as pd
import matplotlib.pyplot as plt


# ===== 固定超参 =====
m = n = 10000
eta = 1e-4
stepsize = 1e-2
tol = 1e-7
n_trials = 1
seed = 41
rng = np.random.default_rng(seed)
RESULT_ROOT = f"Results/alg_compare_m{m}_n{n}_rng{seed}"


# ========= 加载并画图：从 CSV 聚合结果 =========
def load_results(result_root):
    """读取 result_root 下的所有 algo 子目录 CSV，返回 {algo: list_of_dataframes}。"""
    data = {}
    for algo_dir in sorted(glob.glob(os.path.join(result_root, "*"))):
        if not os.path.isdir(algo_dir):
            continue
        algo = os.path.basename(algo_dir)
        files = sorted(glob.glob(os.path.join(algo_dir, "*.csv")))
        dfs = [pd.read_csv(f) for f in files]
        if dfs:
            data[algo] = dfs
    return data

# ==== 用法：后续任何时候直接读取并画图 ====
loaded = load_results(RESULT_ROOT)
# algo = 'Sinkhorn'
# loaded = {algo: loaded[algo]}
print(f"Loaded results for algorithms: {list(loaded.keys())}")


# ==== 额外：打印每个算法的 CG 迭代统计（如果有） ====
def cg_iter_stats_ent(algs):
    alg = loaded[algs][0]
    cg_iter = alg["cg_iter"][1:].to_numpy()
    # cg_time = alg["cg_time"][1:].to_numpy()
    # mean and std
    return cg_iter.mean()


def cg_iter_stats_ori(algs):
    alg = loaded[algs][0]
    cg_iter = alg["cg_iter"][1:].to_numpy()
    # cg_time = alg["cg_time"][1:].to_numpy()
    newton_iter = alg["newton_iter"][1:].to_numpy()
    return cg_iter.sum() / newton_iter.sum()


algos = ["SPLR", "SSNS", "our"]
for a in algos:
    if a in loaded:
        if a == "SSNS" or a == "SPLR":
            mean1 = cg_iter_stats_ent(a)
            print(f"{a}: CG iters per outer iter = {mean1:.2f}")
        elif a == "our" or a == "PINS":
            mean1 = cg_iter_stats_ori(a)
            print(f"{a}: CG iters per Newton iter = {mean1:.2f}")

