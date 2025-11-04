from Alg.Algorithm_Import import *
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain
import ast


def plot_from_loaded(result_root, data, ylog=True, show_bands=True, title_prefix="", title_suffix="",
                     downsample_map=None, default_stride=1, solid_algos=None, figure_show=True):
    if not data:
        print("No data to plot.")
        return

    downsample_map = downsample_map or {}

    interest = 'abs_err'

    def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # 数值化并丢掉关键列中的 NaN / 非数
        for col in ["iter", "time", interest]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["iter", interest])
        # 时间列如果存在且是单步耗时，先转累计时间，再进行下采样
        if "time" in df.columns:
            t = df["time"].to_numpy(dtype=float)
            if t.size > 1 and not np.all(np.diff(t) >= 0):
                df["time"] = np.cumsum(t)
        return df.sort_values("iter")

    # 配色
    algos = data.keys()
    cmap = plt.cm.get_cmap("tab10", len(algos))
    color_map = {name: cmap(i) for i, name in enumerate(algos)}

    # —— 新增：线型选择器 ——
    if isinstance(solid_algos, str):
        solid_set = {solid_algos}
    elif solid_algos is None:
        solid_set = set()
    else:
        solid_set = set(solid_algos)

    def ls(name: str) -> str:
        return "-" if name in solid_set else "--"

    # ========== 图1：误差 vs 迭代 ==========
    plt.figure(figsize=(8, 5))
    for name in algos:
        stride = int(downsample_map.get(name, default_stride))
        # for d in data[name]:
        #     sanitize_df(d)
        dfs = [sanitize_df(d) for d in data[name]]

        # 对每个 trial 做下采样（iter/time/interest 同步取 iloc[::stride]）
        curves = []
        for d in dfs:
            if stride > 1:
                d = d.iloc[::stride, :].reset_index(drop=True)
            arr = d[interest].to_numpy(dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size > 0:
                curves.append(arr)

        if len(curves) == 0:
            continue

        Lmin = min(len(c) for c in curves)
        if Lmin == 0:
            continue
        M = np.vstack([c[:Lmin] for c in curves])

        mean_err = M.mean(0)
        std_err  = M.std(0)
        # 注意：x 轴这里用“下采样后的统一索引”，不是原始 iter 值
        # （不同 trial 的迭代上限可能不同，为了能求均值，我们用相同长度的秩序索引）
        x = np.arange(Lmin)
        plt.plot(x, mean_err, color=color_map[name], linestyle=ls(name), label="Tol1=" + name)
        # if show_bands:
        #     plt.fill_between(x, mean_err - std_err, mean_err + std_err,
        #                      color=color_map[name], alpha=0.15)

    plt.xlabel("Iteration")
    plt.ylabel("Gap")
    if ylog:
        plt.yscale("log")
    plt.grid(alpha=0.3)
    plt.legend(loc="upper right")
    plt.title(f"{title_prefix} {title_suffix}")
    plt.tight_layout()
    plt.savefig(result_root + f"/first_stage_effect_iter_{title_prefix}_{title_suffix}.pdf")
    if figure_show:
        plt.show()

    # ========== 图2：误差 vs 时间 ==========
    plt.figure(figsize=(8, 5))
    for name in algos:
        # stride = int(downsample_map.get(name, default_stride))
        dfs = [sanitize_df(d) for d in data[name]]

        times_list, errs_list = [], []
        for d in dfs:
            # if stride > 1:
            #     d = d.iloc[::stride, :].reset_index(drop=True)
            if "time" not in d.columns:
                continue
            t = d["time"].to_numpy(dtype=float)
            e = d[interest].to_numpy(dtype=float)
            finite = np.isfinite(t) & np.isfinite(e)
            t, e = t[finite], e[finite]
            if t.size == 0 or e.size == 0:
                continue
            L = min(len(t), len(e))
            if L > 0:
                times_list.append(t[:L])
                errs_list.append(e[:L])

        if len(times_list) == 0:
            continue

        Lmin = min(len(t) for t in times_list)
        T = np.vstack([t[:Lmin] for t in times_list])
        E = np.vstack([e[:Lmin] for e in errs_list])

        mean_t = T.mean(0)
        mean_e = E.mean(0)
        std_e  = E.std(0)

        plt.plot(mean_t, mean_e, color=color_map[name], linestyle=ls(name), label="Tol1=" + name)
        # plt.xlim((-0.2, 400))
        # if show_bands:
        #     plt.fill_between(mean_t, mean_e - std_e, mean_e + std_e,
        #                      color=color_map[name], alpha=0.15)

    plt.xlabel("Time (s)")
    plt.ylabel("Gap")
    if ylog:
        plt.yscale("log")
    plt.xlim(left=0, right=None)
    plt.grid(alpha=0.3)
    plt.legend(loc="upper right")
    plt.title(f"{title_prefix} {title_suffix}")
    plt.tight_layout()
    plt.savefig(result_root + f"/first_stage_effect_time_{title_prefix}_{title_suffix}.pdf")
    if figure_show:
        plt.show()


# ===== 固定超参 =====
m = n = 5000
cost_matrix_norm = "Square"  # "Square" or "Uniform"
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


test_list = [[True, None], [False, 1e-1], [False, 1e-2], [False, 1e-3]]

for skip_first_stage, first_stage_tol in test_list:
    print(f" Running {first_stage_tol}...")
    DATA_path = os.path.join(RESULT_ROOT, f"{str(first_stage_tol)}.csv")
    if not os.path.exists(DATA_path):
        solver = BISNsolver(C=C, eta=1e-4, a=a, b=b, obj_truth=opt, skip_first_stage=skip_first_stage)
        X = solver.optimize(max_iter=300, tol=1e-11, first_stage_tol=first_stage_tol)
        history = solver.history

        df = pd.DataFrame({
            "algo": "BISN",
            "m": int(m), "n": int(n),
            "iter": np.arange(len(history['time']), dtype=int),
            "opt": [opt] * len(history.get("time", [])),
            "eta": [eta] * len(history.get("time", [])),
        })
        for key in history:
            if not history[key]:
                history[key] = [None] * len(history['time'])
        df = pd.concat([df, pd.DataFrame(history)], axis=1)

        out_path = os.path.join(RESULT_ROOT, f"{str(first_stage_tol)}.csv")
        df.to_csv(out_path, index=False)
        print(f"  saved -> {out_path}")

# ==== 用法：后续任何时候直接读取并画图 ====
loaded = load_results(RESULT_ROOT)
print(f"Loaded results for algorithms: {list(loaded.keys())}")

plot_from_loaded(
    RESULT_ROOT,
    loaded,
    ylog=True,
    show_bands=True,
    title_prefix=cost_matrix_norm,
    title_suffix=f"(m={m}, n={n})",
    default_stride=1,
    solid_algos="None",
    figure_show=False
)


def cg_iter_stats_ori(algs):
    alg = loaded[algs][0]
    cg_iter = np.array(list(chain.from_iterable(ast.literal_eval(s) for s in alg["cg_iter"][1:])))
    # cg_time = np.array(list(chain.from_iterable(ast.literal_eval(s) for s in alg["cg_time"][1:])))
    newton_iter = alg["newton_iter"][1:].to_numpy()
    total_time = alg["time"].to_numpy()[-1]
    abs_err = alg["abs_err"].to_numpy()[-1]
    return cg_iter.sum(), newton_iter.sum(), total_time, abs_err


for _, a in test_list:
    if str(a) in loaded:
        cg_iter, newton_iter, total_time, abs_err = cg_iter_stats_ori(str(a))
        # print(f"{a}: CG iters per Newton iter = {mean_cg:.1f}({std_cg:.1f})")
        # print(f"{a}: CG time per Newton iter = {mean_time:.2f}({std_time:.2f}) seconds")
        print(f"{a}: total Newton iters = {newton_iter}")
        print(f"{a}: total cg iters = {cg_iter}")
        print(f"{a}: total time = {total_time:.2f} seconds")
        print(f"{a}: final gap = {abs_err:.2e}")

# for latex
for _, a in test_list:
    if str(a) in loaded:
        cg_iter, newton_iter, total_time, gap = cg_iter_stats_ori(str(a))
        # print(f"{a}: CG iters per Newton iter = {mean_cg:.1f}({std_cg:.1f})")
        # print(f"{a}: CG time per Newton iter = {mean_time:.2f}({std_time:.2f}) seconds")
        print(f"{a} & {newton_iter} & {cg_iter} & {total_time:.2f} & {gap:.2e}")



