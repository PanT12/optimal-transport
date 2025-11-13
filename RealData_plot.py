import numpy as np
import os, glob, inspect
import pandas as pd
import matplotlib.pyplot as plt
from Alg.Algorithm_Import import load_results

def plot_from_loaded(result_root, data, ylog=True, show_bands=True, title_prefix="", title_suffix="",
                     downsample_map=None, default_stride=1, solid_algos=None, figure_show=True):
    """
    聚合 CSV 并画两张图（误差-迭代，误差-时间）。
    支持按算法下采样：downsample_map = {"EG-OT":100, "Classic-Sinkhorn":50, ...}
    其余算法使用 default_stride（默认 1 = 不下采样）。

    参数
    ----
    data : dict[str, list[pd.DataFrame]]
        load_results(...) 的返回 {algo: [df_trial0, df_trial1, ...]}
    ylog : bool
        y 轴是否取对数
    show_bands : bool
        是否显示均值 ± 标准差阴影带
    title_suffix : str
        标题后缀
    downsample_map : dict[str, int] | None
        每个算法的下采样步长（每 k 个 iter 取一次）
    default_stride : int
        未在 downsample_map 指定的算法使用的步长（默认 1）
    """
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
        plt.plot(x, mean_err, color=color_map[name], linestyle=ls(name), label=name)
        # if show_bands:
        #     plt.fill_between(x, mean_err - std_err, mean_err + std_err,
        #                      color=color_map[name], alpha=0.15)

    plt.xlabel("Iteration")
    plt.ylabel("Gap")
    if ylog:
        plt.yscale("log")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.title(f"{title_prefix} {title_suffix}")
    plt.tight_layout()
    plt.savefig(result_root + f"/alg_compare_iter_{title_prefix}_{title_suffix}.pdf")
    if figure_show:
        plt.show()

    # ========== 图2：误差 vs 时间 ==========
    plt.figure(figsize=(8, 5))
    for name in algos:
        # stride = int(downsample_map.get(name, default_stride))
        dfs = [sanitize_df(d) for d in data[name]]

        times_list, errs_list = [], []
        for d in sorted(dfs):
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

        plt.plot(mean_t, mean_e, color=color_map[name], linestyle=ls(name), label=name)
        # plt.xlim((-0.2, 400))
        # if show_bands:
        #     plt.fill_between(mean_t, mean_e - std_e, mean_e + std_e,
        #                      color=color_map[name], alpha=0.15)

    plt.xlabel("Time (s)")
    plt.ylabel("Gap")
    if ylog:
        plt.yscale("log")
    plt.xlim(left=0, right=100)
    plt.grid(alpha=0.3)
    plt.legend(loc="upper right")
    plt.title(f"{title_prefix} {title_suffix}")
    plt.tight_layout()
    plt.savefig(result_root + f"/alg_compare_time_{title_prefix}_{title_suffix}.pdf")
    if figure_show:
        plt.show()

ID = {
    "MNIST": {1: [0, 1], 2: [2, 54698], 3: [3, 12580]},
    "FashionMNIST": {1: [0, 1], 2: [2, 54698], 3: [3, 12580]},
    "DOTmark": {1: "Shapes", 2: "ClassicImages", 3: "MicroscopyImages"},
}

# choose data and experiment_id
experiment_id = 2
dataset_name = "MNIST"  # MNIST, FashionMNIST, DOTmark
size = 32
figure_show = True

this_dir = os.path.dirname(__file__)
default_dir = os.path.join(this_dir, "data", dataset_name)

if dataset_name == "FashionMNIST" or dataset_name == "MNIST":
    pid, qid = ID[dataset_name][experiment_id]
    RESULT_ROOT = f"Results/{dataset_name}/pid{pid}_qid{qid}"
elif dataset_name == "DOTmark":
    category = ID[dataset_name][experiment_id]
    pid, qid = 2, 4
    RESULT_ROOT = f"Results/{dataset_name}/{category}_size{size}"
else:
    raise ValueError(f"Unknown dataset name: {dataset_name}")

os.makedirs(RESULT_ROOT, exist_ok=True)

# ==== 用法：后续任何时候直接读取并画图 ====
loaded = load_results(RESULT_ROOT)
# algo = 'Sinkhorn'
# loaded = {algo: loaded[algo]}
print(f"Loaded results for algorithms: {list(loaded.keys())}")

# 例子：Classic-Sinkhorn 每 100 个点取一次，EG-OT 每 50 个，其他算法不下采样
downsample = {
    "ExtraGrad": 20,
    "IPOT": 20,
}
plot_from_loaded(
    RESULT_ROOT,
    loaded,
    ylog=True,
    show_bands=True,
    title_prefix=f"{dataset_name}" if dataset_name != "DOTmark" else f"{category}{size}",
    title_suffix=f"(ID1={pid}, ID2={qid})",
    downsample_map=downsample,
    default_stride=1,
    solid_algos="BISN", # 只让我们的方法用实线
    figure_show=figure_show
)