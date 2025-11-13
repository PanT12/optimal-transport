import matplotlib.pyplot as plt
from Alg.Algorithm_Import import *


def plot_from_loaded(result_root, data, ylog=True, show_bands=True, title_prefix="", title_suffix="",
                     downsample_map=None, default_stride=1, solid_algos=None):
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
    alg_eta = sorted(data.keys())
    cmap = plt.cm.get_cmap("tab10", len(eta_list))
    color_map = {str(eta): cmap(i) for i, eta in enumerate(eta_list)}

    # —— 新增：线型选择器 ——
    if isinstance(solid_algos, str):
        solid_set = {solid_algos}
    elif solid_algos is None:
        solid_set = set()
    else:
        solid_set = set(solid_algos)

    def ls(name: str) -> str:
        if name == "BISN": return "-"
        elif name == "PINS": return "--"
        elif name == "Sinkhorn": return ":"
        # return "-" if name in solid_set else "--"

    # ========== 图1：误差 vs 迭代 ==========
    plt.figure(figsize=(8, 5))
    for algname_eta in alg_eta:
        name, eta = algname_eta.split("_")
        stride = int(downsample_map.get(name, default_stride))
        dfs = [sanitize_df(d) for d in data[algname_eta]]

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
        plt.plot(x, mean_err, color=color_map[eta], linestyle=ls(name), label=name + r'$(\eta=$' + eta + ")")
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
    plt.savefig(result_root + f"/eta_compare_iter_{title_prefix}_{title_suffix}.pdf")
    plt.show()

    # ========== 图2：误差 vs 时间 ==========
    plt.figure(figsize=(8, 5))
    for algname_eta in alg_eta:
        name, eta = algname_eta.split("_")
        # stride = int(downsample_map.get(name, default_stride))
        dfs = [sanitize_df(d) for d in data[algname_eta]]

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

        plt.plot(mean_t, mean_e, color=color_map[eta], linestyle=ls(name), label=name + r'$(\eta=$' + eta + ")")

    plt.xlabel("Time (s)")
    plt.xlim(left=0, right=None)
    plt.ylabel("Gap")
    if ylog:
        plt.yscale("log")
    plt.grid(alpha=0.3)
    plt.legend(loc="upper right")
    plt.title(f"{title_prefix} {title_suffix}")
    plt.tight_layout()
    plt.savefig(result_root + f"/eta_compare_time_{title_prefix}_{title_suffix}.pdf")
    plt.show()


# ===== 固定超参 =====
eta_list = [1e-2, 1e-3]
m = n = 1000
cost_matrix_norm = "Uniform"  # "Square", "Uniform"
time_max = np.inf
tol = 1e-11
seed = 41
rng = np.random.default_rng(seed)
RESULT_ROOT = f"Results/eta_compare_m{m}_n{n}"
RESULT_ROOT = os.path.join(RESULT_ROOT, cost_matrix_norm)
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

# DATA_path = os.path.join(RESULT_ROOT, "0.01", f"{0.01}_trial001.csv")
# if not os.path.exists(DATA_path):
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

ALGORITHMS = {
    "BISN": run_BISN,
    "PINS": run_PINS,
    "Sinkhorn": run_Sinkhorn,
}

for eta in eta_list:
    print(f" Running eta={eta}...")
    for alg_name, fn in ALGORITHMS.items():
        DATA_path = os.path.join(RESULT_ROOT, f"{alg_name}_{eta}.csv")
        if not os.path.exists(DATA_path):
            print(f" Running {alg_name}...")
            X, history, eta_val = fn(C, a, b, eta=eta, tol=1e-11, time_max=10000.0, opt=opt)
            df = pd.DataFrame({
                "algo": alg_name,
                "data_name": cost_matrix_norm,
                "iter": np.arange(len(history['time']), dtype=int),
                "opt": [opt] * len(history.get("time", [])),
                "eta": [eta_val] * len(history.get("time", [])),
            })
            for key in history:
                if not history[key]:
                    history[key] = [None] * len(history['time'])
            df = pd.concat([df, pd.DataFrame(history)], axis=1)

            out_path = os.path.join(RESULT_ROOT, f"{alg_name}_{eta}.csv")
            df.to_csv(out_path, index=False)
            print(f"  saved -> {out_path}")


# ==== 用法：后续任何时候直接读取并画图 ====
loaded = load_results(RESULT_ROOT)
# algo = 'Sinkhorn'
# loaded = {algo: loaded[algo]}
print(f"Loaded results for algorithms: {list(loaded.keys())}")

plot_from_loaded(
    RESULT_ROOT,
    loaded,
    ylog=True,
    show_bands=True,
    title_prefix=cost_matrix_norm,
    title_suffix=f"(m={m}, n={n})",
    default_stride=1,
    solid_algos="BISN",
)
