import matplotlib.pyplot as plt
from jinja2.optimizer import optimize

from Data_Sample_Import import *
from Alg.Algorithm_Import import *
import numpy as np
import os
import pandas as pd


# ===== 算法注册表（先只启用你实现的；其它先注释或保持占位）=====
ALGORITHMS = {
    "BISN": run_BISN,
    # "SSNS": run_ssns,
    # "ExtraGrad": run_extragradient,
    # "HOT": run_hot_halpern,
    # "PINS": run_PINS,
    # "SPLR": run_splr,
    "Sinkhorn": run_Sinkhorn,
}

ID = {
    "MNIST": {1: [0, 1], 2: [2, 54698], 3: [3, 12580]},
    "FashionMNIST": {1: [0, 1], 2: [2, 54698], 3: [3, 12580]},
    "DOTmark": {1: "Shapes", 2: "ClassicImages", 3: "MicroscopyImages"},
}

# choose data and experiment_id
experiment_id = 1
dataset_name = "MNIST"  # MNIST, FashionMNIST, DOTmark
size = 32  # only for DOTmark

this_dir = os.path.dirname(__file__)
default_dir = os.path.join(this_dir, "data", dataset_name)

if dataset_name == "FashionMNIST" or dataset_name == "MNIST":
    pid, qid = ID[dataset_name][experiment_id]
    C, a, b = load_mnist(default_dir, p_id=pid, q_id=qid, norm_type='euclidean', eps=0.01)
    RESULT_ROOT = f"Results/{dataset_name}/pid{pid}_qid{qid}"
elif dataset_name == "DOTmark":
    category = ID[dataset_name][experiment_id]
    pid, qid = 2, 4
    C, a, b = load_dotmark(default_dir, category, size, p_id=pid, q_id=qid, norm_type='euclidean', eps=0.01)
    RESULT_ROOT = f"Results/{dataset_name}/{category}_size{size}"
else:
    raise ValueError(f"Unknown dataset name: {dataset_name}")

time_max = 10000.0
eta = 1e-3
tol = 1e-7
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
from Sinkhorn_L0 import Sinkhorn_l0_Newton
solver = Sinkhorn_l0_Newton(C, 1e-3, a, b, opt)
X = solver.optimize(max_iter=300, tol=1e-11)
plt.plot(solver.history["time"], solver.history["abs_err"])
plt.yscale('log')
plt.xlabel("Time (s)")
plt.ylabel("Absolute Error")
plt.title("Sinkhorn l0 Newton Convergence")
plt.show()

df = pd.DataFrame({
    "iter": np.arange(len(solver.history['time']), dtype=int),
    "opt": [opt] * len(solver.history.get("time", [])),
    "eta": [1e-3] * len(solver.history.get("time", [])),
})
for key in solver.history:
    if not solver.history[key]:
        solver.history[key] = [None] * len(solver.history['time'])
df = pd.concat([df, pd.DataFrame(solver.history)], axis=1)

out_path = os.path.join(RESULT_ROOT, "ori.csv")
df.to_csv(out_path, index=False)
print(f"  saved -> {out_path}")

for alg_name, fn in ALGORITHMS.items():
    print(f" Running {alg_name}...")
    X, history, eta_val = fn(C, a, b, eta=eta, tol=tol, time_max=time_max, opt=opt)
    df = pd.DataFrame({
        "algo": alg_name,
        "data_name": dataset_name,
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
