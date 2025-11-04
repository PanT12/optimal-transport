from Alg.Algorithm_Import import *
import pandas as pd
import os
from itertools import chain
import ast


# ===== 固定超参 =====
m = n = 5000
cost_matrix_norm = "Square"  # "Square", "Uniform"
eta = 1e-4 if cost_matrix_norm == "Square" else 1e-3
time_max = np.inf
tol = 1e-7
seed = 41
rng = np.random.default_rng(seed)
RESULT_ROOT = f"Results/rho_compare_m{m}_n{n}"
RESULT_ROOT = os.path.join(RESULT_ROOT, cost_matrix_norm)  # 专门放 a,b,opt
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

rho_list = [0.0, None]

# ===== 1) 读取或生成 a,b，并读取或计算 opt =====
opt_cache_path = os.path.join(RESULT_ROOT, f"opt_ab.npz")

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

for rho in rho_list:
    DATA_path = os.path.join(RESULT_ROOT, f"{rho}.csv")
    if not os.path.exists(DATA_path):
        print(f" Running rho={rho}...")
        solver = BISNsolver(C, eta, a, b, opt)
        X = solver.optimize(max_iter=300, tol=1e-11, time_max=time_max, rho=rho, verbose=True)
        history = solver.history
        df = pd.DataFrame({
            "algo": "our",
            "m": int(m), "n": int(n),
            "iter": np.arange(len(history['time']), dtype=int),
            "opt": [opt] * len(history.get("time", [])),
            "eta": [1e-4] * len(history.get("time", [])),
            "rho": [rho] * len(history.get("time", [])),
        })
        for key in history:
            if not history[key]:
                history[key] = [None] * len(history['time'])
        df = pd.concat([df, pd.DataFrame(history)], axis=1)

        out_path = os.path.join(RESULT_ROOT, f"{str(rho)}.csv")
        df.to_csv(out_path, index=False)
        print(f"  saved -> {out_path}")

# ==== 用法：后续任何时候直接读取并画图 ====
loaded = load_results(RESULT_ROOT)
print(f"Loaded results for algorithms: {list(loaded.keys())}")

def cg_iter_stats_ori(algs):
    alg = loaded[algs][0]
    # cg_iter = np.array(list(chain.from_iterable(ast.literal_eval(s) for s in alg["cg_iter"][1:])))
    cg_time = np.array(list(chain.from_iterable(ast.literal_eval(s) for s in alg["cg_time"][1:])))
    total_iter = len(alg["time"].to_numpy())
    total_time = alg["time"].to_numpy()[-1]
    newton_iter = alg["newton_iter"][1:].to_numpy()
    abs_err = alg["abs_err"].to_numpy()[-1]
    return cg_time.sum(), total_iter, newton_iter.sum(), total_time, abs_err


for a in rho_list:
    if str(a) in loaded:
        cg_time, total_iter, newton_iter, total_time, gap = cg_iter_stats_ori(str(a))
        # print(f"{a}: CG iters per Newton iter = {mean_cg:.2f} ± {std_cg:.2f}")
        # print(f"{a}: CG time per Newton iter = {mean_time:.2f} ± {std_time:.2f} seconds")
        print(f"{a}: total CG time = {cg_time:.2f} seconds")
        print(f"{a}: total outer iters = {total_iter}")
        print(f"{a}: total Newton iters = {newton_iter}")
        print(f"{a}: total time = {total_time:.2f} seconds")
        print(f"{a}: final gap = {gap:.2e}")

# for latex
for a in rho_list:
    if str(a) in loaded:
        cg_time, _, newton_iter, total_time, gap = cg_iter_stats_ori(str(a))
        # print(f"{a}: CG iters per Newton iter = {mean_cg:.1f}({std_cg:.1f})")
        # print(f"{a}: CG time per Newton iter = {mean_time:.2f}({std_time:.2f}) seconds")
        print(f"{a} & {newton_iter} & {cg_time:.2f} & {total_time:.2f} & {gap:.2e}")

