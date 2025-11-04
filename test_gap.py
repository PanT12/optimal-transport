from Alg.Algorithm_Import import load_results
import numpy as np
import os

m = n = 5000
cost_matrix_norm = "Uniform"  # "Square", "Uniform"
eta = 1e-4 if cost_matrix_norm == "Square" else 1e-3
time_max = np.inf
tol = 1e-7
seed = 41
rng = np.random.default_rng(seed)
RESULT_ROOT = f"Results/eta_compare_m{m}_n{n}"
RESULT_ROOT = os.path.join(RESULT_ROOT, cost_matrix_norm)  # 专门放 a,b,opt

loaded = load_results(RESULT_ROOT)

def cg_iter_stats_ori(algs):
    alg = loaded[algs][0]
    cost = alg["cost"][2:].to_numpy()
    return cost

cost = cg_iter_stats_ori("BISN_0.0001")
cost_diff = []
for i in range(len(cost)-1):
    cost_diff.append(cost[i+1]-cost[i])

inner_tol = []
for k in range(len(cost)):
    inner_tol.append(5000 * max(10**(-4)/(k+1)**2, 10 ** (-11)))

tol = []
for i in range(len(cost)-1):
    tol.append(1e-4 * (inner_tol[i] + inner_tol[i+1]))


for i in range(len(tol)):
    if tol < cost_diff:
        print(i)
        break

