from GurobiSolver import *
from HOT_Solver import HOTSolver
from PINS_solver import PINSsolver
from IPOT_Solver import IPOTSolver
from ExtraGrad_Solver import ExtraGrad
from SSNS_Solver import SSNSSolver
from SPLR_Solver import SPLRSolver
from BISN_Solver import BISNsolver
import numpy as np
import pandas as pd
import os

# ====== 已实现：你的算法 ======
def run_BISN(C, a, b, eta, tol, time_max, opt):
    solver = BISNsolver(C, eta, a, b, opt)
    X = solver.optimize(max_iter=300, tol=1e-11, time_max=time_max, verbose=True)
    return X, solver.history, solver.eta

# ====== 其它算法：留空给你填（保持相同签名 & 返回值） ======
def run_ssns(C, a, b, eta, tol, time_max, opt):
    """
    TODO: Safe & Sparse Newton (SSNS) 实现。
    """
    solver = SSNSSolver(C=C, eta=eta/10, a=a, b=b, obj_truth=opt)
    X = solver.optimize(tol=1e-9, max_iter=700, time_max=time_max, verbose=True)
    return X, solver.history, solver.eta

def run_splr(C, a, b, eta, tol, time_max, opt):
    """
    TODO: The Sparse-Plus-Low-Rank Quasi-Newton Method for Entropic-Regularized Optimal Transport 实现。
    """
    solver = SPLRSolver(C=C, eta=eta/10, a=a, b=b, obj_truth=opt)
    X = solver.optimize(tol=1e-9, max_iter=700, time_max=time_max, verbose=True)
    return X, solver.history, solver.eta

def run_extragradient(C, a, b, eta, tol, time_max, opt):
    """
    TODO: 熵正则 OT 的 Extragradient / Mirror-Prox 实现。
    """
    solver = ExtraGrad(C, eta, a, b, obj_truth=opt)
    X = solver.optimize(tol=tol, max_iter=2000, time_max=time_max, verbose=True)
    return X, solver.history, solver.eta

def run_hot_halpern(C, a, b, eta, tol, time_max, opt):
    """
    TODO: HOT (Halpern-type) 加速算法实现。
    """
    n = len(a)
    c1 = np.zeros(n, dtype=float)  # 这里就是长度 n 的零向量
    c2 = C.flatten(order='F')  # 与 demo.py 保持相同的列优先展平
    c = np.concatenate([c1, c2])
    c_Max = (c2.max() if c2.size else 1.0) + (c1.max() if c1.size else 0.0)
    if c_Max == 0:
        c_Max = 1.0
    c = c / c_Max

    solver = HOTSolver(
        a_seq=a,
        b_seq=b,
        c=c,
        m=1,
        n=n,
        cMax=c_Max,
        max_iters=20_000,
        tolerance=1e-8,
        check_freq=100,
        sigma=None,
        adjust_sigma=True,
        logging=True,
        dtype=None,  # default float64
        device=None,
        obj_truth=opt
    )
    X = solver.optimize(time_max=time_max, verbose=True)
    return X, solver.history, None


def run_PINS(C, a, b, eta, tol, time_max, opt):
    solver = PINSsolver(C, eta, a=a, b=b, obj_truth=opt)
    X = solver.optimize(maxit=100, tol=1e-11, time_max=time_max, verbose=True)
    return X, solver.history, solver.eta


def run_IPOT(C, a, b, eta, tol, time_max, opt):
    solver = IPOTSolver(C, eta=0.1, a=a, b=b, obj_truth=opt)
    X = solver.optimize(maxit=6000, tol=1e-6, time_max=time_max, verbose=True)
    return X, solver.history, solver.eta


# ========= 加载并画图：从 CSV 聚合结果 =========
def load_results(result_root):
    data = {}
    files = sorted(os.listdir(result_root))[::-1]
    for f in files:
        csv_file = os.path.join(result_root, f)
        algo, ext = os.path.splitext(f)
        if ext == '.csv':
            dfs = [pd.read_csv(csv_file)]
            data[algo] = dfs
    return data

# ===== 算法注册表（先只启用你实现的；其它先注释或保持占位）=====
ALGORITHMS = {
    "BISN": run_BISN,
    "SSNS": run_ssns,
    "ExtraGrad": run_extragradient,
    "HOT": run_hot_halpern,
    "PINS": run_PINS,
    "SPLR": run_splr,
}