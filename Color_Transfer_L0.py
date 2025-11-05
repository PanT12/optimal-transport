from skimage import io, color
from skimage.transform import resize
import random
import torch
import numpy as np
from Alg.Algorithm_Import import Gurobisolver

import ot
import sys
from pathlib import Path

# ===== 辅助函数（最小增补，不影响原有主流程）=====
def load_figures(pair_id=None, src_path=None, tgt_path=None, size_hw=(256, 256), show=True):
    """
    载入一对图像（RGB），按给定尺寸缩放，并可视化显示。

    参数：
    - pair_id: 可选的示例编号（若提供，将使用默认演示图片路径）
    - src_path, tgt_path: 显式提供的源/目标图像路径（优先于 pair_id）
    - size_hw: 目标尺寸 (H, W)
    - show: 是否显示原始两幅图

    返回：
    - src_rgb, tar_rgb: 缩放后的 RGB numpy 数组，范围 [0,1]
    - src_lab, tar_lab: 转换到 Lab 空间后的图像
    - src_lumi, tar_lumi: L 通道 (H, W)
    - src_ab, tar_ab: ab 通道展平后的二维数组 (H*W, 2)，按列优先顺序
    """
    from skimage import io, color
    from skimage.transform import resize
    import numpy as np
    import matplotlib.pyplot as plt

    if src_path is None or tgt_path is None:
        # 使用默认 demo 图像（与现脚本顶部示例一致）
        default_src = r"C:\Users\Andy Li\OneDrive - CUHK-Shenzhen\桌面\LGU\Research\Optimal Transport\New_Sparse\OT\ot_9.12\pexelA-0.png"
        default_tgt = r"C:\Users\Andy Li\OneDrive - CUHK-Shenzhen\桌面\LGU\Research\Optimal Transport\New_Sparse\OT\ot_9.12\pexelB-0.png"
        src_path = src_path or default_src
        tgt_path = tgt_path or default_tgt

    m, n = size_hw
    src = io.imread(src_path)
    tar = io.imread(tgt_path)
    src = resize(src, (m, n))
    tar = resize(tar, (m, n))

    if show:
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(src)
        axs[1].imshow(tar)
        axs[0].set_axis_off()
        axs[1].set_axis_off()
        plt.tight_layout()

    src_lab = color.rgb2lab(src)
    tar_lab = color.rgb2lab(tar)

    src_lumi = src_lab[:, :, 0]
    src_ab_img = src_lab[:, :, 1:]
    tar_lumi = tar_lab[:, :, 0]
    tar_ab_img = tar_lab[:, :, 1:]

    # 以列优先展平到 (H*W, 2)
    src_ab = np.reshape(src_ab_img, (-1, 2), order='F')
    tar_ab = np.reshape(tar_ab_img, (-1, 2), order='F')

    return src, tar, src_lab, tar_lab, src_lumi, src_ab, tar_lumi, tar_ab


def to_lab_kmeans(src_ab, tar_ab, K=128, scale_to_01=True, random_state=0):
    """
    在 ab 空间上做 Union-KMeans，返回码本中心、权重 a/b、标签与缩放器，并构造 OT 成本矩阵 C。

    参数：
    - src_ab, tar_ab: 形状 (N,2) 的 ab 数据
    - K: 聚类中心数
    - scale_to_01: 是否先做 [0,1] 归一化再聚类
    - random_state: KMeans 随机种子

    返回：
    - centers: (K,2)
    - a: (K,) 源端归一化计数
    - b: (K,) 目标端归一化计数
    - labels_src, labels_tar: 各自像素的簇标签
    - scaler: 归一化缩放器（若 scale_to_01=False，则返回 None）
    - C: (K,K) ab 空间的 L2 距离平方成本矩阵
    """
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import MinMaxScaler

    union_ab = np.vstack([src_ab, tar_ab])
    if scale_to_01:
        scaler = MinMaxScaler((0, 1))
        union_ab_scaled = scaler.fit_transform(union_ab)
    else:
        scaler = None
        union_ab_scaled = union_ab

    size = src_ab.shape[0]
    labels = KMeans(n_clusters=K, random_state=random_state, n_init="auto").fit_predict(union_ab_scaled)
    centers = KMeans(n_clusters=K, random_state=random_state, n_init="auto").fit(union_ab_scaled).cluster_centers_

    labels_src = labels[:size]
    labels_tar = labels[size:]

    a = np.bincount(labels_src, minlength=K).astype(float) + 1e-12
    a /= a.sum()
    b = np.bincount(labels_tar, minlength=K).astype(float) + 1e-12
    b /= b.sum()

    # 成本矩阵：两两中心差的 L2 距离平方
    diff = centers[:, None, :] - centers[None, :, :]
    C = np.sum(diff * diff, axis=2)

    return centers, a, b, labels_src, labels_tar, scaler, C


def run_one_experiment(C, a, b, algo_name, registry, eta=1e-3, tol=1e-7, time_max=np.inf, opt=None):
    """
    运行一次 OT 实验，返回解、历史与标准化指标。

    参数：
    - C, a, b: OT 问题输入
    - algo_name: 算法名称（用于注册表检索）
    - registry: {name: callable}，与 synthetic_real.py 风格一致
    - eta, tol, time_max, opt: 与各算法参数一致

    返回：
    - X, history, metrics(dict): {runtime, iterations, objective}
    - maybe_eta: 若算法返回了 eta（如 our），则附带返回，否则为 None
    """
    import time
    start = time.time()
    maybe_eta = None
    if algo_name == "our":
        X, history, maybe_eta = registry[algo_name](C, a, b, eta=eta, tol=tol, time_max=time_max, opt=opt)
    else:
        X, history = registry[algo_name](C, a, b, eta=eta, tol=tol, opt=opt)
    runtime = time.time() - start

    # 标准化指标提取
    iterations = len(history.get('time', [])) if isinstance(history, dict) else None
    objective = None
    if isinstance(history, dict):
        if 'obj' in history and len(history['obj']) > 0:
            objective = float(history['obj'][-1])
        elif 'err' in history and len(history['err']) > 0:
            objective = float(history['err'][-1])

    metrics = {
        'runtime': float(runtime),
        'iterations': int(iterations) if iterations is not None else None,
        'objective': objective
    }

    return X, history, metrics, maybe_eta


def remap_colors_from_transport(X, a, centers, labels_src, labels_tar, src_rgb, scaler=None):
    """
    基于最优传输耦合矩阵 X 将源图像颜色重映射，返回合成后的 RGB 图。

    参数：
    - X: (K,K) 传输计划
    - a: (K,) 源端权重（用于归一化）
    - centers: (K,2) 聚类中心（缩放空间）
    - labels_src, labels_tar: 像素到簇的映射标签
    - src_rgb: 源 RGB 图像 (H,W,3)，范围[0,1] 或 [0,255]
    - scaler: 若 KMeans 前做了 MinMaxScaler，则提供以便 inverse_transform 回到 Lab ab 量纲

    返回：
    - synthesis_rgb: 重映射后的 RGB 图像，范围 [0,1]
    """
    import numpy as np
    from skimage import color

    # 目标端每个簇的代表颜色：使用聚类中心（与空簇回退一致，稳健）
    Avg_tar = centers.copy()

    # 混合得到源端新中心（缩放空间）
    row_sum = a.copy()
    row_sum[row_sum < 1e-12] = 1e-12
    Avg_src = (X @ Avg_tar) / row_sum[:, None]

    # 为每个源像素赋新 ab（缩放空间）
    new_ab_scaled = Avg_src[labels_src]

    # 若提供 scaler，则反缩放回 Lab ab；否则直接视为 Lab ab
    if scaler is not None:
        from sklearn.preprocessing import MinMaxScaler  # noqa: F401 (hint for type)
        new_ab = scaler.inverse_transform(new_ab_scaled)
    else:
        new_ab = new_ab_scaled

    # 合并 L 通道并重排回图像
    src_rgb_float = src_rgb.astype(np.float64)
    if src_rgb_float.max() > 1.0:
        src_rgb_float = src_rgb_float / 255.0
    src_lab = color.rgb2lab(src_rgb_float)
    H, W = src_lab.shape[0], src_lab.shape[1]
    src_lumi = src_lab[:, :, 0]

    new_img_arr = np.hstack([src_lumi.reshape(-1, 1, order='F'), new_ab])
    new_img_arr = new_img_arr.reshape(H, W, 3, order='F')
    synthesis = color.lab2rgb(new_img_arr)
    return synthesis
