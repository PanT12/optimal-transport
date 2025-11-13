from skimage import io, color
from skimage.transform import resize
import random
import torch
import numpy as np
# import ot
import sys
from pathlib import Path
from skimage import io, color
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


def load_figures(src_path, tgt_path, size_hw=(256, 256), show=True):

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
        plt.show()

    src_lab = color.rgb2lab(src)
    tar_lab = color.rgb2lab(tar)

    src_lumi = src_lab[:, :, 0]
    src_ab_img = src_lab[:, :, 1:]
    tar_lumi = tar_lab[:, :, 0]
    tar_ab_img = tar_lab[:, :, 1:]

    src_ab = np.reshape(src_ab_img, (-1, 2), order='F')
    tar_ab = np.reshape(tar_ab_img, (-1, 2), order='F')

    return src, tar, src_lab, tar_lab, src_lumi, src_ab, tar_lumi, tar_ab


def to_lab_kmeans(src_ab, tar_ab, K=128, scale_to_01=True):
    union_ab = np.vstack([src_ab, tar_ab])
    if scale_to_01:
        scaler = MinMaxScaler((0, 1))
        union_ab_scaled = scaler.fit_transform(union_ab)
    else:
        scaler = None
        union_ab_scaled = union_ab

    size = src_ab.shape[0]
    kmeans = KMeans(n_clusters=K, random_state=0, n_init="auto").fit(union_ab_scaled)
    centers = kmeans.cluster_centers_  # (K, 2)
    labels = kmeans.labels_  # (2*size,)

    labels_src = labels[:size]
    labels_tar = labels[size:]

    a = np.bincount(labels_src, minlength=K).astype(float) + 1e-3
    a /= a.sum()
    b = np.bincount(labels_tar, minlength=K).astype(float) + 1e-3
    b /= b.sum()

    diff = centers[:, None, :] - centers[None, :, :]  # (K, K, 2)
    C = np.sum(diff * diff, axis=2)
    C /= np.max(C)

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
