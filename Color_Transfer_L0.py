from skimage import io, color
from skimage.transform import resize

# 图像信息的展示-读取-预处理
# Load image and resize
m = n = 256
size = m * n

src = io.imread(r"C:\Users\Andy Li\OneDrive - CUHK-Shenzhen\桌面\LGU\Research\Optimal Transport\New_Sparse\OT\ot_9.12\pexelA-0.png")
tar = io.imread(r"C:\Users\Andy Li\OneDrive - CUHK-Shenzhen\桌面\LGU\Research\Optimal Transport\New_Sparse\OT\ot_9.12\pexelB-0.png")

src = resize(src, (m, n))
tar = resize(tar, (m, n))


# Show original two images]
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2)
axs[0].imshow(src)
axs[1].imshow(tar)
axs[0].set_axis_off()
axs[1].set_axis_off()


# Convert to lab color space
src_lab = color.rgb2lab(src)
tar_lab = color.rgb2lab(tar)


# Split luminance channel and 'ab' channels
# Remarks luminance channel will stand still
src_lumi = src_lab[:, :, 0]
src_ab = src_lab[:, :, 1:]

tar_lumi = tar_lab[:, :, 0]
tar_ab = tar_lab[:, :, 1:]


# Convert 'ab' channels to 2D NumPy arrays
import numpy as np

src_ab = np.reshape(src_ab, (-1, 2), order='F')
tar_ab = np.reshape(tar_ab, (-1, 2), order='F')
# Here we have all the points (a,b) channel value
# src_ab and tar_ab 都是以ab为列的Numpy arrays


# # 对数据进行预处理，KMeans 降低问题维度
# # Use KMeans to cluster 'ab' values
# from sklearn.cluster import KMeans

# clusters = 128
# union_ab = np.vstack([src_ab, tar_ab])
# kmeans = KMeans(n_clusters=clusters, random_state=0, n_init="auto").fit(union_ab)
# # 将 source 和 target 对应的坐标进行聚合，得到128个cluster中心
# # 这个kmeans是根据像素的ab值进行聚合，也就是根据颜色的相近程度

# # 将得到的所有cluster中心对应的 (a,b) 当中的 a 和 b 分离，并排序，为后续的'ab'平面的绘制做坐标
# sorted_a = np.array(sorted(kmeans.cluster_centers_[:, 0], reverse=True))
# sorted_b = np.array(sorted(kmeans.cluster_centers_[:, 1]))


# # Create a 'ab' plane and map each cluster center to a point
# grids = np.zeros((clusters, clusters, 2))

# for i in range(clusters):
#     for j in range(clusters):
#         grids[i][j] = [sorted_b[j], sorted_a[i]]
        
# grids = np.reshape(grids, (-1, 2), order='F')
# # 这里有128*128个格子

# cluster_to_bin = np.zeros(clusters)			# map each cluster center to a unique bin index of the created 'ab' plane
# # 这里表示准备了128个cluster center

# for i in range(clusters):
#     for j in range(grids.shape[0]):
#         if kmeans.cluster_centers_[i][0] == grids[j][1] and kmeans.cluster_centers_[i][1] == grids[j][0]:
#             cluster_to_bin[i] = j
#             break
# # 找到128个cluster center每一个在网格中的位置，即“第i个cluster在第j个格子中”

# M_src = np.zeros((clusters * clusters, size))	# pixel to bin index mapping
# M_tar = np.zeros((clusters * clusters, size))
# # 这两个变量的维度是 （128*128个格子）*（256*256个像素点）

# for i in range(size):
# 	"""
# 	For each pixel in the source image,
# 	1. extract the cluster_center it belongs to,
# 	2. obtain the bin index this cluster_center corresponds to
# 	3. the bin intensity is increased by 1
# 	"""	
# 	center_label_src = kmeans.labels_[i]         # 找到第i个像素点所属的颜色中心
# 	map_grid = cluster_to_bin[center_label_src]  # 找到该颜色中心在grid网格中对应的位置
# 	M_src[int(map_grid)][i] += 1                 # 表示找到了第i个像素点及其在grid网格中的位置
#                                                  # 该格子在第i个像素点的位置+1

# # Do same things for target image
# for j in range(size, 2 * size):
#     center_label_tar = kmeans.labels_[j]
#     map_grid = cluster_to_bin[center_label_tar]
#     M_tar[int(map_grid)][j - size] += 1
# # 以上两个for循环，实现了将每一个像素点都分配到对应的网格格子中

# from sklearn.preprocessing import normalize

# M_src_norm = np.sum(M_src, axis=1, keepdims=True)	# sum up the number of pixels assigned to each bin index 
# M_tar_norm = np.sum(M_tar, axis=1, keepdims=True)
# # 横向加和，计算出每一个网格格子中被分配到的像素点的数量

# M_src_weight = normalize(M_src_norm, norm='l1', axis=0).squeeze()	# l1-normalized
# # 我们一共有128个cluster center，现在在ab平面上有各自的归属，
# # 我们从宏观上来看，128*128的网格中，存在这128个cluster center并且每个
# # center中存有不同分布的像素点数量 
# M_tar_weight = normalize(M_tar_norm, norm='l1', axis=0).squeeze()
# # 计算每个网格中像素点的比例

# # 创建所有网格点的坐标
# a_coords = sorted_a  # (128,)
# b_coords = sorted_b  # (128,)

# # 创建所有网格点的颜色值对
# A, B = np.meshgrid(a_coords, b_coords, indexing='ij')
# points = np.column_stack([A.ravel(), B.ravel()])  # (16384, 2)

# # 使用cdist计算完整成本矩阵（推荐）
# from scipy.spatial.distance import cdist
# C_full = cdist(points, points, metric='sqeuclidean')




from sklearn.cluster import KMeans
# 1) 归一化 + KMeans (union)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler((0,1))
union_ab = np.vstack([src_ab, tar_ab])
union_ab_scaled = scaler.fit_transform(union_ab)
src_ab_scaled = union_ab_scaled[:size]
tar_ab_scaled = union_ab_scaled[size:]
# --- KMeans：用 union 做共享码本 ---
K = 128
# union_ab = np.vstack([src_ab, tar_ab])
kmeans = KMeans(n_clusters=K, random_state=0, n_init="auto").fit(union_ab_scaled)
centers = kmeans.cluster_centers_           # (K, 2)
labels = kmeans.labels_                     # (2*size,)

# --- 计算 a, b （每图在 K 个中心上的归一化计数）---
labels_src = labels[:size]
labels_tar = labels[size:]

a_counts = np.bincount(labels_src, minlength=K).astype(np.float64)
b_counts = np.bincount(labels_tar, minlength=K).astype(np.float64)

# 2) a,b（先用硬计数；若不稳再换软计数）
a = np.bincount(labels_src, minlength=K).astype(float)
a /= a.sum() + 1e-12
b = np.bincount(labels_tar, minlength=K).astype(float)
b /= b.sum() + 1e-12

# a = a_counts / (a_counts.sum() + 1e-12)     # (K,)
# b = b_counts / (b_counts.sum() + 1e-12)     # (K,)

# --- 成本矩阵 C：ab 空间的 L2 距离平方 ---
# C[i,j] = ||centers[i] - centers[j]||^2
diff = centers[:, None, :] - centers[None, :, :]   # (K, K, 2)
# diff[i, j] = [aᵢ - aⱼ, bᵢ - bⱼ]  # 从中心j到中心i的颜色向量
C = np.sum(diff * diff, axis=2)                    # (K, K)

# 到这里你已经有了：C (KxK), a (K,), b (K,)
# 可以直接喂给你的 OT 优化器（LP/PDLP/Sinkhorn/PINS/SSNS...）


import random
import torch
import numpy as np
from Sinkhorn_L0 import Sinkhorn_l0_Newton
import os, glob, inspect
import pandas as pd
from GurobiSolver import *
from HOT_Solver import HOTSolver
# from HOT_Solver_modified import HOTSolverModified
from PINS_solver import PINSsolver
from Sinkhorn_classic import SinkhornSolver
from IPOT_Solver import IPOTSolver
from ExtraGrad_Solver import ExtraGrad
from SSNS_Solver import SSNSSolver
from SPLR_Solver import SPLRSolver
import ot
import sys
from pathlib import Path


# ===== 固定超参 =====
eta = 1e-4
stepsize = 1e-2
tol = 1e-7
n_trials = 2
rng = np.random.default_rng(41)


def run_sinkhorn_l0_newton(C, a, b, eta, tol, time_max, opt):
    solver = Sinkhorn_l0_Newton(C, eta, a, b, opt)
    X = solver.optimize(max_iter= 200, tol=1e-10, time_max=time_max, verbose=True)
    return X, solver.history, solver.eta

# guro = Gurobisolver(C, 0.1, a, b, 0.0)
# gt = guro.Gurobi_Solver_original()
# opt = np.sum(C * gt)
    # ground truth 只用于计算 opt（保证误差口径一致）
X_opt = ot.emd(a, b, C)
opt = float(np.sum(C * X_opt))
print(opt)
print("gurobi finished")
print("Start SinkhornL0")

X, history, eta = run_sinkhorn_l0_newton(C, a, b, 
                     eta=1e-3, tol=tol, time_max=np.inf, opt=opt)
result = np.sum(C * X)
print(X.shape)
# K*K, 128*128 cluster*cluster
print(result,opt)

K = centers.shape[0]
N = src_ab.shape[0]   # H*W

# 1) 计算目标端每个中心的“像素平均颜色” Avg_tar ∈ R^{K×2}
Avg_tar = np.zeros((K, 2), dtype=np.float64)
b_counts = np.bincount(labels_tar, minlength=K).astype(np.float64)
# 累加每个簇内的像素 ab（注意：这里要用“缩放后的 tar_ab_scaled”）
for j in range(K):
    idx = (labels_tar == j)
    # 找到j-cluster对应的像素点
    if idx.any():
        Avg_tar[j] = tar_ab_scaled[idx].mean(axis=0)  # 在缩放空间里平均
        # 得到第j个cluster center的ab颜色的平均值
    else:
        Avg_tar[j] = centers[j]   # 若空簇，退化用中心值（或邻居均值）

# 2) 用 Γ 混出源端每个中心的“新中心颜色” Avg_src ∈ R^{K×2}
#    对第 i 行，若 Γ_i· 的和就是 a[i]，则作加权平均：
row_sum = a.copy()
row_sum[row_sum < 1e-12] = 1e-12
Avg_src = (X @ Avg_tar) / row_sum[:, None]   # K×2，仍在缩放空间

# 3) 把 Avg_src 回填给每个源像素（硬分配）
new_ab_scaled = Avg_src[labels_src]            # (N,2)，缩放空间

# 4) 反缩放回 Lab 的 ab 量纲
new_ab = scaler.inverse_transform(new_ab_scaled) # (N,2)

# 5) 合并 L 通道、重排回图像
new_img_arr = np.hstack([src_lumi.reshape(-1,1, order='F'), new_ab])
new_img_arr = new_img_arr.reshape(m, n, 3, order='F')
synthesis = color.lab2rgb(new_img_arr)

from PIL import Image
syn = (synthesis * 255).astype(np.uint8)
img = Image.fromarray(syn, mode='RGB')
img.save(r"C:\Users\Andy Li\OneDrive - CUHK-Shenzhen\桌面\LGU\Research\Optimal Transport\New_Sparse\OT\ot_9.12\syn1.jpg")