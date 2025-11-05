from skimage import io, color
from skimage.transform import resize

# 图像信息的展示-读取-预处理
# Load image and resize
m = n = 256
size = m * n

src = io.imread(r"/Users/ting/PycharmProjects/OT/pexelA-0.png")
tar = io.imread(r"/Users/ting/PycharmProjects/OT/pexelB-0.png")

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
a = np.bincount(labels_src, minlength=K).astype(float) + 1e-12
a /= a.sum()
b = np.bincount(labels_tar, minlength=K).astype(float) + 1e-12
b /= b.sum()

# a = a_counts / (a_counts.sum() + 1e-12)     # (K,)
# b = b_counts / (b_counts.sum() + 1e-12)     # (K,)

# --- 成本矩阵 C：ab 空间的 L2 距离平方 ---
# C[i,j] = ||centers[i] - centers[j]||^2
diff = centers[:, None, :] - centers[None, :, :]   # (K, K, 2)
# diff[i, j] = [aᵢ - aⱼ, bᵢ - bⱼ]  # 从中心j到中心i的颜色向量
C = np.sum(diff * diff, axis=2)                    # (K, K)

from Alg.Algorithm_Import import *
import ot
import sys
from pathlib import Path


# ===== 固定超参 =====
eta = 1e-4
stepsize = 1e-2
tol = 1e-7
n_trials = 2
rng = np.random.default_rng(41)

X_opt = ot.emd(a, b, C)
opt = float(np.sum(C * X_opt))
print(opt)
print("gurobi finished")
print("Start SinkhornL0")

solver = BISNsolver(C, 1e-4, a, b, obj_truth=opt)
X = solver.optimize(tol=1e-9, max_iter=10)
result = np.sum(C * X)
print(X.shape)
# K*K, 128*128 cluster*cluster
print(result, opt)
print("different between opt is ", np.sum(C * X) - opt)

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
img.save(r"C:\Users\Andy Li\OneDrive - CUHK-Shenzhen\桌面\LGU\Research\Optimal Transport\New_Sparse\OT\ot_9.12\syn2.jpg")

