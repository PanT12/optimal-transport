import numpy as np
from mnist.loader import MNIST
from scipy.spatial.distance import cdist
import os, glob
from pathlib import Path


def get_ot_parameters(dataset, source_indices, target_indices, norm_type='euclidean', eps=0.01):
    """
    Parameters:
    - dataset: MNIST / MNIST Fashion / DOTmark
    - source_indices: 源图像索引列表
    - target_indices: 目标图像索引列表  
    - norm_type: 计算成本的范数类型 ('euclidean', 'manhattan', 'cosine'等)
    - eps: 正则化参数，用于平滑分布
    
    Returns:
    - C: 成本矩阵 (m x n)
    - a: 源分布向量 (m,)
    - b: 目标分布向量 (n,)
    """
    # 获取源图像和目标图像
    source_images = []
    target_images = []
    
    for idx in source_indices:
        if hasattr(dataset[idx], '__getitem__'):
            # 如果是元组格式 (image, label)
            img = dataset[idx][0] if isinstance(dataset[idx], tuple) else dataset[idx]
        else:
            img = dataset[idx]
        
        # 转换为numpy数组并展平
        if hasattr(img, 'numpy'):
            img_array = img.numpy().flatten()
        elif hasattr(img, 'cpu'):
            img_array = img.cpu().numpy().flatten()
        else:
            img_array = np.array(img).flatten()
        
        source_images.append(img_array)
    
    for idx in target_indices:
        if hasattr(dataset[idx], '__getitem__'):
            # 如果是元组格式 (image, label)
            img = dataset[idx][0] if isinstance(dataset[idx], tuple) else dataset[idx]
        else:
            img = dataset[idx]
        
        # 转换为numpy数组并展平
        if hasattr(img, 'numpy'):
            img_array = img.numpy().flatten()
        elif hasattr(img, 'cpu'):
            img_array = img.cpu().numpy().flatten()
        else:
            img_array = np.array(img).flatten()
        
        target_images.append(img_array)
    
    # 转换为numpy数组
    source_images = np.array(source_images)  # shape: (m, d)
    target_images = np.array(target_images)  # shape: (n, d)
    
    # 计算源分布和目标分布（添加正则化）
    a = get_marginal_with_regularization(source_images, eps)
    b = get_marginal_with_regularization(target_images, eps)
    
    # 计算成本矩阵（像素级别）
    C = get_pixel_cost_matrix(source_images, target_images, norm_type)
    
    return C, a, b


def get_marginal_with_regularization(images, eps=0.01):
    """
    获取图像的边缘分布（添加正则化，参考LoadMNIST.py）
    
    Parameters:
    - images: 图像数组，shape: (N, d)
    - eps: 正则化参数
    
    Returns:
    - marginal: 归一化的分布向量
    """
    # 将多个图像合并为一个分布
    mu = images.flatten()
    mu = mu / mu.sum()
    
    # 添加正则化（参考LoadMNIST.py的mnist函数）
    n = len(mu)
    mu = (1 - eps / 8) * mu + eps / (8 * n)
    
    return mu.astype(np.float64)


def get_pixel_cost_matrix(source_images, target_images, norm_type='euclidean'):
    """
    计算像素级别的成本矩阵（参考LoadMNIST.py的实现）
    
    Parameters:
    - source_images: 源图像，shape: (m, d)
    - target_images: 目标图像，shape: (n, d)
    - norm_type: 范数类型
    
    Returns:
    - C: 成本矩阵，shape: (m, n)
    """
    # 获取图像尺寸
    d = source_images.shape[1]  # 像素数量
    m_pixels = int(np.sqrt(d))  # 图像边长
    
    # 生成像素坐标网格
    coords = cartesian_product(np.arange(m_pixels), np.arange(m_pixels))
    
    # 计算像素间的距离矩阵
    if norm_type == 'euclidean':
        C = cdist(coords, coords, metric='euclidean')
        C = C * C  # 平方欧几里得距离
    elif norm_type == 'manhattan':
        C = cdist(coords, coords, metric='cityblock')
    elif norm_type == 'cosine':
        C = cdist(coords, coords, metric='cosine')
    else:
        C = cdist(coords, coords, metric='euclidean')
        C = C * C
    
    # 归一化成本矩阵到[0,1]范围
    C /= np.max(C)
    
    return C.astype(np.float64)


def cartesian_product(*arrays):
    """
    计算笛卡尔积（参考LoadMNIST.py）
    """
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def load_mnist(mnist_dir, p_id=0, q_id=1, norm_type='euclidean', eps=0.01):

    if not os.path.isdir(mnist_dir):
        raise FileNotFoundError(f"MNIST 目录不存在: {mnist_dir}")

    mndata = MNIST(mnist_dir)
    images, _ = mndata.load_training()

    p = np.array(images[p_id], dtype=np.float64).reshape(1, -1)
    q = np.array(images[q_id], dtype=np.float64).reshape(1, -1)

    a = get_marginal_with_regularization(p, eps)
    b = get_marginal_with_regularization(q, eps)
    C = get_pixel_cost_matrix(p, q, norm_type)
    return C, a, b


def load_dotmark(dotmark_dir, category, size, p_id, q_id, normalize=True, norm_type='euclidean', eps=0.01):
    """按类别批量读取 DOTmark 的 csv 文件并堆叠为 (N, size, size) 的矩阵。

    参数:
    - base_data_dir: 指向 data 目录，例如 .../OT/OT/data
    - category: 类别名称，如 "ClassicImages", "CauchyDensity" 等
    - size: 侧长，如 32, 64, 128, 256, 512
    - normalize: 是否将每个矩阵归一化为和为 1（OT 常用）。默认 False
    - return_names: 是否同时返回文件名列表
    - dtype: numpy 数据类型，默认 np.float64

    返回:
    - stack: 形状为 (N, size, size) 的 numpy 数组
    - 如果 return_names=True，额外返回与之对应的文件名列表
    """
    dotmark_dir = Path(f"{dotmark_dir}/Data/{category}")
    pattern = f"data{size}_*.csv"
    files = sorted(dotmark_dir.glob(pattern))

    matrices = []

    for f in files:
        file_path = os.path.join(dotmark_dir, f)
        mat = np.loadtxt(file_path, delimiter=",")
        if normalize:
            s = mat.sum()
            if s != 0:
                mat = mat / s
        matrices.append(mat)

    stack = np.stack(matrices, axis=0)

    p = np.array(stack[p_id], dtype=np.float64).reshape(1, -1)
    q = np.array(stack[q_id], dtype=np.float64).reshape(1, -1)

    a = get_marginal_with_regularization(p, eps)
    b = get_marginal_with_regularization(q, eps)
    C = get_pixel_cost_matrix(p, q, norm_type)

    return C, a, b
