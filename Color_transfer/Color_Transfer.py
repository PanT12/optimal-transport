from skimage import io, color
from skimage.transform import resize
import matplotlib.pyplot as plt
from Alg.Algorithm_Import import *
from load import *
import os


# load images and convert to Lab space
experiment_id = 3
folder_root = f"data{experiment_id}"
opt_cache_path = os.path.join(folder_root, f"opt_Cab_{experiment_id}.npz")
if os.path.exists(opt_cache_path):
    cache = np.load(opt_cache_path, allow_pickle=False)
    a = cache["a"]
    b = cache["b"]
    C = cache["C"]
    centers = cache["centers"]
    labels_src = cache["labels_src"]
    labels_tar = cache["labels_tar"]
    src = cache["src"]
    tar = cache["tar"]
    print(f"  [cache] loaded a,b,C from {opt_cache_path}")
else:
    src_path = f"data{experiment_id}/A.jpg"
    tar_path = f"data{experiment_id}/B.jpg"
    src, tar, src_lab, tar_lab, src_lumi, src_ab, tar_lumi, tar_ab = load_figures(
        src_path, tar_path, size_hw=(1024, 1024), show=False
    )
    # KMeans construction
    centers, a, b, labels_src, labels_tar, scaler, C = to_lab_kmeans(src_ab, tar_ab, K=512, scale_to_01=False)

    np.savez(opt_cache_path, a=a, b=b, C=C, centers=centers, labels_src=labels_src, labels_tar=labels_tar, src=src, tar=tar)
    print(f"  [cache] saved a,b,C to {opt_cache_path}")

from Sinkhorn_L0 import BISNsolver as BS
solver = BS(C, 1e-2, a, b)
X = solver.optimize(max_iter=300, tol=1e-9)


solver = BISNsolver(C, 1e-2, a, b)
X = solver.optimize(tol=1e-11, max_iter=200)

synthesis = remap_colors_from_transport(X, a, centers, labels_src, labels_tar, src, scaler=None)
from PIL import Image
syn = (synthesis * 255).astype(np.uint8)
img = Image.fromarray(syn, mode='RGB')
img.save(f"data{experiment_id}/{experiment_id}_synthesis.jpg")

img = io.imread(f"data{experiment_id}/{experiment_id}_synthesis.jpg")
fig, axs = plt.subplots(1, 3)
axs[0].imshow(src)
axs[1].imshow(tar)
axs[2].imshow(img)
axs[0].set_axis_off()
axs[1].set_axis_off()
axs[2].set_axis_off()
plt.tight_layout()
fig.savefig(f"{folder_root}/{experiment_id}_output.pdf", format="pdf", bbox_inches='tight')
# plt.show()
