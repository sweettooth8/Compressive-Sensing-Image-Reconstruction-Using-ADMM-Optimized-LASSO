import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, sparse
import torch
from sklearn.linear_model import Lasso, Ridge
import time 
from admm_lasso_module import ADMM_LASSO

def _weights(x, dx=1, orig=0):
    x = np.ravel(x)
    floor_x = np.floor((x - orig) / dx).astype(np.int64)
    alpha = (x - orig - floor_x * dx) / dx
    return np.hstack((floor_x, floor_x + 1)), np.hstack((1 - alpha, alpha))


def _generate_center_coordinates(l_x):
    X, Y = np.mgrid[:l_x, :l_x].astype(np.float64)
    center = l_x / 2.0
    X += 0.5 - center
    Y += 0.5 - center
    return X, Y


def build_projection_operator(l_x, n_dir):
  
    X, Y = _generate_center_coordinates(l_x)
    angles = np.linspace(0, np.pi, n_dir, endpoint=False)
    data_inds, weights, camera_inds = [], [], []
    data_unravel_indices = np.arange(l_x**2)
    data_unravel_indices = np.hstack((data_unravel_indices, data_unravel_indices))
    for i, angle in enumerate(angles):
        Xrot = np.cos(angle) * X - np.sin(angle) * Y
        inds, w = _weights(Xrot, dx=1, orig=X.min())
        mask = np.logical_and(inds >= 0, inds < l_x)
        weights += list(w[mask])
        camera_inds += list(inds[mask] + i * l_x)
        data_inds += list(data_unravel_indices[mask])
    proj_operator = sparse.coo_matrix((weights, (camera_inds, data_inds)))
    return proj_operator


def generate_synthetic_data():
    rs = np.random.RandomState(0)
    n_pts = 36
    x, y = np.ogrid[0:l, 0:l]
    mask_outer = (x - l / 2.0) ** 2 + (y - l / 2.0) ** 2 < (l / 2.0) ** 2
    mask = np.zeros((l, l))
    points = l * rs.rand(2, n_pts)
    mask[(points[0]).astype(int), (points[1]).astype(int)] = 1
    mask = ndimage.gaussian_filter(mask, sigma=l / n_pts)
    res = np.logical_and(mask > mask.mean(), mask_outer)
    return np.logical_xor(res, ndimage.binary_erosion(res))


l = 128
proj_operator = build_projection_operator(l, l // 7)
data = generate_synthetic_data()
proj = proj_operator @ data.ravel()[:, np.newaxis]
proj += 0.15 * np.random.randn(*proj.shape)

rgr_ridge = Ridge(alpha=0.2)
rgr_ridge.fit(proj_operator, proj.ravel())
rec_l2 = rgr_ridge.coef_.reshape(l, l)


rgr_lasso = Lasso(alpha=0.001)
lasso_start_time = time.time()
rgr_lasso.fit(proj_operator, proj.ravel())
lasso_end_time = time.time()
rec_l1 = rgr_lasso.coef_.reshape(l, l)

proj_operator = proj_operator.toarray()
solver = ADMM_LASSO(rho_init=1.0, max_iter=1000,tol_abs=1e-6,tol_rel=1e-4, over_relaxation_param=1.5,use_adaptive_rho=True)
admm_lasso_start_time = time.time()
x_lasso = solver.forward(torch.from_numpy(proj_operator), torch.from_numpy(proj.ravel()), alpha=0.01)
admm_lasso_end_time = time.time()
rec_admm_lasso = x_lasso.reshape(l,l)

print(f"SKlearn's LASSO 重建图片运行时间：{lasso_end_time-lasso_start_time:4f}s")
print(f"ADMM LASSO 重建图片运行时间：{admm_lasso_end_time-admm_lasso_start_time:4f}s")


plt.figure(figsize=(8, 3.3))
plt.subplot(131)
plt.imshow(data, cmap=plt.cm.gray, interpolation="nearest")
plt.axis("off")
plt.title("original image")
plt.subplot(132)
plt.imshow(rec_l1, cmap=plt.cm.gray, interpolation="nearest")
plt.title("LASSO Recontruction")
plt.axis("off")
plt.subplot(133)
plt.imshow(rec_admm_lasso.cpu().numpy(), cmap=plt.cm.gray, interpolation="nearest")
plt.title("ADMM LASSO Recontruction")
plt.axis("off")

plt.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)

plt.savefig('example.png')