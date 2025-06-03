import os
import matplotlib.pyplot as plt
from scipy.signal import medfilt2d
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
import multiprocessing as mp
import time
# torch
import torch
from admm_lasso_module import ADMM_LASSO
from admm_lasso_batch_module import batch_ADMM_LASSO

os.environ["OMP_NUM_THREADS"] = "1"     
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)



RESULTS_DIR = 'Output'

os.makedirs(RESULTS_DIR, exist_ok=True)


def work_cv(args):
    block_idx, S_value, alpha_list_full, num_cv_folds = args

    block_cpu = g_blocks[block_idx]
    T_cpu     = g_T                      
    device    = 'cpu'
    solver    = ADMM_LASSO(device=device)

    # 已知像素采样
    _, sp, sidx, _, _ = sample_pixels_from_block(block_cpu.to(device), S_value, copy=True)
    sp = sp.to(torch.float64) 
    # 生成一次 KFold 划分
    splits = [(torch.from_numpy(tr).to(device),
               torch.from_numpy(va).to(device))
              for tr, va in KFold(n_splits=num_cv_folds,
                                  shuffle=True, random_state=42).split(sp)]

    alpha_seq = list(alpha_list_full)
    np.random.shuffle(alpha_seq)
    alpha_seq = sorted(set(alpha_seq), reverse=True)

    prev_x_fold = [None] * num_cv_folds

    alpha_errs = []

    for a in alpha_seq:
        prev_x_fold = [None] * num_cv_folds
        fold_err = []
        for fid, (tr_idx, va_idx) in enumerate(splits):

            x0 = prev_x_fold[fid]        # warm-start（

            tr_T = T_cpu[sidx[tr_idx], :]
            tr_y = sp[tr_idx]
            dct  = solver(tr_T, tr_y, a, x0=x0).detach()

            prev_x_fold[fid] = dct       

            va_est = (T_cpu[sidx[va_idx], :] @ dct.view(-1, 1)).squeeze()
            fold_err.append(mse(va_est, sp[va_idx]))

        alpha_errs.append((a, np.mean(fold_err)))

    best_alpha = min(alpha_errs, key=lambda t: t[1])[0]

    # ---------- 返回 ----------
    return block_idx, float(best_alpha), sp, sidx

def init_worker(blocks_shared, T_shared):
    global g_blocks, g_T
    g_blocks, g_T = blocks_shared, T_shared


def img_read(filename, device):
  
    imgIn = plt.imread(filename)
    if imgIn.dtype == np.uint8:
        imgIn = imgIn / 255.0
    img_tensor = torch.from_numpy(imgIn.copy()).float()

    return img_tensor.to(device)

def img_show(img_out, title):
 
    img_out_np = img_out.cpu().numpy()

    if img_out_np.max() <= 1.0 and img_out_np.min() >=0.0: 
         img_out_np = (img_out_np * 255).astype(np.uint8)
    else: 
        img_out_np = np.clip(img_out_np, 0, 255).astype(np.uint8)

    plt.imshow(img_out_np, cmap='gray')
    if title:
        plt.title(title)
    plt.show()

def img_save(img_out_tensor, filename):
    img_out_np = img_out_tensor.cpu().numpy()
    if img_out_np.max() <= 1.0 and img_out_np.min() >=0.0:
         img_out_np = img_out_np * 255
    else:
        img_out_np = np.clip(img_out_np, 0, 255)

    plt.figure() 
    plt.imshow(img_out_np, cmap='gray', vmin=0, vmax=255)
    if filename:
        plt.title(filename)
    plt.savefig(os.path.join(RESULTS_DIR, f"{filename}.png"), dpi=300, bbox_inches='tight')
    plt.close()

def split_img_into_blocks(img, k):
  
    H, W = img.shape
    assert H % k == 0, ""
    assert W % k == 0, ""
    num_blocks_h = H // k
    num_blocks_w = W // k
    img_blocks = img.reshape(num_blocks_h, k, num_blocks_w, k).permute(0, 2, 1, 3).reshape(-1, k, k)    

    return img_blocks

def sample_pixels_from_block(block, S, copy=False):
  
    block_size = block.numel()
    assert S <= block_size, 'S should not be larger than the number of pixels in the block'
    num_pixels_to_remove = block_size - S

    block_flat = block.flatten()

    if copy:
        block_flat = block_flat.clone()

    indices = torch.arange(0, block_size, dtype=torch.long, device=block.device)

    shuffled_indices = torch.randperm(block_size, device=block.device)

    unknown_pixels_indices = shuffled_indices[:num_pixels_to_remove]

    unknown_pixels_indices = torch.sort(unknown_pixels_indices).values

    unknown_pixels = block_flat[unknown_pixels_indices].clone()

    sampled_pixel_indices = shuffled_indices[num_pixels_to_remove:]

    sampled_pixel_indices = torch.sort(sampled_pixel_indices).values

    sampled_pixels = block_flat[sampled_pixel_indices].clone() 

    min_val_of_block = torch.min(block_flat) 
    block_flat[unknown_pixels_indices] = min_val_of_block

    block_for_viewing = block_flat.reshape(block.shape)

    return block_for_viewing, sampled_pixels, sampled_pixel_indices, unknown_pixels, unknown_pixels_indices

def combine_block_to_get_image(img_blocks, original_shape):

    rows, cols = original_shape
    n_blocks, k_rows, k_cols = img_blocks.shape 

    k = k_rows 

    num_block_rows = rows // k
    num_block_cols = cols // k

    temp_img = img_blocks.reshape(num_block_rows, num_block_cols, k, k)

    temp_img = temp_img.permute(0, 2, 1, 3)

    img = temp_img.reshape(rows, cols)

    return img

def calc_alpha(u, P):

    val_1_P = (1.0 / P)
    val_2_P = (2.0 / P)

    alpha = torch.where(u == 1,
                    torch.sqrt(val_1_P.clone().detach() if isinstance(val_1_P, torch.Tensor) else torch.tensor(val_1_P, device=u.device, dtype=u.dtype)),
                    torch.sqrt(val_2_P.clone().detach() if isinstance(val_2_P, torch.Tensor) else torch.tensor(val_2_P, device=u.device, dtype=u.dtype)))

    return alpha

def calc_beta(v, Q):
    val_1_Q = (1.0 / Q)
    val_2_Q = (2.0 / Q)

    beta = torch.where(v == 1,
                   torch.sqrt(val_1_Q.clone().detach() if isinstance(val_1_Q, torch.Tensor) else torch.tensor(val_1_Q, device=v.device, dtype=v.dtype)),
                   torch.sqrt(val_2_Q.clone().detach() if isinstance(val_2_Q, torch.Tensor) else torch.tensor(val_2_Q, device=v.device, dtype=v.dtype)))
    
    return beta

def get_val_T(x, y, u, v, P, Q):

    P_tensor = torch.tensor(P, dtype=x.dtype, device=x.device)
    Q_tensor = torch.tensor(Q, dtype=x.dtype, device=x.device)

    alpha = calc_alpha(u, P_tensor)
    beta = calc_beta(v, Q_tensor)

    cos_x = torch.cos((torch.pi * (2 * x - 1) * (u - 1)) / (2 * P_tensor))
    cos_y = torch.cos((torch.pi * (2 * y - 1) * (v - 1)) / (2 * Q_tensor))

    return alpha * beta * cos_x * cos_y

def make_T(device, n=8, dtype=torch.float64):
  
    P, Q = n, n
    
    x_coords = torch.arange(1, n + 1, device=device, dtype=dtype).view(1, n, 1, 1) 
    y_coords = torch.arange(1, n + 1, device=device, dtype=dtype).view(n, 1, 1, 1)
    u_coords = torch.arange(1, n + 1, device=device, dtype=dtype).view(1, 1, n, 1) 
    v_coords = torch.arange(1, n + 1, device=device, dtype=dtype).view(1, 1, 1, n) 


    T_values = get_val_T(x_coords, y_coords, u_coords, v_coords, P, Q)

    T_matrix = T_values.view(n * n, n * n)
    T_matrix[:, 0] = 1.0

    return T_matrix

class ImageRecover():
    def __init__(
        self,
        img_path='data/nature.bmp',
        block_size=8, 
        S_values=np.array([50]),
        alpha_val_list=np.logspace(-7, 7, num=50),
        num_cv_folds=5,
        verbose=False
    ):
       
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.T = make_T(self.device)
        self.block_size = block_size
        self.num_cv_folds = num_cv_folds
        self.S_values = S_values
        self.img = img_read(img_path, self.device)
        self.alpha_val_list = alpha_val_list
        self.admm_lasso_solver = batch_ADMM_LASSO(rho_init=1.0, verbose=False, device=self.device)
        self.verbose = verbose


    def cross_validation_for_blocks(self, img_blocks):
        
        T_cpu = self.T.cpu() 
        img_blocks = img_blocks.cpu()
        T_cpu.share_memory_()
        img_blocks.share_memory_()
       
        tasks = [
            (idx, self.S, self.alpha_val_list, self.num_cv_folds) 
            for idx in range(img_blocks.shape[0])
        ]

        all_results = [None] * img_blocks.shape[0]

        best_alpha_list = []
        A_list = []
        b_list = []

        
        num_workers = 24
        chunksize   = 128
        print(f"使用 {num_workers} 个工作进程进行并行计算...")

        with mp.Pool(processes=num_workers, initializer=init_worker, initargs=(img_blocks, T_cpu)) as pool:

            results_iterator = pool.imap_unordered(work_cv, tasks, chunksize=chunksize)
            
            for result in tqdm(results_iterator, total=len(tasks), desc="交叉验证"):
                if result: 
                    block_idx, best_alpha, sampled_pixels, sampled_indices = result
                    all_results[block_idx] = (best_alpha, sampled_pixels, sampled_indices)
            

        print(f"交叉验证完成。")
        
        
        best_alpha_list = [r[0] for r in all_results]
        b_list = [r[1].to(self.device) for r in all_results]
        A_list = [self.T[r[2].to(self.device), :] for r in all_results]

        alpha_batch = torch.tensor(best_alpha_list, dtype=torch.float64, device=self.device)
        b_batch = torch.stack(b_list) 
        A_batch = torch.stack(A_list) 

        print(f"S={self.S} 时选定的 Alpha 均值: {alpha_batch.mean().item():.3e}, 标准差: {alpha_batch.std().item():.3e}")
        print(f"Alpha 范围: [{alpha_batch.min().item():.3e}, {alpha_batch.max().item():.3e}]")
        plt.figure(figsize=(8,6))
        plt.hist(alpha_batch.cpu().numpy(), bins=min(20, len(self.alpha_val_list)), edgecolor='black')
        plt.xscale('log')
        plt.title(f'S={self.S} Best Alpha Distribution (CV Folds={self.num_cv_folds})')
        plt.xlabel('Selected Alpha (log scale)')
        plt.ylabel('Frequency of Blocks')
        plot_filename = f"S_{self.S}_alpha_distribution_log.png"
        plt.savefig(os.path.join(RESULTS_DIR, plot_filename))
        plt.close() 
        
        return alpha_batch, A_batch, b_batch
    
    def recover_blocks(self, A_batch, b_batch, alpha_batch):
        
        x_batch = self.admm_lasso_solver(A_batch, b_batch, alpha_batch)

        reconstructed_flat = (self.T @ x_batch.T).T

        k = self.block_size

        reconstructed_blocks = reconstructed_flat.view(-1, k, k)
        reconstructed_blocks = torch.clamp(reconstructed_blocks, 0, 1)

        return reconstructed_blocks
    
    def recover_image(self):
  
        for S_value in self.S_values:

            self.S = S_value
            
            img_blocks = split_img_into_blocks(self.img, self.block_size)

            alpha_batch, A_batch, b_batch = self.cross_validation_for_blocks(img_blocks)

            reconstructed_blocks = self.recover_blocks(A_batch, b_batch, alpha_batch)

            reconstructed_img = combine_block_to_get_image(reconstructed_blocks, self.img.shape)
            
            img_filtered = median_filter(reconstructed_img * 255.0) / 255.0

            img_save(img_filtered, f"S={self.S}")

            




def median_filter(image, kernel_size=3): 

    image_np = image.cpu().numpy()
    filtered = medfilt2d(image_np, kernel_size=kernel_size)

    return torch.from_numpy(filtered).to(image.device)


def mse(y_true, y_pred):
    y_true = y_true.to(y_pred.device, dtype=y_pred.dtype)
    return torch.mean(torch.square(y_true - y_pred)).item()

if __name__ == '__main__':
    mp.set_start_method('fork', force=True) 
    ir = ImageRecover()
    start_time = time.time()
    ir.recover_image()
    end_time = time.time()
    print(f"运行时间:{end_time-start_time}s")