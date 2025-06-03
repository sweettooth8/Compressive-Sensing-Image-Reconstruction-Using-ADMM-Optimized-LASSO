import numpy as np
import scipy
import torch

def torch_shrinkage(x:torch.tensor,kappa:float) -> torch.Tensor:
    return torch.sign(x) * torch.clamp(torch.abs(x) - kappa,min=0.0)

def ADMM_LASSO_torch(
        A,
        b,
        alpha: float,
        rho_init: float,
        max_iter: int = 1000,
        tol_abs: float = 1e-4,
        tol_rel: float = 1e-2,
        verbose: bool = False,
        over_relaxation_param: float = 1.5,
        use_adaptive_rho: bool = True,
        rho_mu: float = 10.0,
        rho_tau_incr: float = 2.0,
        rho_tau_decr: float = 2.0,
        
):  
    # 检测 gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
            print(f"Using device: {device}")


    A = torch.tensor(A, dtype=torch.float32, device=device)
    b = torch.tensor(b, dtype=torch.float32, device=device).view(-1,1)

    m, n = A.shape

    x = torch.zeros((n,1), device=device)
    z = torch.zeros_like(x)
    y = torch.zeros_like(x)

    rho = rho_init

    history = {
        'objective_value': [], # 目标函数值
        'pri_norm': [],          # 原始残差范数
        'dual_norm': [],       # 对偶残差范数
        'eps_pri': [],         # 原始残差容忍度
        'eps_dual': [],        # 对偶残差容忍度
        'rho_values': []       # rho值变化
    }

    # 预计算常用矩阵

    ATA = A.t().mm(A)
    ATb = A.t().mm(b)

    def compute_P(ATA: torch.Tensor, rho: float) -> torch.Tensor:
        # Cholesky 分解并构造 P = (ATA + rho I)^{-1}
        L = torch.linalg.cholesky(ATA + rho * torch.eye(n, device=device))
        L_inv = torch.inverse(L)
        return L_inv.t().mm(L_inv)

    # 初次计算 P
    P = compute_P(ATA, rho)


    if verbose:
        print(f"{'Iter':>4} | {'Objective':>12} | {'pri_norm':>10} | {'eps_pri':>10} | {'dual_norm':>10} | {'eps_dual':>10}")
        print('-' * 70)

    for k in range(max_iter):
        # x 更新
        q = ATb + rho * (z - y)
        x = P.mm(q)

        # 过松弛
        x_hat = over_relaxation_param * x + (1 - over_relaxation_param) * z

        # z 更新
        z_old = z.clone()
        z = torch_shrinkage(x_hat + y, alpha / rho)

        # y 更新 ？？？
        y = y + (x_hat - z)


        obj_val = 0.5 * torch.norm(A.mm(x) - b)**2 + alpha * torch.norm(x, p=1)
        history['objective_value'].append(obj_val.item())
        history['rho_values'].append(rho)

        # 计算残差及容忍度
        pri_norm = torch.norm(x - z)
        dual_norm = torch.norm(rho * (z - z_old))
        eps_pri = torch.sqrt(torch.tensor(float(n), device=device)) * tol_abs + tol_rel * torch.max(torch.norm(x), torch.norm(-z))
        eps_dual = torch.sqrt(torch.tensor(float(n), device=device)) * tol_abs + tol_rel * torch.norm(rho * y)

        history['pri_norm'].append(pri_norm.item())
        history['dual_norm'].append(dual_norm.item())
        history['eps_pri'].append(eps_pri.item())
        history['eps_dual'].append(eps_dual.item())

        if verbose and (k % 10 == 0 or k == max_iter - 1):
            print(f"{k:4d} | {obj_val:12.4e} | {pri_norm:10.4e} | {eps_pri:10.4e} | {dual_norm:10.4e} | {eps_dual:10.4e}")

        # 自适应 rho
        if use_adaptive_rho:
            if pri_norm > rho_mu * dual_norm and dual_norm < eps_dual:
                rho *= rho_tau_incr
                y /= rho_tau_incr
                P = compute_P(ATA, rho)
            elif dual_norm > rho_mu * pri_norm and pri_norm < eps_pri:
                rho /= rho_tau_decr
                y *= rho_tau_decr
                P = compute_P(ATA, rho)

        # 收敛判断
        if pri_norm < eps_pri and dual_norm < eps_dual:
            if verbose:
                print(f"目标函数在{k+1}次迭代后收敛.")
            break

    return x
