import numpy as np
import torch
import torch.sparse as sparse

def torch_shrinkage(x:torch.tensor,kappa:float) -> torch.Tensor:
    return torch.sign(x) * torch.clamp(torch.abs(x) - kappa,min=0.0)

def sparse_ADMM_LASSO(
        A,
        b,
        alpha: float,
        rho_init: float,
        max_iter: int = 100,
        tol_abs: float = 1e-4,
        tol_rel: float = 1e-2,
        verbose: bool = False,
        over_relaxation_param: float = 1.5,
        use_adaptive_rho: bool = False,
        rho_mu: float = 10.0,
        rho_tau_incr: float = 2.0,
        rho_tau_decr: float = 2.0,
        
):  
    # 检测 gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
            print(f"Using device: {device}")

    
    # 将矩阵 A 类型转为稀疏张量
    A = A.tocoo()
    indices = np.vstack((A.row, A.col))
    indices = torch.tensor(indices, dtype=torch.long, device=device)
    values = torch.tensor(A.data, dtype=torch.float32, device=device)
    shape = A.shape
    A = torch.sparse_coo_tensor(indices, values, shape, device=device, dtype=torch.float32)

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
    ATA = sparse.mm(A.t(),A)
    ATb = sparse.mm(A.t(),b)

    # 预计算常量
    eye_n = torch.eye(n, device=device).to_sparse()
    sqrt_n = torch.sqrt(torch.tensor(float(n), device=device))

    ATA_plus_rhoI = ATA + eye_n

  

    # def matvec(x):
    #     Ax = sparse.mm(A,x.unsqueeze(1))
    #     ATAx = sparse.mm(A.t(),Ax)
    #     rhox = rho * x.unsqueeze(1)
    #     Px = ATAx + rhox
    #     return Px.squeeze(1)
    
    # def conjugate_gradient(matvec, b, x=None, tol=1e-6, max_iter=1000):
    #     if x is None:
    #         x = torch.zeros(n,device=device)
    #     else:
    #         x = x.clone().to(device)
        
    #     b = b.squeeze(1)
    #     r = b - matvec(x)
    #     p = r.clone()
    #     rs_old = torch.dot(r, r)

    #     for i in range(max_iter):
    #         Ap = matvec(p)
    #         # 计算步长
    #         alpha = rs_old / torch.dot(p, Ap)
    #         # 更新解 x
    #         x += alpha * p
    #         # 更新残差 r
    #         r -= alpha * Ap
    #         # 计算新残差的范数平方
    #         rs_new = torch.dot(r, r)

    #         if torch.sqrt(rs_new) < tol:
    #             if verbose:
    #                 print(f"CG converged at iteration {i}")
    #             break

    #         p = r + (rs_new / rs_old) * p
    #         rs_old = rs_new

    #     return x

    if verbose:
        print(f"{'Iter':>4} | {'Objective':>12} | {'pri_norm':>10} | {'eps_pri':>10} | {'dual_norm':>10} | {'eps_dual':>10}")
        print('-' * 70)

    for k in range(max_iter):
        # 更新 x
        RHS = ATb + rho * (z - y)
        x = sparse.spsolve(ATA_plus_rhoI,RHS)
        # 过松弛
        x_hat = over_relaxation_param * x + (1 - over_relaxation_param) * z

        # z 更新
        z_old = z.clone()
        z = torch_shrinkage(x_hat + y, alpha / rho)

        # y 更新
        y = y + (x_hat - z)


        obj_val = 0.5 * torch.norm(sparse.mv(A,x) - b)**2 + alpha * torch.norm(x, p=1)
        

        # 计算残差及容忍度
        pri_norm = torch.norm(x - z)
        dual_norm = torch.norm(rho * (z - z_old))
        eps_pri = sqrt_n * tol_abs + tol_rel * max(torch.norm(x), torch.norm(-z))
        eps_dual = sqrt_n * tol_abs + tol_rel * torch.norm(rho * y)

        history['objective_value'].append(obj_val.item())
        history['rho_values'].append(rho)
        history['pri_norm'].append(pri_norm.item())
        history['dual_norm'].append(dual_norm.item())
        history['eps_pri'].append(eps_pri.item())
        history['eps_dual'].append(eps_dual.item())

        if verbose and (k % 10 == 0 or k == max_iter - 1):
            print(f"{k:4d} | {obj_val:12.4e} | {pri_norm:10.4e} | {eps_pri:10.4e} | {dual_norm:10.4e} | {eps_dual:10.4e}")

        # 自适应更新 rho
        if use_adaptive_rho:
            if pri_norm > rho_mu * dual_norm and dual_norm < eps_dual:
                rho *= rho_tau_incr
                y /= rho_tau_incr
            elif dual_norm > rho_mu * pri_norm and pri_norm < eps_pri:
                rho /= rho_tau_decr
                y *= rho_tau_decr
        # 收敛判断
        if pri_norm < eps_pri and dual_norm < eps_dual:
            if verbose:
                print(f"目标函数在{k+1}次迭代后收敛.")
            break

    return x, history
