import numpy as np
from numpy.linalg import inv, norm, cholesky

# 软阈值算子
def shrinkage(x, kappa):
    x_arr = np.asarray(x)
    return np.sign(x_arr) * np.maximum(np.abs(x_arr) - kappa, 0)




"""
参数：
alpha: Lasso 正则化参数
rho: 增广拉格朗日参数
verbose: 是否打印信息
over_relaxation_param: 过松弛因子
use_adaptive_rho: 是否自适应更新rho
rho_mu: 自适应rho平衡参数
rho_tau_incr: 自适应rho增大参数
rho_tau_decr: 自适应rho减小参数
"""
def ADMM_LASSO(A,b,alpha,rho_init,max_iter=100, tol_abs=1e-4, tol_rel=1e-2, verbose=False,over_relaxation_param=1.5,
               use_adaptive_rho=False,rho_mu=10.0,rho_tau_incr=2.0,rho_tau_decr=2.0):
    
    m, n = A.shape
 
    x = np.zeros((n,1))
    z = np.zeros((n,1))
    y = np.zeros((n,1))

    rho = rho_init
    rho_updated = False  

    history = {
        'objective_value': [], # 目标函数值
        'pri_norm': [],          # 原始残差范数
        'dual_norm': [],       # 对偶残差范数
        'eps_pri': [],         # 原始残差容忍度
        'eps_dual': [],        # 对偶残差容忍度
        'rho_values': []       # rho值变化
    }

    # 预计算常用矩阵
    ATA = A.T @ A
    ATb = A.T @ b.reshape(-1,1)  
    
    # Cholesky分解
    def compute_P(current_ATA, current_rho):
        try:
            L_val = cholesky(current_ATA + current_rho * np.eye(n))
            L_inv_val = inv(L_val)
            P_val = L_inv_val.T @ L_inv_val
            return P_val, True
        except np.linalg.LinAlgError:
            if verbose:
                print(f"Cholesky decomposition failed.")
            return None, False 
        
    P, precomputed_P = compute_P(ATA, rho)
    
    # 打印表头
    if verbose:
        print(f"{'Iter':>4} | {'Objective':>12} | {'pri_norm':>10} | {'eps_pri':>10} | {'dual_norm':>10} | {'eps_dual':>10}")
        print("-" * 70)
    
    for i in range(max_iter):
         
        # 更新 x
        q = ATb + rho * (z - y)
        if precomputed_P:
            x = P @ q
        else:
            x = inv(ATA + rho * np.eye(n)) @ q

        # Over relaxation
        x_hat = over_relaxation_param * x + (1 - over_relaxation_param) * z
        
        # 更新 z
        z_old = z.copy()
        z = shrinkage(x_hat + y, alpha / rho)

        # 更新 y
        y = y + (x_hat - z)

        # 计算目标函数值 
        value = 0.5 * norm(A @ x - b.reshape(-1,1))**2 + alpha * norm(x, 1)
        history['objective_value'].append(value)
        history['rho_values'].append(rho)
    
        # 计算残差和容忍度
        pri_norm = norm(x - z)
        dual_norm = norm(rho * (z - z_old))
        eps_pri = np.sqrt(n) * tol_abs + tol_rel * max(norm(x), norm(-z))
        eps_dual = np.sqrt(n) * tol_abs + tol_rel * norm(rho * y)

        history['pri_norm'].append(pri_norm)
        history['dual_norm'].append(dual_norm)
        history['eps_pri'].append(eps_pri)
        history['eps_dual'].append(eps_dual)

        if verbose and (i % 10 == 0 or i == max_iter - 1):
            print(f"{i:4d} | {value:12.4e} | {pri_norm:10.4e} | {eps_pri:10.4e} | {dual_norm:10.4e} | {eps_dual:10.4e}")

        # 自适应rho更新
        if use_adaptive_rho:
            rho_updated = False
            if pri_norm > rho_mu * dual_norm and dual_norm < eps_dual:
                rho_old = rho
                rho = rho * rho_tau_incr
                y = y / rho_tau_incr 
                rho_updated = True
                if verbose: print(f"Iter {i}: rho increased to {rho:.2e}")
            elif dual_norm > rho_mu * pri_norm and pri_norm < eps_pri:
                rho_old = rho
                rho = rho / rho_tau_decr
                y = y * rho_tau_decr
                rho_updated = True            
                if verbose: print(f"Iter {i}: rho decreased to {rho:.2e}")

        if rho_updated:
            P, precomputed_P = compute_P(ATA, rho)
            if not precomputed_P and P is None: 
                print("Error: 在更新 rho 后 Cholesky 分解失败")
                return x, history 

        # 收敛判断
        if pri_norm < eps_pri and dual_norm < eps_dual:
            if verbose:
                print(f"\n目标函数在{i+1}次迭代后收敛")
            break
    
    else: 
        if verbose:
            print("\n目标函数未正常收敛")

    return x, history


    


