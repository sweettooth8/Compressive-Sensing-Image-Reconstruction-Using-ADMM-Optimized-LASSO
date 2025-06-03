import torch
import torch.nn as nn

def soft_thresholding(a: torch.Tensor, kappa_batch: torch.Tensor) -> torch.Tensor:
    kappa_batch = torch.maximum(kappa_batch, torch.tensor(0.0, device=a.device, dtype=a.dtype))
    return torch.sign(a) * torch.maximum(torch.abs(a) - kappa_batch, torch.tensor(0.0, device=a.device, dtype=a.dtype))


class batch_ADMM_LASSO(nn.Module):
    def __init__(
            self,
            rho_init: float = 0.5,
            max_iter: int = 1000,
            tol_abs: float = 1e-4,
            tol_rel: float = 1e-3,
            verbose: bool = False,
            over_relaxation_param: float = 1.5,
            use_adaptive_rho: bool = True,
            rho_mu: float = 10.0,
            rho_tau_incr: float = 2.0,
            rho_tau_decr: float = 2.0,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super(batch_ADMM_LASSO, self).__init__()
        self.rho_init = rho_init
        self.max_iter = max_iter
        self.tol_abs = tol_abs
        self.tol_rel = tol_rel
        self.verbose = verbose
        self.over_relaxation_param = over_relaxation_param
        self.use_adaptive_rho = use_adaptive_rho
        self.rho_mu = rho_mu
        self.rho_tau_incr = rho_tau_incr
        self.rho_tau_decr = rho_tau_decr
        self.device = device

    def forward(self, A_batch: torch.Tensor, b_batch: torch.Tensor, alpha_batch: torch.Tensor):
        # 打印 device 信息
        if self.verbose:
            print(f"Using device: {self.device}")
        

        A_batch = A_batch.to(self.device)
        b_batch = b_batch.to(self.device).unsqueeze(-1)
        alpha_batch = alpha_batch.to(self.device).view(-1, 1, 1) 


        B, m, n = A_batch.shape
        x = torch.zeros((B,n,1), device= self.device)
        z = torch.zeros_like(x)
        y = torch.zeros_like(x)
        rho = self.rho_init

        # 预计算常用矩阵
        ATA = torch.bmm(A_batch.transpose(1, 2), A_batch)  
        ATb = torch.bmm(A_batch.transpose(1, 2), b_batch)
        eye_n = torch.eye(n, device=self.device).expand(B, n, n)
        sqrt_n = torch.sqrt(torch.tensor(float(n), device=self.device))

        # Cholesky 分解并构造 P = (ATA + rho I)^{-1}
        def compute_P(ATA: torch.Tensor, rho: float, eye_n: torch.Tensor) -> torch.Tensor:
            try:
                L = torch.linalg.cholesky(ATA + rho * eye_n)
                # P = torch.cholesky_inverse(L) # 更高效
                L_inv = torch.inverse(L) # 或者使用 inverse
                P = torch.bmm(L_inv.transpose(1, 2), L_inv)
                return P
            except torch.linalg.LinAlgError:
                print("警告: Cholesky 分解失败。尝试使用 torch.inverse 直接计算。")
                # 如果 Cholesky 失败，尝试直接求逆（可能更慢且不稳定）
                return torch.inverse(ATA + rho * eye_n)

        P = compute_P(ATA, rho, eye_n)

        with torch.no_grad():
            for i in range(self.max_iter):
                # 更新 x
                q = ATb + rho * (z - y)
                x = torch.bmm(P,q)

                # 过松弛
                x_hat = self.over_relaxation_param * x + (1 - self.over_relaxation_param) * z

                # 更新 z
                z_old = z.clone()
                # 使用我们自定义的、支持批处理 kappa 的软阈值函数
                kappa_batch = alpha_batch / rho 
                z = soft_thresholding(x_hat + y, kappa_batch)
                # 更新 y
                y = y + x_hat - z

                # 计算残差及容忍度
                pri_norm = torch.norm(x - z, dim=(1, 2))
                dual_norm = torch.norm(rho * (z - z_old), dim=(1, 2))
                eps_pri = sqrt_n * self.tol_abs + self.tol_rel * torch.max(torch.norm(x, dim=(1, 2)), torch.norm(-z, dim=(1, 2)))
                eps_dual = sqrt_n * self.tol_abs + self.tol_rel * torch.norm(rho * y, dim=(1, 2))

                # 自适应更新 rho
                rho_changed = False
                if self.use_adaptive_rho:
                    increase_mask = (pri_norm > self.rho_mu * dual_norm) & (dual_norm < eps_dual)
                    decrease_mask = (dual_norm > self.rho_mu * pri_norm) & (pri_norm < eps_pri)

                    if increase_mask.any():
                        rho *= self.rho_tau_incr
                        y = y / self.rho_tau_incr
                        rho_changed = True
                    elif decrease_mask.any():
                        rho /= self.rho_tau_decr
                        y = y * self.rho_tau_decr
                        rho_changed = True
                    
                    if rho_changed:
                        P = compute_P(ATA, rho, eye_n)
                converged = (pri_norm < eps_pri) & (dual_norm < eps_dual)
                if converged.all():
                    if self.verbose:
                        print(f"批次在第{i+1}轮全部收敛")
                break

        return x.squeeze(-1)

