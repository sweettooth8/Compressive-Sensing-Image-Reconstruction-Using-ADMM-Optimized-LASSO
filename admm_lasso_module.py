import torch
import torch.nn as nn
import torch.nn.functional as F


class ADMM_LASSO(nn.Module):
    def __init__(
            self,
            rho_init: float = 1.0,
            max_iter: int = 500,
            tol_abs: float = 1e-4,
            tol_rel: float = 1e-3,
            verbose: bool = False,
            over_relaxation_param: float = 1.0,
            use_adaptive_rho: bool = False,
            rho_mu: float = 10.0,
            rho_tau_incr: float = 2.0,
            rho_tau_decr: float = 2.0,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super(ADMM_LASSO, self).__init__()
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
        
    def forward(self, A: torch.Tensor, b: torch.Tensor, alpha, x0=None):
        # 打印 device 信息
        if self.verbose:
            print(f"Using device: {self.device}")

        

        A = A.to(self.device)
        b = b.to(self.device).view(-1,1)

        m, n = A.shape
        rho = self.rho_init

        if x0 is not None:
            x = x0.view(-1,1).to(self.device, dtype=A.dtype).clone()
        else:
            x = torch.zeros((n,1), device=self.device, dtype=A.dtype)
        z = F.softshrink(x, alpha / rho)    # warm-start z
        y = torch.zeros_like(x)

        # 预计算常用矩阵
        ATA = A.t().mm(A)
        ATb = A.t().mm(b)

        # 预计算常量
        eye_n = torch.eye(n, device=self.device)
        sqrt_n = torch.sqrt(torch.tensor(float(n), device=self.device))

        def compute_P(ATA: torch.Tensor, rho: float, eye_n) -> torch.Tensor:
            try:
                L = torch.linalg.cholesky(ATA + rho * eye_n)
                return torch.cholesky_inverse(L)
            except torch.linalg.LinAlgError:
                if self.verbose: # 假设 self.verbose 在此作用域可见，或者通过参数传入
                    print(f"警告: Cholesky 分解在 rho={rho:.2e} 时失败，回退到 torch.inverse。")
                return torch.inverse(ATA + rho * eye_n)
        
        
        P = compute_P(ATA, rho, eye_n)

        with torch.no_grad():
            for i in range(self.max_iter):
                # 更新 x
                q = ATb + rho * (z - y)
                x = P.mm(q)

                # 过松弛
                x_hat = self.over_relaxation_param * x + (1 - self.over_relaxation_param) * z

                # 更新 z
                z_old = z.clone()
                z = F.softshrink(x_hat + y, alpha / rho)
                # 更新 y
                y = y + x_hat - z

                # 计算残差及容忍度
                pri_norm = torch.norm(x - z)
                dual_norm = torch.norm(rho * (z - z_old))
                eps_pri = sqrt_n * self.tol_abs + self.tol_rel * torch.max(torch.norm(x), torch.norm(-z))
                eps_dual = sqrt_n * self.tol_abs + self.tol_rel * torch.norm(rho * y)

                # 自适应更新 rho
                if self.use_adaptive_rho:
                    if pri_norm > self.rho_mu * dual_norm and dual_norm < eps_dual:
                        rho = rho * self.rho_tau_incr
                        y = y / self.rho_tau_incr
                        P = compute_P(ATA, rho, eye_n)
                    elif dual_norm > self.rho_mu * pri_norm and pri_norm < eps_pri:
                        rho = rho / self.rho_tau_decr
                        y = y * self.rho_tau_decr
                        P = compute_P(ATA, rho, eye_n)
                        

                # 收敛判断
                if pri_norm < eps_pri and dual_norm < eps_dual:
                    if self.verbose:
                        print(f"目标函数在{i+1}次迭代后收敛.")
                    break

        return x
        