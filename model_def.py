import numpy as np
import torch
import torch.nn as nn

# ---------- interpolation ----------
def torch_linear_interp_1d(x_grid: torch.Tensor, y_grid: torch.Tensor, x_query: torch.Tensor) -> torch.Tensor:
    idx = torch.bucketize(x_query, x_grid)
    idx0 = torch.clamp(idx - 1, 0, x_grid.numel() - 1)
    idx1 = torch.clamp(idx,     0, x_grid.numel() - 1)

    x0 = x_grid[idx0]
    x1 = x_grid[idx1]
    y0 = torch.gather(y_grid, 1, idx0)
    y1 = torch.gather(y_grid, 1, idx1)

    w = (x_query - x0) / (x1 - x0 + 1e-12)
    return y0 + w * (y1 - y0)

# ---------- physics layer ----------
class RockPhysicsLayer(nn.Module):
    def __init__(self, r_grid_m: np.ndarray, T0=273.15, rho_l=1000.0, Lm=3.34e5):
        super().__init__()
        rg = torch.tensor(r_grid_m, dtype=torch.float32)
        self.register_buffer("r_grid", rg)
        self.T0 = float(T0)
        self.rho_l = float(rho_l)
        self.Lm = float(Lm)

        dr = torch.zeros_like(rg)
        dr[:-1] = rg[1:] - rg[:-1]
        dr[-1] = dr[-2]
        self.register_buffer("dr", dr)

    def forward(self, T, f_pdf, sigma_iw, A_H, clamp_phi=True):
        dT = torch.clamp(self.T0 - T, min=1e-3)

        rc = 2.0 * sigma_iw * self.T0 / (self.rho_l * self.Lm * dT)
        denom = 6.0 * torch.pi * self.rho_l * self.Lm * dT + 1e-30
        h = torch.pow((A_H * self.T0) / denom, 1.0/3.0)

        r = self.r_grid[None, :]
        dr = self.dr[None, :]

        lam = r / (h + 1e-12)
        phi = (2.0 * lam - 1.0) / torch.pow(lam + 1.0, 2.0)
        if clamp_phi:
            phi = torch.clamp(phi, 0.0, 1.0)

        mask1 = (r <= rc)
        W1 = torch.sum(f_pdf * dr * mask1, dim=1, keepdim=True)

        r_plus_h = r + h
        f_shift = torch_linear_interp_1d(self.r_grid, f_pdf, r_plus_h)
        mask2 = (r > rc)
        W2 = torch.sum(f_shift * phi * dr * mask2, dim=1, keepdim=True)

        Wf = W1 + W2
        Wf = torch.clamp(Wf, 0.0, 1.0)
        Wf = torch.where(T >= self.T0, torch.ones_like(Wf), Wf)
        return Wf

# ---------- backbone ----------
class TabularBackbone(nn.Module):
    def __init__(self, in_dim, hidden=256, n_blocks=3, scale=0.5):
        super().__init__()
        self.input_layer = nn.Linear(in_dim, hidden)
        self.act = nn.GELU()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU()
            ) for _ in range(n_blocks)
        ])
        self.scale = scale

    def forward(self, x):
        h = self.act(self.input_layer(x))
        for blk in self.blocks:
            h = h + self.scale * blk(h)
        return h

# ---------- main model ----------
class RockUWCNet(nn.Module):
    def __init__(self, tab_in_dim, r_grid_m, hidden=256, n_blocks=3, nmr_dim=128, residual_scale=0.7):
        super().__init__()
        self.residual_scale = residual_scale

        self.backbone = TabularBackbone(tab_in_dim, hidden=hidden, n_blocks=n_blocks, scale=0.5)

        M = len(r_grid_m)
        self.nmr_enc = nn.Sequential(
            nn.Linear(M, nmr_dim), nn.GELU(),
            nn.Linear(nmr_dim, nmr_dim), nn.GELU()
        )

        fusion_dim = hidden + nmr_dim + 1

        self.head_params = nn.Sequential(
            nn.Linear(fusion_dim, 128), nn.GELU(),
            nn.Linear(128, 2)
        )
        self.head_res = nn.Sequential(
            nn.Linear(fusion_dim, 128), nn.GELU(),
            nn.Linear(128, 1)
        )

        self.phys = RockPhysicsLayer(r_grid_m)

        self.sigma_min, self.sigma_max = 0.01, 0.06
        self.ah_min, self.ah_max = 1e-21, 1e-19

        nn.init.zeros_(self.head_res[-1].weight)
        nn.init.zeros_(self.head_res[-1].bias)
        nn.init.zeros_(self.head_params[-1].weight)
        nn.init.zeros_(self.head_params[-1].bias)

    def forward(self, Xn, X_raw, proxy, f_enc, f_phys_pdf):
        h_tab = self.backbone(Xn)
        z_nmr = self.nmr_enc(f_enc)
        hp = torch.cat([h_tab, z_nmr, proxy], dim=1)

        p = self.head_params(hp)
        sigma_raw = p[:, 0:1]
        ah_raw    = p[:, 1:2]

        sigma_iw = self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(sigma_raw)
        A_H      = self.ah_min   + (self.ah_max   - self.ah_min)   * torch.sigmoid(ah_raw)

        theta_init = X_raw[:, 0:1]
        theta_r    = X_raw[:, 1:2]
        T          = X_raw[:, -1:]
        delta = torch.clamp(theta_init - theta_r, min=1e-6)

        frac_phys = self.phys(T, f_phys_pdf, sigma_iw, A_H)
        frac_residual = self.residual_scale * torch.tanh(self.head_res(hp))
        frac_pred_raw = frac_phys + frac_residual

        theta_pred_raw = theta_r + frac_pred_raw * delta
        theta_phys     = theta_r + frac_phys     * delta

        return (theta_pred_raw, theta_phys, frac_residual, sigma_iw, A_H,
                theta_init, theta_r, T, frac_pred_raw, frac_phys)


# ---------- build & load ----------
def build_model(tab_in_dim, r_grid_path, ckpt_path, device="cpu"):
    """
    Build RockUWCNet and load trained weights.
    """
    r_grid_m = np.load(r_grid_path)

    model = RockUWCNet(
        tab_in_dim=tab_in_dim,
        r_grid_m=r_grid_m,
        hidden=256,
        n_blocks=3,
        residual_scale=0.7
    ).to(device)

    sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd, strict=False)

    model.eval()
    return model
