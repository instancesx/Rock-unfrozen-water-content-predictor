import os
import numpy as np
import pandas as pd
import torch
import streamlit as st
import matplotlib.pyplot as plt

from model_def import build_model

# =============================
# Config / Paths
# =============================
DEVICE = "cpu"

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
R_GRID_PATH = os.path.join(BASE_DIR, "r_grid_m.npy")
CKPT_PATH   = os.path.join(BASE_DIR, "best_model.pth")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.npz")

# Must match training feature order
TAB_FEATURES = [
    "Moisture_Content",
    "Theta_residual",
    "Dry_Density",
    "Specific_surface_area",
    "Porosity",
    "Temperature"   # Kelvin
]

NMR_TAIL_Q = 0.99  # consistent with training


# =============================
# Load scaler / model (cached)
# =============================
@st.cache_resource
def load_scaler(path: str):
    data = np.load(path)
    mean = data["mean"].astype(np.float32)
    std  = data["std"].astype(np.float32)
    return mean, std

MEAN, STD = load_scaler(SCALER_PATH)

def standardize(x_tab: np.ndarray) -> np.ndarray:
    return (x_tab.astype(np.float32) - MEAN) / STD

@st.cache_resource
def load_model():
    model = build_model(
        tab_in_dim=len(TAB_FEATURES),
        r_grid_path=R_GRID_PATH,
        ckpt_path=CKPT_PATH,
        device=DEVICE
    )
    model.eval()
    return model

MODEL = load_model()

R_GRID_M  = np.load(R_GRID_PATH).astype(np.float64)
R_GRID_UM = (R_GRID_M * 1e6).astype(np.float64)  # pore radius in um grid


# =============================
# NMR encoding (match training)
# =============================
def nmr_to_enc_and_fphys(r_um: np.ndarray, content: np.ndarray):
    """
    Match training pipeline:
    - tail clip by pore quantile q=0.99
    - interpolate g(log10 r) onto fixed grid in log space
    - compute f_phys PDF and normalize ∫ f dr = 1
    - enc = log1p(g_grid * 10)
    """
    r_um = np.asarray(r_um, dtype=np.float64)
    c    = np.asarray(content, dtype=np.float64)

    m = np.isfinite(r_um) & np.isfinite(c) & (r_um > 0)
    r_um = r_um[m]
    c    = c[m]

    idx = np.argsort(r_um)
    r_um = r_um[idx]
    c    = c[idx]

    # Tail clip (per-sample)
    hi = np.quantile(r_um, NMR_TAIL_Q)
    m2 = r_um <= hi
    r_um = r_um[m2]
    c    = c[m2]

    # IMPORTANT: training uses raw Content (NO /100)
    c = np.clip(c, 0.0, None)

    # g(log10 r) grid in log-space
    log_r    = np.log10(r_um)
    log_grid = np.log10(R_GRID_UM)
    g_grid   = np.interp(log_grid, log_r, c, left=0.0, right=0.0).astype(np.float64)

    # PDF conversion and normalization over dr (in meters)
    dr_m = np.zeros_like(R_GRID_M, dtype=np.float64)
    dr_m[:-1] = R_GRID_M[1:] - R_GRID_M[:-1]
    dr_m[-1]  = dr_m[-2]

    f_um = g_grid / (R_GRID_UM * np.log(10.0) + 1e-30)  # per um
    f_m  = f_um * 1e6                                   # per m
    area = float(np.sum(f_m * dr_m))

    if (not np.isfinite(area)) or area <= 1e-20:
        f_phys = np.zeros_like(f_m, dtype=np.float32)
    else:
        f_phys = (f_m / area).astype(np.float32)

    enc = np.log1p(g_grid * 10.0).astype(np.float32)
    return enc, f_phys, g_grid


# =============================
# Temperature grid helper (FIXED)
# =============================
def make_temperature_grid(t_min: float, t_max: float, n_pts: int) -> np.ndarray:
    """
    目标：当范围跨过 0℃ 且 t_max>0 时：
    - 保证 0℃以下的温度点与 [-20, 0]（同 n_pts）时完全一致 -> 曲线形状不变
    - 仅在右侧额外追加 0℃以上的温度点 -> 平台延伸
    """
    t_min = float(t_min)
    t_max = float(t_max)
    n_pts = int(n_pts)

    # 纯负温（或纯正温）就按用户设置的 n_pts 均匀采样
    if (t_max <= 0.0) or (t_min >= 0.0):
        return np.linspace(t_min, t_max, n_pts, dtype=np.float32)

    # 跨 0℃：负温段固定使用 [t_min, 0] 的 linspace（点数就是 n_pts）
    Ts_neg = np.linspace(t_min, 0.0, n_pts, dtype=np.float32)

    # 用负温段的步长在正温段延伸，保证“只是右侧追加”
    if n_pts >= 2:
        step = float(Ts_neg[1] - Ts_neg[0])
        step = abs(step) if step != 0 else 1.0
    else:
        step = max(abs(t_min) / 10.0, 0.5)

    # 正温段：从 0 往右走到 t_max，步长与负温段一致
    Ts_pos = np.arange(0.0, t_max + step * 0.5, step, dtype=np.float32)
    if Ts_pos.size == 0:
        Ts_pos = np.array([0.0, t_max], dtype=np.float32)
    else:
        if not np.isclose(float(Ts_pos[-1]), t_max, atol=1e-6):
            Ts_pos = np.append(Ts_pos, np.float32(t_max))

    # 合并（去掉重复的 0）
    Ts = np.concatenate([Ts_neg, Ts_pos[1:]]).astype(np.float32)

    # 保证单调递增
    Ts = np.unique(Ts)  # unique 会排序
    return Ts.astype(np.float32)


# =============================
# Prediction (match training)
# =============================
def predict_curve(
    theta_init_pct: float,
    theta_r_pct: float,
    rho_d: float,
    ssa: float,
    porosity_pct: float,
    Ts_C: np.ndarray,
    enc: np.ndarray,
    f_phys: np.ndarray
):
    """
    Align with training evaluation:
    - tab features in the same units as training (percent numbers, NOT /100)
    - Temperature in Kelvin
    - proxy = max(273.15 - T, 0)
    - postprocess: clip to [theta_r, theta_init], and if T >= 0C set theta=theta_init
    """
    Ts_C = np.asarray(Ts_C, dtype=np.float32)
    Ts_K = Ts_C + 273.15

    preds = []
    with torch.no_grad():
        for T_K in Ts_K:
            x_tab = np.array([
                theta_init_pct,
                theta_r_pct,
                rho_d,
                ssa,
                porosity_pct,
                float(T_K)
            ], dtype=np.float32)

            Xn = standardize(x_tab)[None, :]  # (1, 6)

            X_raw = np.array([[theta_init_pct, theta_r_pct, float(T_K)]], dtype=np.float32)  # (1, 3)
            proxy = np.array([[max(273.15 - float(T_K), 0.0)]], dtype=np.float32)            # (1, 1)

            theta_pred, *_ = MODEL(
                torch.tensor(Xn, device=DEVICE),
                torch.tensor(X_raw, device=DEVICE),
                torch.tensor(proxy, device=DEVICE),
                torch.tensor(enc[None, :], device=DEVICE),
                torch.tensor(f_phys[None, :], device=DEVICE),
            )

            pred = float(theta_pred.item())

            # ---- training-aligned postprocess ----
            pred = min(pred, float(theta_init_pct))
            pred = max(pred, float(theta_r_pct))
            if float(T_K) >= 273.15:
                pred = float(theta_init_pct)

            preds.append(pred)

    return np.array(preds, dtype=np.float32)


# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="Rock UWC Predictor", layout="centered")
st.title("🧊 Rock Unfrozen Water Content Predictor")

st.sidebar.header("Input rock physical parameters")

theta_init_pct = st.sidebar.number_input("Initial moisture content θ_init (%)", 0.0, 100.0, 25.0)
theta_r_pct    = st.sidebar.number_input("Residual water content θ_r (%)", 0.0, 50.0, 5.0)
porosity_pct   = st.sidebar.number_input("Porosity n (%)", 1.0, 80.0, 25.0)

rho_d = st.sidebar.number_input("Dry density ρ_d (g/cm³)", 1.0, 3.0, 2.2)
ssa   = st.sidebar.number_input("Specific surface area SSA (m²/g)", 0.1, 200.0, 10.0)

st.sidebar.header("Temperature settings")
t_min = st.sidebar.number_input("T min (°C)", -60.0, 20.0, -20.0)
t_max = st.sidebar.number_input("T max (°C)", -60.0, 20.0, 0.0)
n_pts = st.sidebar.slider("Number of points", 5, 200, 41)

#if (t_min < 0.0) and (t_max > 0.0):
    #st.sidebar.caption("✅ 跨 0℃ 时：将保持 0℃以下温度点不变，仅在右侧额外追加正温点（平台延伸）。")

run_btn = st.sidebar.button("🚀 Predict")

st.subheader("📂 Upload NMR inversion CSV")
st.markdown("CSV format: **pore (μm)** , **content (raw amplitude / intensity)** ")

csv_file = st.file_uploader("Upload NMR CSV", type=["csv", "txt"])

enc = None
f_phys = None
g_grid = None

if csv_file is not None:
    df = pd.read_csv(csv_file, sep=None, engine="python")
    r_um = df.iloc[:, 0].values
    content = df.iloc[:, 1].values

    enc, f_phys, g_grid = nmr_to_enc_and_fphys(r_um, content)

    st.success(f"NMR loaded. Bins={len(enc)}; tail clipped at q={NMR_TAIL_Q}")

    with st.expander("📊 Preview pore-size distributions", expanded=True):
        # Slightly larger figsize + constrained_layout to avoid overlap
        fig, axes = plt.subplots(2, 1, figsize=(7.5, 8.0), constrained_layout=True)

        axes[0].plot(R_GRID_UM, g_grid)
        axes[0].set_xscale("log")
        axes[0].set_xlabel("r (μm, log)")
        axes[0].set_ylabel("g(log10 r)")
        axes[0].set_title("Encoder base g(log10 r)")
        axes[0].grid(True, which="both", ls="--", alpha=0.3)

        axes[1].plot(R_GRID_UM, f_phys * 1e-6)
        axes[1].set_xscale("log")
        axes[1].set_xlabel("r (μm, log)")
        axes[1].set_ylabel("f_pdf(r) (per μm),  ∫f dr = 1")
        axes[1].set_title("Physical PDF f(r) derived from g(log10 r)")
        axes[1].grid(True, which="both", ls="--", alpha=0.3)

        st.pyplot(fig, use_container_width=True)

    with st.expander("🧪 Debug (should match training scale)", expanded=False):
        st.write("enc stats:", {
            "enc_min": float(np.min(enc)),
            "enc_mean": float(np.mean(enc)),
            "enc_max": float(np.max(enc))
        })
        dr_m = np.zeros_like(R_GRID_M)
        dr_m[:-1] = R_GRID_M[1:] - R_GRID_M[:-1]
        dr_m[-1]  = dr_m[-2]
        st.write("∫ f(r) dr (should be ~1):", float(np.sum(f_phys.astype(np.float64) * dr_m)))
else:
    st.info("Please upload an NMR CSV first.")

if run_btn:
    if enc is None or f_phys is None:
        st.error("Upload NMR CSV before prediction.")
        st.stop()

    if t_min > t_max:
        st.error("T min must be <= T max.")
        st.stop()

    Ts_C = make_temperature_grid(float(t_min), float(t_max), int(n_pts))
    preds_pct = predict_curve(
        theta_init_pct=float(theta_init_pct),
        theta_r_pct=float(theta_r_pct),
        rho_d=float(rho_d),
        ssa=float(ssa),
        porosity_pct=float(porosity_pct),
        Ts_C=Ts_C,
        enc=enc,
        f_phys=f_phys
    )

    st.subheader("📈 Predicted Unfrozen Water Content Curve (θ vs T)")
    fig2, ax2 = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    ax2.plot(Ts_C, preds_pct, marker="o", label="Predicted θ (%)")
    ax2.axhline(float(theta_init_pct), color="r", ls="--", label="θ_init (%)")
    ax2.set_xlabel("Temperature (°C)")
    ax2.set_ylabel("Unfrozen water content θ (%)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    st.pyplot(fig2, use_container_width=True)

    st.subheader("📊 Prediction Table (%)")
    df_out = pd.DataFrame({
        "Temperature (°C)": Ts_C,
        "θ_pred (%)": preds_pct
    })
    st.dataframe(df_out, use_container_width=True)
