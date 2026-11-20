from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import SpatialBranch, SpatialSpectralNet, SpectrumBranch, dynamic_graph_construction
from training_utils import (LabelSelection_SRK, Predict, clr_transform,
                            contrastive_loss, load_sample_indices, save_sample_indices,
                            sigmoid_normalize, zscore_normalize)

# 运行前按实际目录修改这些参数。
METHOD = "srk_residual"  # "srk" 或 "idw" 或 “srk_residual”
DATA_ROOT = Path(".")
SAMPLE_INDEX_PATH = DATA_ROOT / "sample_indices_srk.npz"
REFERENCE_TIF = r"reference.tif"
OUTPUT_DIR = Path("outputs/dual-branch")
seed = 4

torch.manual_seed(seed)
np.random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PointXY = np.load(DATA_ROOT / "input_data/PointXY.npy").T
Label = np.load(DATA_ROOT / "input_data/label1.npy").ravel().copy()
Label[Label == -1] = 0
GeologyData = np.load(DATA_ROOT / "input_data/GeologyData.npy")
df_resid = pd.read_csv(DATA_ROOT / "data/srk_krige_resid.csv")
df_trend = pd.read_csv(DATA_ROOT / "data/srk_trend.csv")

if METHOD == "idw":
    # ---- IDW 分支 ----
    source = DATA_ROOT / "idw_data"
    Residual = np.load(source / "SpatialInput.npy")
    x_spectral = np.load(source / "TransformerInput.npy").reshape(len(PointXY), 39, 9)

    clr_Residual = clr_transform(Residual)
    x_spatial = clr_Residual

elif METHOD == "srk":
    # ---- SRK 分支 ----
    source = DATA_ROOT / "input_data"
    Residual = np.load(source / "SRK_pred_SpatialInput.npy")
    x_spectral = np.load(source / "SRK_pred_TransformerInput.npy").reshape(len(PointXY), 39, 9)
    clr_Residual = clr_transform(Residual)
    x_spatial = clr_Residual

elif METHOD == "srk_residual":
    source = DATA_ROOT / "input_data"
    Residual = np.load(source / "SpatialInput.npy")
    x_spectral = np.load(source / "TransformerInput.npy").reshape(len(PointXY), 39, 9)
    x_spatial = np.hstack((zscore_normalize(Residual), GeologyData))


    feature_cols = [c for c in df_resid if c not in ("x", "y")
                    and pd.api.types.is_numeric_dtype(df_resid[c])]
    a_tilde = df_resid[feature_cols].apply(sigmoid_normalize).to_numpy()
    a_tilde = torch.as_tensor(a_tilde, dtype=torch.float32, device=device)

# SRK 首次运行生成索引；IDW 复用相同索引。
if SAMPLE_INDEX_PATH.exists():
    S = load_sample_indices(SAMPLE_INDEX_PATH, to_tensor=True, device=device)
    if S["point_xy"] is None or not np.allclose(S["point_xy"], PointXY, atol=1e-6):
        raise ValueError("PointXY 与样本索引文件不一致")
    if S["label"] is None or not np.array_equal(S["label"].ravel(), Label):
        raise ValueError("Label 与样本索引文件不一致")
    pos_indices, neg_indices = S["pos_indices"], S["neg_indices"]
elif METHOD == "srk_residual":

    pos_indices, neg_indices = LabelSelection_SRK(Label, df_resid[feature_cols].mean(axis=1).to_numpy(),
                                                  df_trend[feature_cols].mean(axis=1).to_numpy(), PointXY,
                                                  trend_q=70, n_neg_per_pos=3, d_min=15000, seed=seed)
    save_sample_indices(SAMPLE_INDEX_PATH, pos_indices, neg_indices,
                        point_xy=PointXY, label=Label,
                        meta={"seed": seed, "trend_q": 70, "n_neg_per_pos": 3, "d_min": 15000})
    pos_indices = torch.as_tensor(pos_indices, device=device)
    neg_indices = torch.as_tensor(neg_indices, device=device)
else:
    raise FileNotFoundError(f"缺少样本索引文件：{SAMPLE_INDEX_PATH}")

initial_edge_index, initial_edge_weight = dynamic_graph_construction(x_spatial, PointXY.T, KNeighbor=30, device=str(device))
x_spatial = torch.as_tensor(x_spatial, dtype=torch.float32, device=device)
x_spectral = torch.as_tensor(x_spectral, dtype=torch.float32, device=device)
cls_label = torch.full((len(Label),), -1, dtype=torch.long, device=device)
cls_label[neg_indices] = 0
cls_label[pos_indices] = 1

spatial_branch = SpatialBranch(InputSize=x_spatial.shape[1], HiddenSize=32, locationData=PointXY.T, DropOut=0.2, K=30,
                               device=device)
spectral_branch = SpectrumBranch(token_dim=9, d_model=64, nhead=4, num_layers=2, seq_len=39, dim_feedforward=64)
model = SpatialSpectralNet(spatial_branch, spectral_branch, 32, 64, dropout=0.2, proj_dim=64).to(device)
epochs = 70
tau = 0.1
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
criterion = nn.CrossEntropyLoss(ignore_index=-1)
loss_history = {k: [] for k in ("epoch", "total_loss", "L_contrast", "L_consist", "L_cls")}

for epoch in range(epochs):
    started = time.time()
    model.train()
    optimizer.zero_grad()
    recon, z_proj, y_hat, _, _ = model(x_spatial, initial_edge_index, initial_edge_weight, x_spectral)
    L_contrast = contrastive_loss(z_proj, pos_indices, neg_indices, tau=tau)
    if METHOD == "srk_residual":
        L_consist = F.mse_loss(recon, a_tilde)
    else:
        L_consist = F.mse_loss(recon, x_spatial)
    L_cls = criterion(y_hat, cls_label)
    loss = L_contrast + 0.5 * L_consist + 0.5 * L_cls
    loss.backward()
    optimizer.step()
    scheduler.step()
    loss_history["epoch"].append(epoch)
    loss_history["total_loss"].append(loss.item())
    loss_history["L_contrast"].append(L_contrast.item())
    loss_history["L_consist"].append(L_consist.item())
    loss_history["L_cls"].append(L_cls.item())

    print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | "
          f"L_contrast: {L_contrast.item():.4f} | "
          f"L_consist: {L_consist.item():.4f} | "
          f"L_cls: {L_cls.item():.4f} | "
          f"Time: {time.time() - started:.2f}s")

loss_df = pd.DataFrame(loss_history)
loss_df.to_csv(OUTPUT_DIR / "loss_history.csv", index=False)
axes = loss_df.plot(x="epoch", y=["total_loss", "L_cls", "L_contrast", "L_consist"],
                    subplots=True, layout=(2, 2), figsize=(12, 8), legend=False)
fig = axes[0, 0].figure
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "loss_curves.png", dpi=300)

model.eval()
with torch.no_grad():
    _, _, y_hat, _, _ = model(x_spatial, initial_edge_index, initial_edge_weight, x_spectral)
    probab = F.softmax(y_hat, dim=-1)[:, 1].cpu().numpy()
    Predict(probab, PointXY, OUTPUT_DIR / f"{METHOD.upper()}_Dual_model.tif", REFERENCE_TIF)
