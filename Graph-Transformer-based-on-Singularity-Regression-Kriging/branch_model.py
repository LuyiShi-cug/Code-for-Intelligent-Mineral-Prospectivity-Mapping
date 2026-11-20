from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import SpatialBranch, SpectrumBranch, dynamic_graph_construction
from training_utils import Predict, contrastive_loss, load_sample_indices

BRANCH = "spectrum"  # "spatial" 或 "spectrum"
DATA_ROOT = Path(".")
SAMPLE_INDEX_PATH = DATA_ROOT / "sample_indices_srk.npz"
REFERENCE_TIF = r"reference.tif"
OUTPUT_DIR = Path("outputs/single-branch")
seed = 4


torch.manual_seed(seed)
np.random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PointXY = np.load(DATA_ROOT / "input_data2/PointXY.npy").T
Label = np.load(DATA_ROOT / "input_data2/label1.npy").ravel().copy()
Label[Label == -1] = 0
S = load_sample_indices(SAMPLE_INDEX_PATH, to_tensor=True, device=device)

pos_indices, neg_indices = S["pos_indices"], S["neg_indices"]
cls_label = torch.full((len(Label),), -1, dtype=torch.long, device=device)
cls_label[neg_indices] = 0
cls_label[pos_indices] = 1
epochs, learning_rate, recon_weight = 70, 1e-3, 0.01
if BRANCH == "spatial":
    Residual = np.load(DATA_ROOT / "input_data2/SpatialInput.npy")
    GeologyData = np.load(DATA_ROOT / "input_data2/GeologyData.npy")
    x_spatial = np.hstack((Residual, GeologyData))
    initial_edge_index, initial_edge_weight = dynamic_graph_construction(x_spatial, PointXY.T, KNeighbor=30,
                                                                         device=str(device))
    x_spatial = torch.as_tensor(x_spatial, dtype=torch.float32, device=device)
    model = SpatialBranch(InputSize=x_spatial.shape[1], HiddenSize=32, locationData=PointXY.T, DropOut=0.2, K=30,
                          device=str(device)).to(device)

else:
    x_spectral = np.load(DATA_ROOT / "input_data2/TransformerInput.npy").reshape(len(PointXY), 39, 9)
    x_spectral = torch.as_tensor(x_spectral, dtype=torch.float32, device=device)
    model = SpectrumBranch(token_dim=9, d_model=64, nhead=4, num_layers=2, seq_len=39, dim_feedforward=64).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
criterion = nn.CrossEntropyLoss(ignore_index=-1)
loss_history = {k: [] for k in ("epoch", "total_loss", "L_contrast", "L_consist", "L_cls")}

for epoch in range(epochs):
    started = time.time()
    model.train()
    optimizer.zero_grad()
    if BRANCH == "spatial":
        HiddenX, recon, cls = model(x_spatial, initial_edge_index, initial_edge_weight)
        original = x_spatial
    else:
        recon, HiddenX, cls = model(x_spectral)
        original = x_spectral
    L_contrast = contrastive_loss(HiddenX, pos_indices, neg_indices, tau=0.1)
    L_consist = recon_weight * F.mse_loss(recon, original)
    L_cls = criterion(cls, cls_label)
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
    if BRANCH == "spatial":
        _, _, cls = model(x_spatial, initial_edge_index, initial_edge_weight)
    else:
        _, _, cls = model(x_spectral)
    probab = F.softmax(cls, dim=-1)[:, 1].cpu().numpy()
    Predict(probab, PointXY, OUTPUT_DIR / f"{BRANCH}_model.tif", REFERENCE_TIF)
