import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import torch.optim as optim
from torch_geometric.nn import GCNConv
from numpy import sqrt
from sklearn.preprocessing import MinMaxScaler


# 动态构图函数（基于当前节点特征和位置信息重新计算边）
def dynamic_graph_construction(features, locationData, KNeighbor=30, Distance=5000, device='cuda'):
    edge_index = []
    edge_weight = []
    if isinstance(features, np.ndarray):
        feature_np = features
    elif isinstance(features, torch.Tensor):
        feature_np = features.detach().cpu().numpy()
    else:
        feature_np = np.asarray(features)

    for i in range(len(locationData[0])):

        X_diff = locationData[0] - locationData[0][i]
        Y_diff = locationData[1] - locationData[1][i]
        spatial_dist = sqrt(X_diff ** 2 + Y_diff ** 2)

        neighborIndex = np.where(spatial_dist <= Distance)[0]
        neighbor_features = feature_np[neighborIndex]
        feature_dist = np.sqrt(np.sum((neighbor_features - feature_np[i]) ** 2, axis=1))


        if len(feature_dist) >= KNeighbor:
            KMinIndex = feature_dist.argsort()[:KNeighbor]
        else:
            KMinIndex = feature_dist.argsort()[:len(feature_dist)]

        KminNeighborIndex = neighborIndex[KMinIndex]
        selected_feature_dist = feature_dist[KMinIndex]
        selected_spatial_dist = spatial_dist[KminNeighborIndex]

        scaler_feature = MinMaxScaler()
        norm_feature = 1 - scaler_feature.fit_transform(selected_feature_dist.reshape(-1, 1)).flatten()

        scaler_spatial = MinMaxScaler()
        norm_spatial = 1 - scaler_spatial.fit_transform(selected_spatial_dist.reshape(-1, 1)).flatten()

        for idx in range(len(KminNeighborIndex)):
            edge_index.append([KminNeighborIndex[idx], i])
            edge_weight.append(norm_feature[idx] + norm_spatial[idx])

    edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float, device=device)
    return edge_index, edge_weight


# ------------------------------------------------------------
# 1. 位置编码（用于 Transformer 谱分支）
# ------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (N, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


# ------------------------------------------------------------
# 2. 谱分支：Transformer（ViT 思想）
# ------------------------------------------------------------
class SpectrumBranch(nn.Module):
    def __init__(self, token_dim=9, d_model=64, nhead=4, num_layers=2, seq_len=39, dim_feedforward=128):
        super().__init__()
        self.proj = nn.Linear(token_dim, d_model)  # 将 9 维邻域特征投影到 d_model
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.d_model = d_model
        self.ReconClassifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 9),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.Sigmoid()
        )

    def forward(self, x, return_attn=False):
        # x: (N, seq_len=39, token_dim=9)
        x = self.proj(x)  # (N, 39, d_model)
        x = self.pos_encoder(x)
        attn_weights = []
        if return_attn:
            # 逐层手动执行，捕获每层注意力
            for layer in self.transformer.layers:
                x2, attn = layer.self_attn(x, x, x, need_weights=True, average_attn_weights=False)
                attn_weights.append(attn.detach().cpu())
                x = layer.norm1(x + layer.dropout1(x2))
                x2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
                x = layer.norm2(x + layer.dropout2(x2))
            ReconX = self.ReconClassifier(x)
            HiddenX = x.mean(dim=1)
            x_cls = self.classifier(HiddenX)
            return ReconX, HiddenX, x_cls, attn_weights
        else:
            x = self.transformer(x)
            ReconX = self.ReconClassifier(x)
            HiddenX = x.mean(dim=1)
            x_cls = self.classifier(HiddenX)
            return ReconX, HiddenX, x_cls


# ------------------------------------------------------------
# 3. 空间分支：GCN
# ------------------------------------------------------------
class SpatialBranch(nn.Module):
    def __init__(self, InputSize, HiddenSize, locationData, DropOut=0.3, K=30, device='cuda'):
        super(SpatialBranch, self).__init__()
        self.Conv1 = GCNConv(InputSize, 64, bias=False)
        self.Conv2 = GCNConv(64, HiddenSize, bias=False)
        self.recon_head = nn.Sequential(
            nn.LayerNorm(HiddenSize),
            nn.Linear(HiddenSize, InputSize)
        )
        self.classifier = nn.Sequential(
            nn.Linear(HiddenSize, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

        self.locationData = locationData
        self.K = K
        self.device = device

        self.dropout = DropOut

    def encode(self, x, initial_edge_index, initial_edge_weight):
        self.layer1_graph = (initial_edge_index, initial_edge_weight)  # 记录第一层输入图
        x = self.Conv1(x, initial_edge_index, initial_edge_weight)
        x = F.relu(x)

        # 第一层后重新构图，作为第二层输入
        edge_index2, edge_weight2 = dynamic_graph_construction(x, self.locationData, self.K, device=self.device)
        self.layer2_graph = (edge_index2, edge_weight2)  # 记录第二层输入图

        # 第二层GCN
        x = self.Conv2(x, edge_index2, edge_weight2)
        x = F.relu(x)

        HiddenX = x
        return HiddenX

    def forward(self, x, initial_edge_index, initial_edge_weight):
        # 编码：生成隐层特征并保存每层的图
        HiddenX = self.encode(x, initial_edge_index, initial_edge_weight)
        cls = self.classifier(HiddenX)
        recon_x = self.recon_head(HiddenX)

        return HiddenX, recon_x, cls


# ============================================================
# SpatialSpectralNet
# ============================================================
class SpatialSpectralNet(nn.Module):
    def __init__(self, spatial_branch, spectrum_branch, spatial_out, spectrum_out, dropout=0.2, proj_dim=64):
        super(SpatialSpectralNet, self).__init__()

        self.spatial_branch = spatial_branch
        self.spectrum_branch = spectrum_branch
        self.dropout = nn.Dropout(dropout)

        # 融合 + 分类
        self.fc1 = nn.Linear(spatial_out + spectrum_out, 64)
        self.ln = nn.LayerNorm(64)
        self.recon_head = nn.Linear(64, 39)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

        # 对比学习投影头 → z_proj
        self.proj_head = nn.Sequential(
            nn.Linear(spatial_out + spectrum_out, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x_spatial, edge_index, edge_weight, x_spectrum):
        """
        x_spatial    : (N, InputSize)        → 空间分支（GCN）
        edge_index   : (2, E)                → 空间分支第 1 层初始图
        edge_weight  : (E,)                  → 空间分支第 1 层初始边权
        x_spectral   : (N, seq_len, token_dim) → 谱分支（Transformer）
        返回
        ----
        H_fused : (N, 64)    融合特征，用于一致性损失
        z_proj  : (N, proj_dim) 投影特征，用于对比损失
        y_hat   : (N,)       正类概率，用于分类损失
        """

        spatial_output, _, _ = self.spatial_branch(x_spatial, edge_index, edge_weight)
        _, spectrum_output, _ = self.spectrum_branch(x_spectrum)
        combined = torch.cat((spatial_output, spectrum_output), dim=-1)

        H_fused = self.ln(F.relu(self.fc1(combined)))
        H_fused = self.dropout(H_fused)
        recon_x = self.recon_head(H_fused)

        # 对比学习投影
        z_proj = F.normalize(self.proj_head(combined), dim=1)

        # 分类：取正类概率
        logits = self.classifier(H_fused)

        return recon_x, z_proj, logits, spatial_output, spectrum_output
