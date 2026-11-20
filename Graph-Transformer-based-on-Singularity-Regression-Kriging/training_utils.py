"""训练脚本共用的数据处理与输出函数。"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree


def zscore_normalize(data, eps=1e-8):
    data = np.asarray(data, dtype=float)
    mean = np.nanmean(data, axis=0, keepdims=True)
    std = np.nanstd(data, axis=0, keepdims=True)
    return (data - mean) / np.where(std < eps, 1.0, std)


def normalization(data):
    data = np.asarray(data)
    span = data.max() - data.min()
    return (data - data.min()) / span if span else np.zeros_like(data)


def sigmoid_normalize(series, k=1.0, center=None):
    if center is None:
        center = series.median()
    scale = series.std()
    if scale == 0 or np.isnan(scale):
        scale = 1.0
    return 1.0 / (1.0 + np.exp(-k * (series - center) / scale))


def reconstruction_loss(recon_x, x_spatial):
    """用输入方差归一化重构误差。"""
    return F.mse_loss(recon_x, x_spatial) / x_spatial.var().clamp(min=1e-8)


def consistency_loss(y_hat, a_tilde):
    return F.mse_loss(y_hat, a_tilde)


def clr_transform(data, eps=1e-9):
    """对正值组成数据执行中心化对数比变换。"""
    data = np.asarray(data, dtype=float)
    if np.any(data < 0):
        raise ValueError("CLR 输入不能包含负值")
    logged = np.log(np.maximum(data, eps))
    return logged - logged.mean(axis=1, keepdims=True)


def contrastive_loss(z, pos_indices, neg_indices, tau=0.1):
    if len(pos_indices) < 2 or len(neg_indices) == 0:
        raise ValueError("对比损失至少需要两个正样本和一个负样本")
    positive_features = F.normalize(z[pos_indices], dim=1)
    negative_features = F.normalize(z[neg_indices], dim=1)
    positive_sim = positive_features @ positive_features.T / tau
    positive_sim = positive_sim.masked_fill(
        torch.eye(len(pos_indices), dtype=torch.bool, device=z.device), -float("inf"))
    negative_sim = positive_features @ negative_features.T / tau
    numerator = torch.logsumexp(positive_sim, dim=1)
    denominator = torch.logsumexp(torch.cat((positive_sim, negative_sim), dim=1), dim=1)
    return (denominator - numerator).mean()


def load_sample_indices(load_path, to_tensor=False, device="cpu"):
    """读取保存的正负样本索引及附加信息。"""
    with np.load(load_path, allow_pickle=False) as data:
        result = {
            "pos_indices": data["pos_indices"].copy(),
            "neg_indices": data["neg_indices"].copy(),
            "point_xy": data["point_xy"].copy() if "point_xy" in data else None,
            "label": data["label"].copy() if "label" in data else None,
            "soft_label": data["soft_label"].copy() if "soft_label" in data else None,
            "meta": json.loads(str(data["meta_json"].item())) if "meta_json" in data else {},
        }
    if to_tensor:
        result["pos_indices"] = torch.as_tensor(result["pos_indices"], dtype=torch.long, device=device)
        result["neg_indices"] = torch.as_tensor(result["neg_indices"], dtype=torch.long, device=device)
        if result["soft_label"] is not None:
            result["soft_label"] = torch.as_tensor(result["soft_label"], dtype=torch.float32, device=device)
    return result


def save_sample_indices(save_path, pos_indices, neg_indices, point_xy=None,
                        label=None, soft_label=None, meta=None):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    def as_array(value):
        if value is None:
            return np.empty(0)
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value)
    np.savez_compressed(
        save_path, pos_indices=as_array(pos_indices).astype(np.int64),
        neg_indices=as_array(neg_indices).astype(np.int64),
        point_xy=as_array(point_xy), label=as_array(label),
        soft_label=as_array(soft_label),
        meta_json=np.array(json.dumps(meta or {}, ensure_ascii=False)),
    )


def LabelSelection_SRK(Label, srk_residual, srk_trend, point_xy, trend_q=70,
                       n_neg_per_pos=3, d_min=15000, seed=4):
    """从低残差、高趋势且远离已知矿床的点中抽取负样本。"""
    positive = np.flatnonzero(Label == 1)
    if len(positive) < 2:
        raise ValueError("至少需要两个正样本")
    mask = ((srk_residual > np.percentile(srk_residual, 10)) &
            (srk_residual < np.percentile(srk_residual, 30)) &
            (srk_trend > np.percentile(srk_trend, trend_q)) & (Label == 0))
    candidates = np.flatnonzero(mask)
    distance, _ = cKDTree(point_xy[positive]).query(point_xy[candidates])
    candidates = candidates[distance > d_min]
    count = min(len(candidates), len(positive) * n_neg_per_pos)
    if count == 0:
        raise ValueError("没有符合条件的负样本，请调整筛选参数")
    negative = np.random.default_rng(seed).choice(candidates, count, replace=False)
    return positive, negative


def Predict(Hidden, XY, TifName, reference_path):
    """按规则网格坐标写入 GeoTIFF，缺失像元记为 -99。"""
    from osgeo import gdal

    point_xy = np.asarray(XY)
    values = np.asarray(Hidden)
    values = normalization(values)
    output_path = TifName
    reference = gdal.Open(str(reference_path))
    if reference is None:
        raise FileNotFoundError(reference_path)
    x_values = np.unique(point_xy[:, 0])
    y_values = np.unique(point_xy[:, 1])[::-1]
    if len(x_values) < 2 or len(y_values) < 2:
        raise ValueError("栅格至少需要两行两列")
    dx = np.diff(x_values)
    dy = -np.diff(y_values)
    if not (np.allclose(dx, dx[0]) and np.allclose(dy, dy[0])):
        raise ValueError("坐标不是规则网格")
    raster = np.full((len(y_values), len(x_values)), -99, dtype=np.float32)
    columns = np.searchsorted(x_values, point_xy[:, 0])
    rows = len(y_values) - 1 - np.searchsorted(y_values[::-1], point_xy[:, 1])
    raster[rows, columns] = np.asarray(values, dtype=np.float32)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = gdal.GetDriverByName("GTiff").Create(
        str(output_path), len(x_values), len(y_values), 1, gdal.GDT_Float32)
    if output is None:
        raise OSError(f"无法创建 {output_path}")
    output.SetGeoTransform((x_values[0] - dx[0] / 2, dx[0], 0,
                            y_values[0] + dy[0] / 2, 0, -dy[0]))
    output.SetProjection(reference.GetProjection())
    band = output.GetRasterBand(1)
    band.SetNoDataValue(-99)
    band.WriteArray(raster)
    output.FlushCache()
    output = None
