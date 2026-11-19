from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import GCNConv
from torch.utils.checkpoint import checkpoint


BUNDLE_ROOT = Path(__file__).parent.parent.parent
PROJECT_ROOT = BUNDLE_ROOT
FINAL_ROOT = BUNDLE_ROOT / "results"
MAIN_ROOT = BUNDLE_ROOT
GRAPH_DIR = BUNDLE_ROOT / "data" / "processed"
PRETRAIN_EPOCHS = 100
FINETUNE_EPOCHS = 80
SUMMARY = BUNDLE_ROOT / "results" / "training_reference"
CONSTRAINT_SOURCE = BUNDLE_ROOT / "data" / "inputs" / "constraints.csv"
CONFIG_SOURCE = BUNDLE_ROOT / "data" / "inputs" / "model_config.json"
GRAPH_SOURCE = GRAPH_DIR / "model_input_arrays.npz"
GRAPH_METADATA_SOURCE = BUNDLE_ROOT / "data" / "inputs" / "graph_metadata.json"

FRACTIONS = (10, 20, 30, 40, 50)
REPEATS = 5
BASE_SEED = 20261111
SOFT_WEIGHT_STRENGTH = 1.0
EDGE_HOLDOUT_RATIO = 0.6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GROUPS = {
    "GCN-Transformer_no": {"encoder": "hybrid", "pretrain": True, "soft": False, "hard": False},
    "GCN-Transformer_soft": {"encoder": "hybrid", "pretrain": True, "soft": True, "hard": False},
    "GCN-Transformer_hard": {"encoder": "hybrid", "pretrain": True, "soft": False, "hard": True},
    "GCN-Transformer_soft_hard": {"encoder": "hybrid", "pretrain": True, "soft": True, "hard": True},
}
BASELINE_GROUPS = ("GCN", "Transformer", "GCN-Transformer")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def append_metric(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    has_header = path.is_file() and path.stat().st_size > 0
    for attempt in range(120):
        try:
            pd.DataFrame([result]).to_csv(
                path, mode="a", header=not has_header, index=False,
                encoding="utf-8" if has_header else "utf-8-sig",
            )
            return
        except PermissionError:
            if attempt == 119:
                raise
            time.sleep(5)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GlobalTransformerBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, dropout: float, query_block: int):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden dimension must be divisible by attention heads")
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.query_block = query_block
        self.norm_attention = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, hidden * 3)
        self.output = nn.Linear(hidden, hidden)
        self.norm_feedforward = nn.LayerNorm(hidden)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
        )
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = dropout

    def forward(self, values):
        count = values.shape[0]
        qkv = self.qkv(self.norm_attention(values)).view(
            count, 3, self.heads, self.head_dim
        ).permute(1, 2, 0, 3)
        query, key, value = (part.unsqueeze(0) for part in qkv.unbind(dim=0))
        attended = []
        for start in range(0, count, self.query_block):
            query_chunk = query[:, :, start : start + self.query_block]
            if self.training and torch.is_grad_enabled():
                chunk = checkpoint(
                    self._attend_chunk, query_chunk, key, value, use_reentrant=False
                )
            else:
                chunk = self._attend_chunk(query_chunk, key, value)
            attended.append(chunk)
        output = torch.cat(attended, dim=2).squeeze(0).transpose(0, 1).reshape(
            count, self.hidden
        )
        values = values + self.dropout(self.output(output))
        return values + self.dropout(self.feedforward(self.norm_feedforward(values)))

    def _attend_chunk(self, query, key, value):
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )


class AblationBlock(nn.Module):
    def __init__(self, config: dict, mode: str):
        super().__init__()
        hidden = int(config["hidden_dim"])
        dropout = float(config["dropout"])
        self.mode = mode
        self.gcn = GCNConv(hidden, hidden, add_self_loops=True, normalize=True, cached=False)
        self.gcn_norm = nn.LayerNorm(hidden)
        self.transformer = GlobalTransformerBlock(
            hidden,
            int(config["attention_heads"]),
            dropout,
            int(config["attention_query_block"]),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, edge_index, edge_weight):
        if self.mode != "transformer":
            message = self.gcn(values, edge_index, edge_weight=edge_weight)
            values = self.gcn_norm(values + self.dropout(F.gelu(message)))
        if self.mode != "gcn":
            values = self.transformer(values)
        return values


class GraphProspectivityModel(nn.Module):
    def __init__(self, config: dict, encoder: str):
        super().__init__()
        self.config = config
        self.node_type_names = tuple(config["node_type_names"])
        self.type_specific_embeddings = nn.ModuleDict(
            {
                name: nn.Linear(int(config["input_dim"]), int(config["hidden_dim"]))
                for name in self.node_type_names
            }
        )
        self.node_type_embedding = nn.Embedding(
            int(config["node_type_count"]), int(config["hidden_dim"])
        )
        self.position_projection = nn.Linear(2, int(config["hidden_dim"]), bias=False)
        self.chemical_mask_token = nn.Parameter(torch.zeros(int(config["chemical_dim"])))
        self.geophysics_mask_token = nn.Parameter(torch.zeros(2))
        self.blocks = nn.ModuleList(
            [AblationBlock(config, encoder) for _ in range(int(config["encoder_layers"]))]
        )
        self.output_norm = nn.LayerNorm(int(config["hidden_dim"]))
        self.chemistry_head = nn.Linear(int(config["hidden_dim"]), int(config["chemical_dim"]))
        self.geophysics_head = nn.Linear(
            int(config["hidden_dim"]), int(config["geophysics_target_dim"])
        )
        self.edge_relation_head = nn.Linear(int(config["hidden_dim"]) * 2, 1)
        self.feature_mask_token = nn.Parameter(torch.zeros(int(config["input_dim"])))
        self.prospectivity_head = nn.Linear(int(config["hidden_dim"]), 1)
        self.knowledge_projections = nn.ModuleList(
            [nn.Linear(3, int(config["hidden_dim"])) for _ in self.blocks]
        )

    def forward(
        self,
        features,
        node_type,
        positions,
        edge_index,
        edge_weight,
        chemical_mask=None,
        geophysics_mask=None,
        feature_mask=None,
        constraint_layers=None,
        anchor_mask=None,
        inject_knowledge=False,
    ):
        if feature_mask is not None:
            if feature_mask.shape != features.shape:
                raise ValueError("feature_mask must match node feature shape")
            features = torch.where(
                feature_mask,
                self.feature_mask_token.to(features.dtype).unsqueeze(0),
                features,
            )
        if chemical_mask is not None:
            values = torch.where(
                chemical_mask,
                self.chemical_mask_token.to(features.dtype).unsqueeze(0),
                features[:, :39],
            )
            features = torch.cat([values, features[:, 39:]], dim=1)
        if geophysics_mask is not None:
            values = torch.where(
                geophysics_mask,
                self.geophysics_mask_token.to(features.dtype).unsqueeze(0),
                features[:, 39:41],
            )
            features = torch.cat([features[:, :39], values], dim=1)
        hidden = None
        for type_id, name in enumerate(self.node_type_names):
            ids = torch.nonzero(node_type == type_id, as_tuple=False).flatten()
            if ids.numel():
                type_embedding = self.type_specific_embeddings[name](features.index_select(0, ids))
                if hidden is None:
                    hidden = type_embedding.new_zeros(
                        (features.shape[0], int(self.config["hidden_dim"]))
                    )
                hidden = hidden.index_copy(0, ids, type_embedding)
        hidden = hidden + self.node_type_embedding(node_type) + self.position_projection(positions)
        if inject_knowledge:
            if constraint_layers is None or constraint_layers.shape != (features.shape[0], 3):
                raise ValueError("Knowledge injection requires three constraint values per node")
            if anchor_mask is None or anchor_mask.shape != (features.shape[0],):
                raise ValueError("Knowledge injection requires an anchor mask")
        for layer_index, block in enumerate(self.blocks):
            hidden = block(hidden, edge_index, edge_weight)
            if inject_knowledge:
                knowledge = self.knowledge_projections[layer_index](
                    constraint_layers.to(hidden.dtype)
                )
                hidden = hidden + knowledge * anchor_mask[:, None].to(hidden.dtype)
        embeddings = self.output_norm(hidden)
        logits = self.prospectivity_head(embeddings).squeeze(-1)
        return {
            "embedding": embeddings,
            "chemistry": self.chemistry_head(embeddings),
            "geophysics": self.geophysics_head(embeddings),
            "prospectivity_logit": logits,
        }

    def predict_edges(self, embeddings, edge_index):
        source, target = edge_index
        pair_features = torch.cat([embeddings[source], embeddings[target]], dim=-1)
        return self.edge_relation_head(pair_features).squeeze(-1)


def load_data():
    with np.load(GRAPH_SOURCE, allow_pickle=False) as archive:
        graph = {name: archive[name] for name in archive.files}
    metadata = json.loads(GRAPH_METADATA_SOURCE.read_text(encoding="utf-8"))
    model_config = json.loads(CONFIG_SOURCE.read_text(encoding="utf-8"))
    with np.load(BUNDLE_ROOT / "data" / "inputs" / "labels_targets_and_scores.npz", allow_pickle=False) as labels:
        positive_mask = labels["positive_mask"].astype(bool)
        target_id = labels["target_id"]
    constraints = pd.read_csv(CONSTRAINT_SOURCE).sort_values("anchor_id")
    n = len(positive_mask)
    if n != 15550 or len(constraints) != n:
        raise ValueError("Bundled anchor/constraint count changed")
    source_fields = ["k_contact", "k_intrusion", "k_carbonate"]
    source = constraints[source_fields].to_numpy(dtype=np.float64)
    if not np.isfinite(source).all() or np.any((source < 0) | (source > 1)):
        raise ValueError("The three source constraint layers must be finite values in [0, 1]")
    constraints["k_mean"] = source.mean(axis=1).astype(np.float32)
    if target_id.shape != (n,):
        raise ValueError("Bundled target labels are not aligned to anchor rows")
    positive_ids = np.flatnonzero(positive_mask)
    if len(positive_ids) != 36:
        raise ValueError(f"Expected 36 known positive anchors, found {len(positive_ids)}")
    if graph["node_features"].shape != (16148, 41) or graph["edge_index"].shape[1] != 165260:
        raise ValueError("Main graph dimensions differ from the saved experiment")
    return {
        "graph": graph,
        "metadata": metadata,
        "checkpoint": {"model_config": model_config},
        "finetune_config": {},
        "positive_ids": positive_ids.astype(np.int64),
        "positive_mask": positive_mask,
        "constraints": constraints,
    }


def tensor_data(data: dict):
    graph = data["graph"]
    config = data["checkpoint"]["model_config"]
    n_sites = len(data["positive_mask"])
    features = torch.as_tensor(graph["node_features"], dtype=torch.float32, device=DEVICE)
    node_type = torch.as_tensor(graph["node_type"], dtype=torch.long, device=DEVICE)
    positions_np = graph["positions"].astype(np.float32)
    pos_min = positions_np.min(axis=0)
    pos_span = np.maximum(positions_np.max(axis=0) - pos_min, 1.0)
    positions_np = 2.0 * (positions_np - pos_min) / pos_span - 1.0
    edge_index = torch.as_tensor(graph["edge_index"], dtype=torch.long, device=DEVICE)
    edge_type = graph["edge_type"].astype(np.int64)
    edge_weight = torch.as_tensor(graph["edge_weight"], dtype=torch.float32, device=DEVICE)
    constraint_np = np.zeros((len(node_type), 3), dtype=np.float32)
    constraint_np[:n_sites] = data["constraints"][["k_contact", "k_intrusion", "k_carbonate"]].to_numpy(dtype=np.float32)
    constraint_layers = torch.as_tensor(constraint_np, dtype=torch.float32, device=DEVICE)
    anchor_mask = torch.zeros(len(node_type), dtype=torch.bool, device=DEVICE)
    anchor_mask[:n_sites] = True
    return {
        "config": config,
        "features": features,
        "node_type": node_type,
        "node_type_np": graph["node_type"].astype(np.int64),
        "positions_np": graph["positions"].astype(np.float32),
        "edge_index_np": graph["edge_index"].astype(np.int64),
        "positions": torch.as_tensor(positions_np, dtype=torch.float32, device=DEVICE),
        "position_min": pos_min,
        "position_span": pos_span,
        "edge_index": edge_index,
        "edge_type_np": edge_type,
        "edge_weight": edge_weight,
        "constraint_layers": constraint_layers,
        "anchor_mask": anchor_mask,
        "n_sites": n_sites,
        "n_nodes": len(node_type),
        "geo_targets": features[:n_sites, 39:41].clone(),
    }


def nnpu_risk(pos_logits, unlabeled_logits, prior: float):
    positive_as_positive = F.softplus(-pos_logits).mean()
    positive_as_negative = F.softplus(pos_logits).mean()
    unlabeled_as_negative = F.softplus(unlabeled_logits).mean()
    negative_risk = unlabeled_as_negative - prior * positive_as_negative
    return prior * positive_as_positive + torch.clamp(negative_risk, min=0.0)


def setup_pretraining_edges(data: dict, td: dict):
    metadata = data["metadata"]
    rels = metadata["relations"]
    type_ids = {name: int(value) for name, value in metadata["node_type_ids"].items()}
    forward_rels = [
        rel for rel in rels
        if type_ids[rel["source_type"]] <= type_ids[rel["target_type"]]
        and int(rel.get("edge_count", 0)) > 0
    ]
    forward_ids = np.asarray([int(rel["relation_id"]) for rel in forward_rels], dtype=np.int64)
    edge_type = td["edge_type_np"]
    edge_index = td["edge_index_np"]
    mask = np.isin(edge_type, forward_ids)
    for rel in forward_rels:
        if rel["source_type"] == rel["target_type"]:
            rid = int(rel["relation_id"])
            mask &= (edge_type != rid) | (edge_index[0] < edge_index[1])
    forward_edges = edge_index[:, mask]
    forward_types = edge_type[mask]
    forward_weights = data["graph"]["edge_weight"][mask].astype(np.float32)
    inverse = np.asarray([int(rel["inverse_relation_id"]) for rel in rels], dtype=np.int64)
    rel_specs = {
        int(rel["relation_id"]): (type_ids[rel["source_type"]], type_ids[rel["target_type"]])
        for rel in rels
    }
    nodes_by_type = {
        int(kind): np.flatnonzero(td["node_type_np"] == kind)
        for kind in range(len(type_ids))
    }
    observed = set(
        (edge_index[0].astype(np.int64) * td["n_nodes"] + edge_index[1]).tolist()
    )
    return forward_ids, forward_edges, forward_types, forward_weights, inverse, rel_specs, nodes_by_type, observed


def sample_negative_edges(positive_types, rng, rel_specs, nodes_by_type, n_nodes, observed):
    src_out, dst_out = [], []
    used = set()
    for relation_id in np.unique(positive_types):
        count = int(np.count_nonzero(positive_types == relation_id))
        source_type, target_type = rel_specs[int(relation_id)]
        src_pool, dst_pool = nodes_by_type[source_type], nodes_by_type[target_type]
        found = 0
        while found < count:
            batch = max(64, (count - found) * 2)
            src = rng.choice(src_pool, size=batch, replace=True)
            dst = rng.choice(dst_pool, size=batch, replace=True)
            for a, b in zip(src, dst):
                if a == b:
                    continue
                key = int(a) * n_nodes + int(b)
                if key in observed or key in used:
                    continue
                used.add(key)
                src_out.append(int(a))
                dst_out.append(int(b))
                found += 1
                if found >= count:
                    break
    return np.asarray(src_out, dtype=np.int64), np.asarray(dst_out, dtype=np.int64)


def minmax(values):
    low, high = values.min(), values.max()
    return torch.zeros_like(values) if float((high - low).detach().cpu()) <= 0 else (values - low) / (high - low)


def pretrain_one(data: dict, td: dict, repeat: int, soft: bool, hard: bool, encoder: str, out_dir: Path):
    config = td["config"]
    seed = BASE_SEED + repeat
    seed_all(seed)
    model = GraphProspectivityModel(config, encoder).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    _, forward_edges_np, forward_types_np, forward_weights_np, inverse_relation_ids, rel_specs, nodes_by_type, observed = setup_pretraining_edges(data, td)
    forward_edges = torch.as_tensor(forward_edges_np, dtype=torch.long, device=DEVICE)
    forward_types = torch.as_tensor(forward_types_np, dtype=torch.long, device=DEVICE)
    if len(forward_weights_np) != forward_edges_np.shape[1]:
        raise ValueError("Pretraining edge weights do not align with selected edges")
    forward_weights = torch.as_tensor(forward_weights_np, dtype=torch.float32, device=DEVICE)
    rng = np.random.default_rng(seed)
    history = []
    n_nodes = td["n_nodes"]
    n_sites = td["n_sites"]
    if td["features"].shape[1] != 41:
        raise ValueError(f"Expected 41 node input features, found {td['features'].shape[1]}")
    inverse_relation_ids = torch.as_tensor(inverse_relation_ids, dtype=torch.long, device=DEVICE)
    for epoch in range(PRETRAIN_EPOCHS):
        model.train()
        held = torch.rand(len(forward_types), device=DEVICE) < EDGE_HOLDOUT_RATIO
        keep = ~held
        kept = forward_edges[:, keep]
        msg_edges = torch.cat([kept, kept.flip(0)], dim=1)
        msg_weights = torch.cat([forward_weights[keep], forward_weights[keep]])
        held_edges = forward_edges[:, held]
        if held_edges.shape[1] == 0:
            raise RuntimeError("No positive edges were held out for edge reconstruction")
        held_types = forward_types[held]
        positive_edges = torch.cat([held_edges, held_edges.flip(0)], dim=1)
        positive_types = torch.cat([held_types, inverse_relation_ids[held_types]])
        neg_src, neg_dst = sample_negative_edges(
            positive_types.detach().cpu().numpy(), rng, rel_specs, nodes_by_type, n_nodes, observed
        )
        edge_pairs = np.column_stack([
            np.concatenate([positive_edges[0].cpu().numpy(), neg_src]),
            np.concatenate([positive_edges[1].cpu().numpy(), neg_dst]),
        ])
        candidate_edges = torch.as_tensor(edge_pairs.T, dtype=torch.long, device=DEVICE)
        labels = torch.cat([
            torch.ones(len(positive_types), dtype=torch.float32, device=DEVICE),
            torch.zeros(len(neg_src), dtype=torch.float32, device=DEVICE),
        ])
        feature_mask = torch.zeros_like(td["features"], dtype=torch.bool)
        masked_channels = torch.rand((n_nodes, 41), device=DEVICE).topk(25, dim=1).indices
        feature_mask.scatter_(1, masked_channels, True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
            edge_outputs = model(
                td["features"], td["node_type"], td["positions"], msg_edges, msg_weights,
                constraint_layers=td["constraint_layers"], anchor_mask=td["anchor_mask"],
                inject_knowledge=hard,
            )
            edge_logits = model.predict_edges(edge_outputs["embedding"], candidate_edges)
            edge_loss = F.binary_cross_entropy_with_logits(edge_logits.float(), labels)
            node_outputs = model(
                td["features"], td["node_type"], td["positions"], td["edge_index"], td["edge_weight"],
                feature_mask=feature_mask,
                constraint_layers=td["constraint_layers"], anchor_mask=td["anchor_mask"],
                inject_knowledge=hard,
            )
            reconstructed = torch.cat([node_outputs["chemistry"], node_outputs["geophysics"]], dim=1).float()
            targets = td["features"].float()
            per_node_error = (
                torch.square(reconstructed - targets) * feature_mask
            ).sum(dim=1) / feature_mask.sum(dim=1).clamp_min(1)
            masked_nodes = feature_mask.any(dim=1)
            node_weights = torch.ones_like(per_node_error)
            soft_loss = torch.zeros((), device=DEVICE)
            if soft:
                anchor_constraint_mean = td["constraint_layers"][:n_sites].mean(dim=1)
                raw_anchor_weights = 1.0 + SOFT_WEIGHT_STRENGTH * anchor_constraint_mean
                node_weights[:n_sites] = raw_anchor_weights
                node_weights = node_weights / node_weights[masked_nodes].mean().clamp_min(1e-8)
                anchor_masked = masked_nodes & td["anchor_mask"]
                soft_loss = (
                    node_weights[anchor_masked] * per_node_error[anchor_masked]
                ).sum() / node_weights[anchor_masked].sum().clamp_min(1e-8)
            node_loss = (
                node_weights[masked_nodes] * per_node_error[masked_nodes]
            ).sum() / node_weights[masked_nodes].sum().clamp_min(1e-8)
            loss = edge_loss + node_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        history.append({"epoch": epoch + 1, "total": float(loss.detach().float().cpu()), "edge_relation": float(edge_loss.detach().float().cpu()), "node_feature_mse": float(node_loss.detach().float().cpu()), "soft_weighted_anchor_feature_mse": float(soft_loss.detach().float().cpu()), "masked_nodes": int(masked_nodes.sum()), "masked_anchors": int((masked_nodes & td['anchor_mask']).sum())})
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"pretrain repeat={repeat:02d} epoch={epoch+1:03d}/{PRETRAIN_EPOCHS} edge={history[-1]['edge_relation']:.5f} node={history[-1]['node_feature_mse']:.5f} soft_anchor={history[-1]['soft_weighted_anchor_feature_mse']:.5f}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_config": config, "position_min": td["position_min"], "position_span": td["position_span"], "seed": seed, "soft_constraint": soft, "hard_constraint": hard, "pretraining_objectives": ["edge_relation_reconstruction", "per-node-feature-masking-25-of-41"], "edge_mask_rate_undirected": EDGE_HOLDOUT_RATIO, "soft_weight_strength": SOFT_WEIGHT_STRENGTH}, out_dir / "pretrained.pt")
    pd.DataFrame(history).to_csv(out_dir / "pretraining_history.csv", index=False, encoding="utf-8-sig")
    del model, optimizer
    torch.cuda.empty_cache()


def finetune_one(data: dict, td: dict, group: str, settings: dict, fraction: int, repeat: int, selected_ids: np.ndarray, pretrain_path: Path | None, run_dir: Path):
    seed = BASE_SEED + 10000 + repeat
    seed_all(seed)
    model = GraphProspectivityModel(td["config"], settings["encoder"]).to(DEVICE)
    if pretrain_path is not None:
        saved = torch.load(pretrain_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["state_dict"], strict=True)
        position_min = np.asarray(saved["position_min"], dtype=np.float32)
        position_span = np.asarray(saved["position_span"], dtype=np.float32)
        pos = 2.0 * (data["graph"]["positions"].astype(np.float32) - position_min) / position_span - 1.0
        positions = torch.as_tensor(pos, dtype=torch.float32, device=DEVICE)
    else:
        positions = td["positions"]
    encoder_parameters = (
        list(model.type_specific_embeddings.parameters())
        + list(model.node_type_embedding.parameters())
        + list(model.position_projection.parameters())
        + list(model.blocks.parameters())
        + list(model.knowledge_projections.parameters())
        + list(model.output_norm.parameters())
    )
    classifier_parameters = list(model.prospectivity_head.parameters())
    optimizer = torch.optim.AdamW(
        [{"params": encoder_parameters, "lr": 5e-4}, {"params": classifier_parameters, "lr": 5e-2}],
        weight_decay=1e-4,
    )
    positive_ids = torch.as_tensor(selected_ids, dtype=torch.long, device=DEVICE)
    selected_mask = np.zeros(td["n_sites"], dtype=bool)
    selected_mask[selected_ids] = True
    unlabeled_ids = torch.as_tensor(np.flatnonzero(~selected_mask), dtype=torch.long, device=DEVICE)
    history = []
    model_seed = BASE_SEED + 10000 + repeat
    for epoch in range(FINETUNE_EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
            outputs = model(
                td["features"], td["node_type"], positions, td["edge_index"], td["edge_weight"],
                constraint_layers=td["constraint_layers"], anchor_mask=td["anchor_mask"],
                inject_knowledge=settings["hard"],
            )
            logits = outputs["prospectivity_logit"][:td["n_sites"]].float()
            loss = nnpu_risk(logits[positive_ids], logits[unlabeled_ids], 0.02)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder_parameters + classifier_parameters, 5.0)
        optimizer.step()
        history.append({"epoch": epoch + 1, "train_nnpu_risk": float(loss.detach().float().cpu())})
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
        logits = model(
            td["features"], td["node_type"], positions, td["edge_index"], td["edge_weight"],
            constraint_layers=td["constraint_layers"], anchor_mask=td["anchor_mask"],
            inject_knowledge=settings["hard"],
        )["prospectivity_logit"][:td["n_sites"]].float()
        scores = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
    auc = float(roc_auc_score(data["positive_mask"], scores))
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(run_dir / "scores_and_labels.npz", scores=scores, positive_mask=data["positive_mask"], sampled_positive_ids=selected_ids)
    pd.DataFrame(history).to_csv(run_dir / "finetuning_history.csv", index=False, encoding="utf-8-sig")
    result = {"group": group, "fraction_percent": fraction, "sampled_positive_count": len(selected_ids), "repeat": repeat, "seed": model_seed, "roc_auc_resubstitution": auc, "finetune_epochs": len(history)}
    write_json(run_dir / "metrics.json", result)
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def fraction_counts(n_positive: int):
    return {pct: max(1, min(n_positive, int(math.floor(n_positive * pct / 100 + 0.5)))) for pct in FRACTIONS}


def plot_results(raw: pd.DataFrame, summary: pd.DataFrame):
    colors = {
        "GCN-Transformer_no": "#276FBF", "GCN-Transformer_soft": "#D1495B",
        "GCN-Transformer_hard": "#4C956C", "GCN-Transformer_soft_hard": "#E09F3E",
    }
    labels = {
        "GCN-Transformer_no": "GCN-T 自监督，无约束",
        "GCN-Transformer_soft": "GCN-T 自监督，单层软约束",
        "GCN-Transformer_hard": "GCN-T 自监督，硬约束",
        "GCN-Transformer_soft_hard": "GCN-T 自监督，软硬约束",
    }
    plt.rcParams.update({"font.family": ["Microsoft YaHei", "DejaVu Sans"], "font.size": 10})
    fig, ax = plt.subplots(figsize=(10, 6))
    for group in GROUPS:
        part = summary[summary.group == group].sort_values("fraction_percent")
        x, y, sd = part.fraction_percent.to_numpy(), part["mean"].to_numpy(), part["standard_deviation"].to_numpy()
        ax.plot(x, y, marker="o", linewidth=2, color=colors[group], label=labels[group])
        ax.fill_between(x, y - sd, y + sd, color=colors[group], alpha=0.14)
    ax.set(xlabel="训练正样本抽样比例 (%)", ylabel="样本内 ROC-AUC（均值 ± 标准差）", title="自监督图 Transformer 在不同正样本比例下的排序表现", xlim=(5, 105), ylim=(0.45, 1.01), xticks=list(FRACTIONS))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.text(0.5, 0.01, "每个比例 10 次配对随机抽样；全部已知矿床参与 AUC，包含当次训练正样本。", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(SUMMARY / "roc_auc_mean_sd.png", dpi=300)
    fig.savefig(SUMMARY / "roc_auc_mean_sd.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for group in GROUPS:
        part = summary[summary.group == group].sort_values("fraction_percent")
        ax.plot(part.fraction_percent, part.variance, marker="o", linewidth=1.8, color=colors[group], label=labels[group])
    ax.set(xlabel="训练正样本抽样比例 (%)", ylabel="ROC-AUC 样本方差（10 次重复）", title="ROC-AUC 方差随正样本比例的变化", xlim=(5, 105), xticks=list(FRACTIONS))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(SUMMARY / "roc_auc_variance.png", dpi=300)
    fig.savefig(SUMMARY / "roc_auc_variance.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(16, 7), sharey=True)
    for index, fraction in enumerate(FRACTIONS):
        ax = axes.flat[index]
        values = [raw.loc[(raw.group == group) & (raw.fraction_percent == fraction), "roc_auc_resubstitution"].to_numpy() for group in GROUPS]
        boxes = ax.boxplot(values, patch_artist=True, showfliers=False)
        for box, group in zip(boxes["boxes"], GROUPS):
            box.set_facecolor(colors[group])
            box.set_alpha(0.55)
        count = fraction_counts(36)[fraction]
        ax.set_title(f"{fraction}% (n={count})")
        ax.set_xticks(range(1, len(GROUPS) + 1), ["no", "soft", "hard", "soft+hard"], rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.22)
        if index in (0, 5):
            ax.set_ylabel("重代入 ROC-AUC")
    fig.suptitle("新自监督 GCN-Transformer 约束消融：各正样本比例的 10 次 AUC 分布")
    fig.tight_layout()
    fig.savefig(SUMMARY / "roc_auc_repeat_distributions.png", dpi=300)
    fig.savefig(SUMMARY / "roc_auc_repeat_distributions.pdf")
    plt.close(fig)


def compare_with_hybrid_baselines(raw_new: pd.DataFrame) -> None:
    baseline = pd.read_csv(FINAL_ROOT / "对照试验" / "summary" / "run_metrics.csv")
    old_pretrained = baseline[baseline["group"].isin(GROUPS)].copy()
    old_pretrained["group"] = old_pretrained["group"].map(lambda value: f"{value}_80ep")
    supervised = baseline[baseline["group"] == "GCN-Transformer"].copy()
    current = raw_new.copy()
    current_plot = current.copy()
    current_plot["group"] = current_plot["group"].map(lambda value: f"{value}_240ep")
    combined = pd.concat([supervised, old_pretrained, current_plot], ignore_index=True, sort=False)
    summary = combined.groupby(["group", "fraction_percent", "sampled_positive_count"])["roc_auc_resubstitution"].agg(
        mean="mean", variance="var", standard_deviation="std", n="count"
    ).reset_index()
    summary.to_csv(SUMMARY / "comparison_with_gcn_transformer.csv", index=False, encoding="utf-8-sig")

    epoch_rows = []
    for new_group in GROUPS:
        new_rows = current[current["group"] == new_group]
        old_rows = old_pretrained[old_pretrained["group"] == f"{new_group}_80ep"]
        for fraction in FRACTIONS:
            newer = new_rows[new_rows["fraction_percent"] == fraction].sort_values("repeat")
            older = old_rows[old_rows["fraction_percent"] == fraction].sort_values("repeat")
            if len(newer) != REPEATS or len(older) != REPEATS:
                raise ValueError(f"Incomplete 240-vs-80 paired comparison: {new_group}, {fraction}%")
            delta = newer["roc_auc_resubstitution"].to_numpy() - older["roc_auc_resubstitution"].to_numpy()
            epoch_rows.append({
                "model_group": new_group,
                "comparison": "240_epoch_minus_80_epoch",
                "fraction_percent": fraction,
                "mean_auc_difference_240_minus_80": float(np.mean(delta)),
                "sd_of_paired_differences": float(np.std(delta, ddof=1)),
                "240_epoch_wins": int(np.sum(delta > 0)),
                "ties": int(np.sum(delta == 0)),
                "paired_repeats": len(delta),
            })
    pd.DataFrame(epoch_rows).to_csv(SUMMARY / "paired_auc_differences_240_vs_80.csv", index=False, encoding="utf-8-sig")

    soft_rows = []
    no_soft = current[current["group"] == "GCN-Transformer_no"]
    with_soft = current[current["group"] == "GCN-Transformer_"]
    for fraction in FRACTIONS:
        newer = with_soft[with_soft["fraction_percent"] == fraction].sort_values("repeat")
        older = no_soft[no_soft["fraction_percent"] == fraction].sort_values("repeat")
        delta = newer["roc_auc_resubstitution"].to_numpy() - older["roc_auc_resubstitution"].to_numpy()
        soft_rows.append({
            "model_group": "GCN-Transformer_",
            "baseline_group": "GCN-Transformer_no",
            "fraction_percent": fraction,
            "mean_auc_difference_new_minus_baseline": float(np.mean(delta)),
            "sd_of_paired_differences": float(np.std(delta, ddof=1)),
            "new_group_wins": int(np.sum(delta > 0)),
            "ties": int(np.sum(delta == 0)),
            "paired_repeats": len(delta),
        })
    pd.DataFrame(soft_rows).to_csv(SUMMARY / "paired_auc_differences_soft_vs_no_soft_240.csv", index=False, encoding="utf-8-sig")

    colors = {
        "GCN-Transformer": "#4C956C", "GCN-Transformer_no_80ep": "#8CBCE0", "GCN-Transformer__80ep": "#E99AA3",
        "GCN-Transformer_no_240ep": "#276FBF", "GCN-Transformer__240ep": "#D1495B",
    }
    labels = {
        "GCN-Transformer_no_80ep": "图 Transformer 自监督 无软约束 80轮",
        "GCN-Transformer__80ep": "图 Transformer 自监督 有软约束 80轮",
        "GCN-Transformer_no_240ep": "图 Transformer 自监督 无软约束 240轮",
        "GCN-Transformer__240ep": "图 Transformer 自监督 有软约束 240轮",
    }
    plt.rcParams.update({"font.family": ["Microsoft YaHei", "DejaVu Sans"], "font.size": 10})
    fig, ax = plt.subplots(figsize=(10, 6))
    for group in ("GCN-Transformer", "GCN-Transformer_no_80ep", "GCN-Transformer__80ep", "GCN-Transformer_no_240ep", "GCN-Transformer__240ep"):
        part = summary[summary["group"] == group].sort_values("fraction_percent")
        ax.plot(part["fraction_percent"], part["mean"], marker="o", linewidth=1.8, color=colors[group], label=labels.get(group, group))
        ax.fill_between(part["fraction_percent"], part["mean"] - part["standard_deviation"], part["mean"] + part["standard_deviation"], color=colors[group], alpha=0.10)
    ax.set(xlabel="训练正样本抽样比例 (%)", ylabel="样本内 ROC-AUC（均值 ± 标准差）", title="自监督图 Transformer：80轮与240轮微调对比", xlim=(5, 105), ylim=(0.45, 1.01), xticks=list(FRACTIONS))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.text(0.5, 0.01, "各组每个比例 10 次相同子集配对；AUC 含当次训练正样本，为样本内重代入评价。", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(SUMMARY / "graph_transformer_80_vs_240_mean_sd.png", dpi=300)
    fig.savefig(SUMMARY / "graph_transformer_80_vs_240_mean_sd.pdf")
    plt.close(fig)


def compare_direct_baselines(raw_new: pd.DataFrame) -> None:
    baseline_path = FINAL_ROOT / "对照试验" / "summary" / "run_metrics.csv"
    baseline = pd.read_csv(baseline_path)
    expected_groups = ["GCN", "Transformer", "GCN-Transformer"]
    baseline = baseline[baseline["group"].isin(expected_groups)].copy()
    baseline = baseline[baseline["fraction_percent"].isin(FRACTIONS)]
    expected = len(FRACTIONS) * REPEATS
    for group in expected_groups:
        subset = baseline[baseline["group"] == group]
        if len(subset) != expected:
            raise ValueError(f"Baseline {group} has {len(subset)} rows; expected {expected}")
    new = raw_new.copy()
    combined = pd.concat([baseline, new], ignore_index=True, sort=False)
    summary = combined.groupby(["group", "fraction_percent", "sampled_positive_count"])["roc_auc_resubstitution"].agg(
        mean="mean", variance="var", standard_deviation="std", n="count"
    ).reset_index()
    summary.to_csv(SUMMARY / "comparison_direct_baselines_summary.csv", index=False, encoding="utf-8-sig")

    colors = {
        "GCN": "#D1495B", "Transformer": "#276FBF", "GCN-Transformer": "#4C956C",
        **{"GCN-Transformer_no": "#7A6F9B", "GCN-Transformer_soft": "#E09F3E",
           "GCN-Transformer_hard": "#1B9E77", "GCN-Transformer_soft_hard": "#8C564B"},
    }
    labels = {
        "GCN": "GCN", "Transformer": "Transformer", "GCN-Transformer": "GCN-T 直接监督",
        "GCN-Transformer_no": "GCN-T 自监督 no", "GCN-Transformer_soft": "GCN-T 自监督 soft",
        "GCN-Transformer_hard": "GCN-T 自监督 hard", "GCN-Transformer_soft_hard": "GCN-T 自监督 soft+hard",
    }
    plt.rcParams.update({"font.family": ["Microsoft YaHei", "DejaVu Sans"], "font.size": 10})
    fig, ax = plt.subplots(figsize=(10, 6))
    for group in expected_groups + list(GROUPS):
        part = summary[summary["group"] == group].sort_values("fraction_percent")
        ax.plot(part["fraction_percent"], part["mean"], marker="o", linewidth=1.8, color=colors[group], label=labels[group])
        ax.fill_between(part["fraction_percent"], part["mean"] - part["standard_deviation"], part["mean"] + part["standard_deviation"], color=colors[group], alpha=0.10)
    ax.set(xlabel="训练正样本抽样比例 (%)", ylabel="重代入 ROC-AUC（均值 ± 标准差）", title="GCN-Transformer 直接监督与新版自监督约束消融", xlim=(5, 105), ylim=(0.45, 1.01), xticks=list(FRACTIONS))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.text(0.5, 0.01, "每档 10 个配对随机种子；ROC-AUC 包含当次训练正样本，属于样本内重代入评价。", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(SUMMARY / "comparison_all_models_mean_sd.png", dpi=300)
    fig.savefig(SUMMARY / "comparison_all_models_mean_sd.pdf")
    fig, axes = plt.subplots(2, 5, figsize=(20, 9), sharey=True)
    ordered_groups = expected_groups + list(GROUPS)
    for index, fraction in enumerate(FRACTIONS):
        axis = axes.flat[index]
        values = [combined.loc[(combined.group == group) & (combined.fraction_percent.astype(int) == fraction), "roc_auc_resubstitution"].to_numpy(dtype=float) for group in ordered_groups]
        boxes = axis.boxplot(values, patch_artist=True, showfliers=False)
        for box, group in zip(boxes["boxes"], ordered_groups):
            box.set_facecolor(colors[group])
            box.set_alpha(0.6)
        axis.set_title(f"{fraction}% (n={fraction_counts(36)[fraction]})")
        axis.set_xticks(range(1, len(ordered_groups) + 1), ["GCN", "Transf.", "GCN-T", "no", "soft", "hard", "soft+hard"], rotation=45, ha="right")
        axis.grid(axis="y", alpha=0.22)
        if index in (0, 5):
            axis.set_ylabel("重代入 ROC-AUC")
    fig.suptitle("七组模型每个比例下的 10 次 AUC 分布")
    fig.tight_layout()
    fig.savefig(SUMMARY / "comparison_all_models_repeat_distributions.png", dpi=300)
    fig.savefig(SUMMARY / "comparison_all_models_repeat_distributions.pdf")
    plt.close(fig)

def make_requested_comparisons(raw_new: pd.DataFrame) -> None:
    old_path = SUMMARY / "historical_20261110" / "run_metrics.csv"
    old = pd.read_csv(old_path)
    old = old[old.group.isin(GROUPS) & old.fraction_percent.isin(FRACTIONS)].copy()
    old["version"] = "20261110旧版"
    new = raw_new[raw_new.fraction_percent.isin(FRACTIONS)].copy()
    new["version"] = "20261111新版"
    colors = {
        "GCN-Transformer_no": "#276FBF", "GCN-Transformer_soft": "#D1495B",
        "GCN-Transformer_hard": "#4C956C", "GCN-Transformer_soft_hard": "#E09F3E",
    }
    labels = {
        "GCN-Transformer_no": "no", "GCN-Transformer_soft": "soft",
        "GCN-Transformer_hard": "hard", "GCN-Transformer_soft_hard": "soft+hard",
    }
    plt.rcParams.update({"font.family": ["Microsoft YaHei", "DejaVu Sans"], "font.size": 10})

    version_data = pd.concat([old, new], ignore_index=True, sort=False)
    version_summary = version_data.groupby(
        ["version", "group", "fraction_percent", "sampled_positive_count"]
    )["roc_auc_resubstitution"].agg(mean="mean", standard_deviation="std", n="count").reset_index()
    version_summary.to_csv(SUMMARY / "comparison_vs_20261110_summary.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, group in zip(axes.flat, GROUPS):
        for version, style in (("20261110旧版", "--"), ("20261111新版", "-")):
            part = version_summary[(version_summary.version == version) & (version_summary.group == group)].sort_values("fraction_percent")
            ax.plot(part.fraction_percent, part["mean"], marker="o", linestyle=style, color=colors[group], label=f"{version} (n={int(part.n.iloc[0])})")
            ax.fill_between(part.fraction_percent, part["mean"] - part.standard_deviation, part["mean"] + part.standard_deviation, color=colors[group], alpha=0.10)
        ax.set_title(labels[group])
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("训练正样本抽样比例 (%)")
    for ax in axes[:, 0]:
        ax.set_ylabel("样本内 ROC-AUC（均值 ± 标准差）")
    fig.suptitle("新旧知识注入方法对比：按各自重复次数汇总")
    fig.text(0.5, 0.01, "20261111 每档 5 次，20261110 每档 10 次；AUC 包含训练正样本，属于样本内重代入评价。", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(SUMMARY / "comparison_new_vs_20261110.png", dpi=300)
    fig.savefig(SUMMARY / "comparison_new_vs_20261110.pdf")
    plt.close(fig)

    baseline_path = FINAL_ROOT / "对照试验" / "summary" / "run_metrics.csv"
    baseline = pd.read_csv(baseline_path)
    baseline_groups = ["GCN", "Transformer", "GCN-Transformer"]
    baseline = baseline[baseline.group.isin(baseline_groups) & baseline.fraction_percent.isin(FRACTIONS)].copy()
    baseline["comparison_group"] = baseline["group"]
    baseline["source_repeats"] = 10
    new_baselines = new.copy()
    new_baselines["comparison_group"] = new_baselines["group"]
    new_baselines["source_repeats"] = REPEATS
    historical_data = pd.concat([baseline, new_baselines], ignore_index=True, sort=False)
    historical_summary = historical_data.groupby(
        ["comparison_group", "fraction_percent", "sampled_positive_count"]
    )["roc_auc_resubstitution"].agg(mean="mean", standard_deviation="std", n="count").reset_index()
    historical_summary.to_csv(SUMMARY / "comparison_vs_historical_baselines_summary.csv", index=False, encoding="utf-8-sig")
    baseline_colors = {
        "GCN": "#D1495B", "Transformer": "#276FBF", "GCN-Transformer": "#4C956C",
        **colors,
    }
    baseline_labels = {
        "GCN": "GCN", "Transformer": "Transformer", "GCN-Transformer": "GCN-T 直接监督",
        **{group: f"GCN-T 新版 {labels[group]}" for group in GROUPS},
    }
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for group in baseline_groups + list(GROUPS):
        part = historical_summary[historical_summary.comparison_group == group].sort_values("fraction_percent")
        ax.plot(part.fraction_percent, part["mean"], marker="o", linewidth=1.8, color=baseline_colors[group], label=baseline_labels[group])
        ax.fill_between(part.fraction_percent, part["mean"] - part.standard_deviation, part["mean"] + part.standard_deviation, color=baseline_colors[group], alpha=0.09)
    ax.set(xlabel="训练正样本抽样比例 (%)", ylabel="样本内 ROC-AUC（均值 ± 标准差）", title="新版知识注入方法与历史模型基线对比", xlim=(8, 52), ylim=(0.35, 1.01), xticks=list(FRACTIONS))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.text(0.5, 0.01, "历史基线各档 10 次，新版各档 5 次；不同版本独立实验、非配对。AUC 为样本内重代入评价。", ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(SUMMARY / "comparison_new_vs_historical_baselines.png", dpi=300)
    fig.savefig(SUMMARY / "comparison_new_vs_historical_baselines.pdf")
    plt.close(fig)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("主实验使用 CUDA；对照实验要求可用 CUDA 设备")
    SUMMARY.mkdir(parents=True, exist_ok=True)
    data = load_data()
    td = tensor_data(data)
    counts = fraction_counts(len(data["positive_ids"]))
    data["constraints"].to_csv(SUMMARY / "anchor_constraint_layers.csv", index=False, encoding="utf-8-sig")
    graph_hash = sha256(GRAPH_SOURCE)
    config_file = json.loads(CONFIG_SOURCE.read_text(encoding="utf-8"))
    write_json(SUMMARY / "experiment_manifest.json", {
        "reference_main_experiment": ".", "device": torch.cuda.get_device_name(0),
        "graph_file": GRAPH_SOURCE.as_posix(), "graph_sha256": graph_hash,
        "main_experiment_config": config_file, "model_config": td["config"],
        "pretrain_epochs": PRETRAIN_EPOCHS, "finetune_epochs": FINETUNE_EPOCHS, "early_stopping": False,
        "pretraining_reused_without_retraining": False, "nnpu_prior": 0.02,
        "soft_anchor_weight_strength": SOFT_WEIGHT_STRENGTH, "edge_holdout_ratio": EDGE_HOLDOUT_RATIO,
        "feature_mask_channels_per_node": 25,
        "constraint_layers": {
            "source_fields": ["k_contact", "k_intrusion", "k_carbonate"],
            "source_range": [0.0, 1.0],
            "hard_injection": "per-encoder-layer Linear(3, hidden_dim) projection, residual addition at anchor nodes only",
            "soft_anchor_weight": "normalized node weight proportional to 1 + mean(k_contact, k_intrusion, k_carbonate)",
            "mean_min": float(data["constraints"]["k_mean"].min()),
            "mean_max": float(data["constraints"]["k_mean"].max()),
        },
        "pretraining_objectives": [
            "edge relation reconstruction with 60% paired undirected-edge masking and type-matched negative edges",
            "per-node masking of exactly 25 of 41 feature channels; reconstruct masked node-feature cells only",
            "soft constraint: weight each masked anchor's per-node feature reconstruction loss by its three-layer mean constraint value",
            "hard constraint: inject a learned three-layer geological representation residually after every encoder layer at anchors during SSL and nnPU",
        ],
        "forward_pass_design": "Separate forwards for edge relation prediction and per-node feature reconstruction; task losses are summed after independent predictions. Soft weighting is used only in SSL; hard injection remains active in SSL and nnPU.",
        "anchors": td["n_sites"], "known_positive_anchors": len(data["positive_ids"]),
        "percentages": list(FRACTIONS), "sampled_positive_counts": counts, "repeats": REPEATS,
        "sampling_design": "Within each repeat, one random permutation of all 36 known positive anchors; each percentage uses a nested prefix. All groups use the same subset and repeat seed.",
        "auc_definition": "Resubstitution ROC-AUC using all 36 known positive anchors versus all remaining anchors. Training positives are included in evaluation; remaining anchors are unlabeled locations treated as operational negatives, not confirmed barren locations.",
        "groups": GROUPS,
        "comparison_baselines": BASELINE_GROUPS,
        "source_scope": "Training inputs and outputs are resolved within this bundle.",
    })

    metrics_path = SUMMARY / "run_metrics.csv"
    existing = pd.read_csv(metrics_path).to_dict("records") if metrics_path.is_file() else []
    completed = {(r["group"], int(r["fraction_percent"]), int(r["repeat"])) for r in existing}
    results = list(existing)
    write_json(SUMMARY / "status.json", {"status": "running", "completed_runs": len(completed), "expected_runs": len(GROUPS) * len(FRACTIONS) * REPEATS})
    pretrain_paths: dict[tuple[str, int], Path] = {}
    permutations = {
        repeat: np.random.default_rng(BASE_SEED + repeat).permutation(data["positive_ids"])
        for repeat in range(1, REPEATS + 1)
    }

    # Pretrain once per repeat and variant; reuse across nested positive fractions.
    for repeat in range(1, REPEATS + 1):
        for group in GROUPS:
            settings = GROUPS[group]
            path = SUMMARY / "pretraining" / group / f"repeat_{repeat:02d}"
            checkpoint = path / "pretrained.pt"
            if not checkpoint.is_file():
                path.mkdir(parents=True, exist_ok=True)
                pretrain_one(data, td, repeat, settings["soft"], settings["hard"], settings["encoder"], path)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if int(saved.get("seed", -1)) != BASE_SEED + repeat:
                raise ValueError(f"Unexpected pretraining seed in {checkpoint}")
            if saved.get("model_config") != td["config"] or saved.get("pretraining_objectives") != ["edge_relation_reconstruction", "per-node-feature-masking-25-of-41"] or saved.get("hard_constraint") != settings["hard"] or saved.get("soft_constraint") != settings["soft"]:
                raise ValueError(f"Pretraining checkpoint has an unexpected model or objective in {checkpoint}")
            history_path = path / "pretraining_history.csv"
            if not history_path.is_file() or len(pd.read_csv(history_path)) != PRETRAIN_EPOCHS:
                raise ValueError(f"Expected {PRETRAIN_EPOCHS} completed pretraining epochs in {history_path}")
            pretrain_paths[(group, repeat)] = checkpoint

    for group, settings in GROUPS.items():
        for fraction in FRACTIONS:
            n_sampled = counts[fraction]
            for repeat in range(1, REPEATS + 1):
                key = (group, fraction, repeat)
                if key in completed:
                    continue
                selected = permutations[repeat][:n_sampled]
                run_dir = SUMMARY / "runs" / group / f"positive_{fraction:03d}" / f"repeat_{repeat:02d}"
                pretrain_path = pretrain_paths[(group, repeat)]
                start = time.time()
                result = finetune_one(data, td, group, settings, fraction, repeat, selected, pretrain_path, run_dir)
                result["elapsed_seconds"] = time.time() - start
                results.append(result)
                completed.add(key)
                append_metric(metrics_path, result)
                print(f"{group} {fraction:3d}% ({n_sampled:2d} pos) repeat={repeat:02d}/{REPEATS} AUC={result['roc_auc_resubstitution']:.5f}", flush=True)

    raw = pd.DataFrame(results)
    expected = len(GROUPS) * len(FRACTIONS) * REPEATS
    write_json(SUMMARY / "status.json", {"status": "running", "completed_runs": len(raw), "expected_runs": expected})
    if len(raw) != expected:
        raise RuntimeError(f"Expected {expected} runs, found {len(raw)}")
    summary = raw.groupby(["group", "fraction_percent", "sampled_positive_count"])["roc_auc_resubstitution"].agg(mean="mean", variance="var", standard_deviation="std", n="count").reset_index()
    summary.to_csv(SUMMARY / "auc_summary.csv", index=False, encoding="utf-8-sig")
    make_requested_comparisons(raw)
    write_json(SUMMARY / "status.json", {"status": "complete", "completed_runs": len(raw), "expected_runs": expected})
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
