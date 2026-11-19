from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).parent.parent.parent
EXPERIMENT_ROOT = PROJECT_ROOT / "results"
OUTPUT_ROOT = EXPERIMENT_ROOT / "training_archive"
BASE_CODE = Path(__file__).parent
sys.path.insert(0, str(BASE_CODE))
import run_knowledge_injection_comparison as base  # noqa: E402


PRETRAIN_MAX_EPOCHS = 500
FINETUNE_MAX_EPOCHS = 500
MIN_EPOCHS = 30
PATIENCE = 15
MIN_DELTA = 1e-4
MOVING_WINDOW = 5
VALIDATION_EDGE_FRACTION = 0.10
AREA_CUTOFFS = (5, 10, 20)
QUERY_BLOCK = 512
FRACTIONS = tuple(range(10, 101, 10))
REPEATS = tuple(range(1, 11))
BASE_SEED = 20261111
DEVICE = base.DEVICE

GROUPS = {
    "GCN_anchor_only": {"encoder": "gcn", "graph": "anchor_only", "pretrain": False, "soft": False, "hard": False},
    "GCN_heterogeneous": {"encoder": "gcn", "graph": "heterogeneous", "pretrain": False, "soft": False, "hard": False},
    "GCN-Transformer_direct": {"encoder": "hybrid", "graph": "heterogeneous", "pretrain": False, "soft": False, "hard": False},
    "GCN-Transformer_SSL": {"encoder": "hybrid", "graph": "heterogeneous", "pretrain": True, "soft": False, "hard": False},
    "GCN-Transformer_SSL_soft": {"encoder": "hybrid", "graph": "heterogeneous", "pretrain": True, "soft": True, "hard": False},
    "GCN-Transformer_SSL_hard": {"encoder": "hybrid", "graph": "heterogeneous", "pretrain": True, "soft": False, "hard": True},
    "GCN-Transformer_SSL_soft_hard": {"encoder": "hybrid", "graph": "heterogeneous", "pretrain": True, "soft": True, "hard": True},
}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def clone_state_dict(model) -> dict:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def make_feature_mask(shape: tuple[int, int], seed: int) -> torch.Tensor:
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(seed)
    selected = torch.rand(shape, generator=generator, device=DEVICE).topk(25, dim=1).indices
    mask = torch.zeros(shape, dtype=torch.bool, device=DEVICE)
    mask.scatter_(1, selected, True)
    return mask


def reconstruction_loss(outputs: dict, td: dict, feature_mask: torch.Tensor, soft: bool) -> tuple[torch.Tensor, torch.Tensor]:
    reconstruction = torch.cat([outputs["chemistry"], outputs["geophysics"]], dim=1).float()
    targets = td["features"].float()
    per_node_error = (torch.square(reconstruction - targets) * feature_mask).sum(dim=1)
    per_node_error = per_node_error / feature_mask.sum(dim=1).clamp_min(1)
    masked_nodes = feature_mask.any(dim=1)
    weights = torch.ones_like(per_node_error)
    soft_anchor_loss = torch.zeros((), dtype=torch.float32, device=DEVICE)
    if soft:
        mean_constraint = td["constraint_layers"][: td["n_sites"]].mean(dim=1)
        weights[: td["n_sites"]] = 1.0 + base.SOFT_WEIGHT_STRENGTH * mean_constraint
        weights = weights / weights[masked_nodes].mean().clamp_min(1e-8)
        anchor_masked = masked_nodes & td["anchor_mask"]
        soft_anchor_loss = (weights[anchor_masked] * per_node_error[anchor_masked]).sum() / weights[anchor_masked].sum().clamp_min(1e-8)
    loss = (weights[masked_nodes] * per_node_error[masked_nodes]).sum()
    loss = loss / weights[masked_nodes].sum().clamp_min(1e-8)
    return loss, soft_anchor_loss


def build_pretraining_split(data: dict, td: dict, repeat: int) -> dict:
    forward_ids, edges_np, types_np, weights_np, inverse_np, rel_specs, nodes_by_type, observed = base.setup_pretraining_edges(data, td)
    split_rng = np.random.default_rng(BASE_SEED + 40000 + repeat)
    validation_indices = []
    training_indices = []
    for relation_id in forward_ids:
        relation_indices = np.flatnonzero(types_np == relation_id)
        if len(relation_indices) < 2:
            raise RuntimeError(f"Relation {relation_id} has too few edges for validation")
        shuffled = split_rng.permutation(relation_indices)
        n_validation = min(max(1, int(round(len(relation_indices) * VALIDATION_EDGE_FRACTION))), len(relation_indices) - 1)
        validation_indices.extend(shuffled[:n_validation].tolist())
        training_indices.extend(shuffled[n_validation:].tolist())

    validation_indices = np.asarray(sorted(validation_indices), dtype=np.int64)
    training_indices = np.asarray(sorted(training_indices), dtype=np.int64)
    train_edges, train_types, train_weights = edges_np[:, training_indices], types_np[training_indices], weights_np[training_indices]
    val_edges, val_types, val_weights = edges_np[:, validation_indices], types_np[validation_indices], weights_np[validation_indices]
    inverse = torch.as_tensor(inverse_np, dtype=torch.long, device=DEVICE)
    val_edge_index = torch.as_tensor(val_edges, dtype=torch.long, device=DEVICE)
    val_edge_types = torch.as_tensor(val_types, dtype=torch.long, device=DEVICE)
    val_positive_edges = torch.cat([val_edge_index, val_edge_index.flip(0)], dim=1)
    val_positive_types = torch.cat([val_edge_types, inverse[val_edge_types]])
    val_neg_src, val_neg_dst = base.sample_negative_edges(val_positive_types.cpu().numpy(), split_rng, rel_specs, nodes_by_type, td["n_nodes"], observed)
    val_candidates = torch.as_tensor(np.column_stack([
        np.concatenate([val_positive_edges[0].cpu().numpy(), val_neg_src]),
        np.concatenate([val_positive_edges[1].cpu().numpy(), val_neg_dst]),
    ]).T, dtype=torch.long, device=DEVICE)
    val_labels = torch.cat([
        torch.ones(len(val_positive_types), dtype=torch.float32, device=DEVICE),
        torch.zeros(len(val_neg_src), dtype=torch.float32, device=DEVICE),
    ])
    val_msg_edges = torch.as_tensor(np.concatenate([train_edges, train_edges[::-1]], axis=1), dtype=torch.long, device=DEVICE)
    val_msg_weights = torch.as_tensor(np.concatenate([train_weights, train_weights]), dtype=torch.float32, device=DEVICE)
    return {
        "forward_ids": np.asarray(forward_ids, dtype=np.int64), "train_edges": train_edges,
        "train_types": train_types, "train_weights": train_weights, "inverse": inverse,
        "rel_specs": rel_specs, "nodes_by_type": nodes_by_type, "observed": observed,
        "val_types": val_types, "val_candidates": val_candidates, "val_labels": val_labels,
        "val_msg_edges": val_msg_edges, "val_msg_weights": val_msg_weights,
        "val_feature_mask": make_feature_mask(td["features"].shape, BASE_SEED + 50000 + repeat),
    }


def model_forward(model, td: dict, edge_index, edge_weight, *, feature_mask=None, hard=False):
    return model(
        td["features"], td["node_type"], td["positions"], edge_index, edge_weight,
        feature_mask=feature_mask,
        constraint_layers=td["constraint_layers"], anchor_mask=td["anchor_mask"],
        inject_knowledge=hard,
    )


def validate_pretraining(model, td: dict, split: dict, settings: dict) -> dict:
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
        outputs = model_forward(model, td, split["val_msg_edges"], split["val_msg_weights"], feature_mask=split["val_feature_mask"], hard=settings["hard"])
        edge_logits = model.predict_edges(outputs["embedding"], split["val_candidates"])
        edge_loss = F.binary_cross_entropy_with_logits(edge_logits.float(), split["val_labels"])
        feature_loss, soft_anchor_loss = reconstruction_loss(outputs, td, split["val_feature_mask"], settings["soft"])
        total = edge_loss + feature_loss
    return {
        "validation_total": float(total.float().cpu()),
        "validation_edge": float(edge_loss.float().cpu()),
        "validation_feature": float(feature_loss.float().cpu()),
        "validation_soft_anchor_feature": float(soft_anchor_loss.float().cpu()),
    }


def pretrain_one(data: dict, td: dict, repeat: int, group: str, settings: dict, out_dir: Path) -> dict:
    seed = BASE_SEED + repeat
    base.seed_all(seed)
    split = build_pretraining_split(data, td, repeat)
    model = base.GraphProspectivityModel(td["config"], "hybrid").to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    history = []
    rng = np.random.default_rng(seed)
    best_value, best_epoch, best_state, bad_epochs = math.inf, 0, None, 0

    for epoch in range(1, PRETRAIN_MAX_EPOCHS + 1):
        model.train()
        held = torch.rand(len(split["train_types"]), device=DEVICE) < base.EDGE_HOLDOUT_RATIO
        for relation_id in split["forward_ids"]:
            relation_mask = torch.as_tensor(split["train_types"] == relation_id, device=DEVICE)
            if not torch.any(held & relation_mask):
                held[torch.nonzero(relation_mask, as_tuple=False).flatten()[0]] = True
            if not torch.any((~held) & relation_mask):
                held[torch.nonzero(relation_mask, as_tuple=False).flatten()[-1]] = False
        keep = ~held.detach().cpu().numpy()
        message_edges = split["train_edges"][:, keep]
        message_weights = split["train_weights"][keep]
        msg_edge_index = torch.as_tensor(np.concatenate([message_edges, message_edges[::-1]], axis=1), dtype=torch.long, device=DEVICE)
        msg_edge_weights = torch.as_tensor(np.concatenate([message_weights, message_weights]), dtype=torch.float32, device=DEVICE)
        held_edges = split["train_edges"][:, ~keep]
        held_types = split["train_types"][~keep]
        positive_edges = np.concatenate([held_edges, held_edges[::-1]], axis=1)
        inverse_np = split["inverse"].detach().cpu().numpy()
        positive_types_np = np.concatenate([held_types, inverse_np[held_types]])
        neg_src, neg_dst = base.sample_negative_edges(positive_types_np, rng, split["rel_specs"], split["nodes_by_type"], td["n_nodes"], split["observed"])
        candidates = torch.as_tensor(np.column_stack([
            np.concatenate([positive_edges[0], neg_src]), np.concatenate([positive_edges[1], neg_dst]),
        ]).T, dtype=torch.long, device=DEVICE)
        edge_labels = torch.cat([
            torch.ones(len(positive_types_np), dtype=torch.float32, device=DEVICE),
            torch.zeros(len(neg_src), dtype=torch.float32, device=DEVICE),
        ])
        feature_mask = make_feature_mask(td["features"].shape, seed * 1000 + epoch)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
            edge_outputs = model_forward(model, td, msg_edge_index, msg_edge_weights, hard=settings["hard"])
            edge_loss = F.binary_cross_entropy_with_logits(
                model.predict_edges(edge_outputs["embedding"], candidates).float(), edge_labels
            )
            node_outputs = model_forward(model, td, msg_edge_index, msg_edge_weights, feature_mask=feature_mask, hard=settings["hard"])
            feature_loss, soft_anchor_loss = reconstruction_loss(node_outputs, td, feature_mask, settings["soft"])
            loss = edge_loss + feature_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        validation = validate_pretraining(model, td, split, settings)
        row = {
            "epoch": epoch, "train_total": float(loss.detach().float().cpu()),
            "train_edge": float(edge_loss.detach().float().cpu()),
            "train_feature": float(feature_loss.detach().float().cpu()),
            "train_soft_anchor_feature": float(soft_anchor_loss.detach().float().cpu()),
            **validation,
        }
        history.append(row)
        current = validation["validation_total"]
        if current < best_value - MIN_DELTA:
            best_value, best_epoch, best_state, bad_epochs = current, epoch, clone_state_dict(model), 0
        elif epoch >= MIN_EPOCHS:
            bad_epochs += 1
        if epoch == 1 or epoch % 25 == 0:
            print(f"pretrain {group} repeat={repeat:02d} epoch={epoch:03d}/{PRETRAIN_MAX_EPOCHS} train={row['train_total']:.5f} val={current:.5f} best={best_value:.5f} bad={bad_epochs}/{PATIENCE}", flush=True)
        if epoch >= MIN_EPOCHS and bad_epochs >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError(f"No valid pretraining checkpoint for {group} repeat {repeat}")
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "pretrained.pt"
    torch.save({
        "state_dict": best_state, "model_config": td["config"], "position_min": td["position_min"],
        "position_span": td["position_span"], "seed": seed, "group": group,
        "soft_constraint": settings["soft"], "hard_constraint": settings["hard"],
        "best_epoch": best_epoch, "stopped_epoch": len(history), "best_validation_loss": best_value,
        "early_stopping": {"patience": PATIENCE, "min_delta": MIN_DELTA, "min_epochs": MIN_EPOCHS},
    }, checkpoint)
    pd.DataFrame(history).to_csv(out_dir / "pretraining_history.csv", index=False, encoding="utf-8-sig")
    row = {"group": group, "repeat": repeat, "seed": seed, "best_epoch": best_epoch,
           "stopped_epoch": len(history), "best_validation_loss": best_value, "checkpoint": str(checkpoint)}
    write_json(out_dir / "early_stopping.json", row)
    del model, optimizer
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return row


def prepare_gcn_cache(model) -> None:
    for block in model.blocks:
        if hasattr(block, "gcn"):
            block.gcn.cached = True


def model_parameters(model, settings: dict):
    encoder = (
        list(model.type_specific_embeddings.parameters())
        + list(model.node_type_embedding.parameters())
        + list(model.position_projection.parameters())
        + list(model.blocks.parameters())
        + list(model.output_norm.parameters())
    )
    if settings["encoder"] == "hybrid" and settings["hard"]:
        encoder += list(model.knowledge_projections.parameters())
    classifier = list(model.prospectivity_head.parameters())
    return encoder, classifier


def finetune_one(data: dict, td: dict, group: str, settings: dict, repeat: int, fraction: int,
                 selected_ids: np.ndarray, checkpoint: Path | None, run_dir: Path,
                 checkpoint_output: Path | None = None) -> dict:
    seed = BASE_SEED + 10000 + repeat
    base.seed_all(seed)
    model = base.GraphProspectivityModel(td["config"], settings["encoder"]).to(DEVICE)
    pretrain_meta = {"best_epoch": 0, "stopped_epoch": 0, "best_validation_loss": float("nan")}
    if checkpoint is not None:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["state_dict"], strict=True)
        pretrain_meta = {key: saved[key] for key in ("best_epoch", "stopped_epoch", "best_validation_loss")}
    prepare_gcn_cache(model)
    encoder_parameters, classifier_parameters = model_parameters(model, settings)
    optimizer = torch.optim.AdamW(
        [{"params": encoder_parameters, "lr": 5e-4}, {"params": classifier_parameters, "lr": 5e-2}],
        weight_decay=1e-4,
    )
    positive_ids = torch.as_tensor(selected_ids, dtype=torch.long, device=DEVICE)
    selected_mask = np.zeros(td["n_sites"], dtype=bool)
    selected_mask[selected_ids] = True
    unlabeled_ids = torch.as_tensor(np.flatnonzero(~selected_mask), dtype=torch.long, device=DEVICE)
    history, recent_risks = [], []
    best_monitor, best_epoch, best_state, bad_epochs = math.inf, 0, None, 0

    for epoch in range(1, FINETUNE_MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
            outputs = model_forward(model, td, td["edge_index"], td["edge_weight"], hard=settings["hard"])
            logits = outputs["prospectivity_logit"][:td["n_sites"]].float()
            loss = base.nnpu_risk(logits[positive_ids], logits[unlabeled_ids], 0.02)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder_parameters + classifier_parameters, 5.0)
        optimizer.step()
        risk = float(loss.detach().float().cpu())
        recent_risks.append(risk)
        monitor = float(np.mean(recent_risks[-MOVING_WINDOW:]))
        history.append({"epoch": epoch, "train_nnpu_risk": risk, "monitor_moving_mean": monitor})
        if len(recent_risks) >= MOVING_WINDOW:
            if best_state is None or monitor < best_monitor - MIN_DELTA:
                best_monitor, best_epoch = monitor, epoch
                best_state, bad_epochs = clone_state_dict(model), 0
            elif epoch >= MIN_EPOCHS:
                bad_epochs += 1
        if epoch == 1 or epoch % 25 == 0:
            print(f"finetune {group} {fraction:03d}% repeat={repeat:02d} epoch={epoch:03d}/{FINETUNE_MAX_EPOCHS} risk={risk:.5f} monitor={monitor:.5f} best={best_monitor:.5f} bad={bad_epochs}/{PATIENCE}", flush=True)
        if epoch >= MIN_EPOCHS and bad_epochs >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {group}, {fraction}%, repeat {repeat}")
    model.load_state_dict(best_state)
    model.eval()
    if checkpoint_output is not None:
        checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": clone_state_dict(model),
            "model_config": td["config"],
            "group": group,
            "repeat": repeat,
            "fraction_percent": fraction,
            "seed": seed,
            "best_epoch": int(best_epoch),
            "stopped_epoch": len(history),
            "best_train_risk_moving_mean": float(best_monitor),
            "pretraining_checkpoint": str(checkpoint) if checkpoint is not None else None,
        }, checkpoint_output)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
        logits = model_forward(model, td, td["edge_index"], td["edge_weight"], hard=settings["hard"])["prospectivity_logit"][:td["n_sites"]].float()
        scores = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
    labels = data["positive_mask"]
    x_curve, y_curve, success_auc = success_curve(scores, labels)
    roc_auc = float(base.roc_auc_score(labels, scores))
    captures = {cutoff: float(np.interp(cutoff / 100.0, x_curve, y_curve)) * 100 for cutoff in AREA_CUTOFFS}
    expected = {cutoff: float(np.interp(cutoff / 100.0, x_curve, y_curve)) * int(labels.sum()) for cutoff in AREA_CUTOFFS}

    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(run_dir / "scores_and_labels.npz", scores=scores.astype(np.float32), positive_mask=labels, sampled_positive_ids=selected_ids)
    pd.DataFrame({"cumulative_area_ratio": x_curve, "cumulative_deposit_ratio": y_curve}).to_csv(run_dir / "success_rate_curve.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(run_dir / "finetuning_history.csv", index=False, encoding="utf-8-sig")
    result = {
        "group": group, "graph": settings["graph"], "fraction_percent": fraction,
        "sampled_positive_count": int(len(selected_ids)), "repeat": repeat, "seed": seed,
        "pretrained": checkpoint is not None, "pretraining_best_epoch": pretrain_meta["best_epoch"],
        "pretraining_stopped_epoch": pretrain_meta["stopped_epoch"],
        "pretraining_best_validation_loss": pretrain_meta["best_validation_loss"],
        "finetune_best_epoch": int(best_epoch), "finetune_stopped_epoch": len(history),
        "finetune_best_train_risk_moving_mean": float(best_monitor),
        "roc_auc_resubstitution": roc_auc, "success_rate_curve_auc": success_auc,
        "known_deposit_count": int(labels.sum()),
        "capture_area_5_percent": captures[5], "capture_area_10_percent": captures[10], "capture_area_20_percent": captures[20],
        "expected_deposits_area_5_percent": expected[5], "expected_deposits_area_10_percent": expected[10], "expected_deposits_area_20_percent": expected[20],
        "finetune_epochs": len(history),
    }
    write_json(run_dir / "metrics.json", result)
    write_json(run_dir / "early_stopping.json", {
        "monitor": "five-epoch moving mean of training nnPU risk; not a generalization validation metric",
        "best_epoch": int(best_epoch), "stopped_epoch": len(history), "best_monitor": float(best_monitor),
        "patience": PATIENCE, "min_delta": MIN_DELTA, "min_epochs": MIN_EPOCHS,
    })
    del model, optimizer
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return result


def success_curve(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    if scores.shape != labels.shape or not np.isfinite(scores).all() or labels.sum() == 0:
        raise ValueError("Invalid scores or deposit labels")
    order = np.argsort(-scores, kind="stable")
    ranked_scores, ranked_labels = scores[order], labels[order].astype(np.int64)
    tie_ends = np.flatnonzero(ranked_scores[1:] != ranked_scores[:-1]) + 1
    starts = np.concatenate(([0], tie_ends))
    ends = np.concatenate((tie_ends, [len(scores)]))
    captured = np.cumsum(np.add.reduceat(ranked_labels, starts))
    x = np.concatenate(([0.0], ends / len(scores)))
    y = np.concatenate(([0.0], captured / labels.sum()))
    return x, y, float(np.trapz(y, x))


def build_anchor_td(td: dict) -> dict:
    n = td["n_sites"]
    edge_mask = (td["edge_index"][0] < n) & (td["edge_index"][1] < n)
    anchor_td = dict(td)
    for key in ("features", "node_type", "positions", "constraint_layers"):
        anchor_td[key] = td[key][:n]
    anchor_td["edge_index"] = td["edge_index"][:, edge_mask]
    anchor_td["edge_weight"] = td["edge_weight"][edge_mask]
    anchor_td["anchor_mask"] = torch.ones(n, dtype=torch.bool, device=DEVICE)
    anchor_td["n_nodes"] = n
    return anchor_td


def summarize_results(raw: pd.DataFrame) -> None:
    group_cols = ["group", "graph", "fraction_percent", "sampled_positive_count"]
    metrics = ["roc_auc_resubstitution", "success_rate_curve_auc", "capture_area_5_percent", "capture_area_10_percent", "capture_area_20_percent"]
    summary = raw.groupby(group_cols)[metrics].agg(["mean", "std", "var", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    raw.to_csv(OUTPUT_ROOT / "run_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_ROOT / "comparison_summary.csv", index=False, encoding="utf-8-sig")

    grid = np.linspace(0, 1, 101)
    curve_rows = []
    for group in GROUPS:
        for fraction in FRACTIONS:
            curves = []
            for repeat in REPEATS:
                path = OUTPUT_ROOT / "runs" / group / f"positive_{fraction:03d}" / f"repeat_{repeat:02d}" / "success_rate_curve.csv"
                curve = pd.read_csv(path)
                curves.append(np.interp(grid, curve.cumulative_area_ratio, curve.cumulative_deposit_ratio))
            matrix = np.stack(curves)
            means = matrix.mean(axis=0)
            sds = matrix.std(axis=0, ddof=1)
            curve_rows.extend({"group": group, "fraction_percent": fraction, "cumulative_area_ratio": float(x),
                               "mean_cumulative_deposit_ratio": float(y), "sample_sd_cumulative_deposit_ratio": float(sd)}
                              for x, y, sd in zip(grid, means, sds))
    curve_summary = pd.DataFrame(curve_rows)
    curve_summary.to_csv(OUTPUT_ROOT / "success_rate_curves_mean_sd.csv", index=False, encoding="utf-8-sig")

    colors = dict(zip(GROUPS, ["#5C677D", "#D1495B", "#2A6F97", "#E09F3E", "#6A994E", "#8D5A97", "#17A398"]))
    plt.rcParams.update({"font.family": ["Microsoft YaHei", "DejaVu Sans"], "font.size": 9})
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for group in GROUPS:
        part = summary[summary.group == group].sort_values("fraction_percent")
        x = part.fraction_percent.to_numpy()
        mean = part.success_rate_curve_auc_mean.to_numpy()
        sd = part.success_rate_curve_auc_std.to_numpy()
        ax.plot(x, mean, marker="o", linewidth=1.7, markersize=3.5, color=colors[group], label=group)
        ax.fill_between(x, mean - sd, mean + sd, color=colors[group], alpha=0.09)
    ax.set(xlabel="训练正样本比例 (%)", ylabel="成功率曲线 AUC", title="七组模型的成功率曲线 AUC（均值 ± 10 种子标准差）", xlim=(5, 105), ylim=(0.45, 1.0), xticks=FRACTIONS)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "success_rate_auc_comparison.png", dpi=300)
    fig.savefig(OUTPUT_ROOT / "success_rate_auc_comparison.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
    for ax, cutoff in zip(axes, AREA_CUTOFFS):
        col = f"capture_area_{cutoff}_percent"
        for group in GROUPS:
            part = summary[summary.group == group].sort_values("fraction_percent")
            x = part.fraction_percent.to_numpy()
            mean, sd = part[f"{col}_mean"].to_numpy(), part[f"{col}_std"].to_numpy()
            ax.plot(x, mean, marker="o", linewidth=1.5, markersize=3.2, color=colors[group], label=group)
            ax.fill_between(x, np.clip(mean - sd, 0, 100), np.clip(mean + sd, 0, 100), color=colors[group], alpha=0.08)
        ax.set_title(f"面积前 {cutoff}%")
        ax.set_xlabel("训练正样本比例 (%)")
        ax.set_xticks(FRACTIONS)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("已知矿床捕获率 (%)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.1), fontsize=8)
    fig.suptitle("固定勘查面积捕获率（均值 ± 10 种子标准差）")
    fig.tight_layout(rect=(0, 0.14, 1, 0.94))
    fig.savefig(OUTPUT_ROOT / "fixed_area_capture_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_ROOT / "fixed_area_capture_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(18, 7.5), sharex=True, sharey=True)
    for ax, fraction in zip(axes.flat, FRACTIONS):
        part = curve_summary[curve_summary.fraction_percent == fraction]
        for group in GROUPS:
            g = part[part.group == group]
            x = g.cumulative_area_ratio.to_numpy()
            y = g.mean_cumulative_deposit_ratio.to_numpy()
            sd = g.sample_sd_cumulative_deposit_ratio.to_numpy()
            ax.plot(x, y, color=colors[group], linewidth=1.3, label=group if fraction == FRACTIONS[0] else None)
            ax.fill_between(x, np.clip(y - sd, 0, 1), np.clip(y + sd, 0, 1), color=colors[group], alpha=0.07)
        auc = summary[(summary.fraction_percent == fraction)].set_index("group").success_rate_curve_auc_mean
        ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=0.7)
        ax.set_title(f"{fraction}% 正例\nAUC 最高 {auc.max():.3f}")
        ax.set_xlabel("累计面积比")
        ax.grid(alpha=0.18)
    axes[0, 0].set_ylabel("累计矿床比")
    axes[1, 0].set_ylabel("累计矿床比")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04), fontsize=8)
    fig.suptitle("七组模型成功率曲线（均值 ± 10 种子标准差）")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(OUTPUT_ROOT / "success_rate_curves_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_ROOT / "success_rate_curves_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full experiment")
    if DEVICE.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    data = base.load_data()
    td_full = base.tensor_data(data)
    td_full["config"] = dict(td_full["config"])
    td_full["config"]["encoder_layers"] = 2
    td_full["config"]["attention_query_block"] = QUERY_BLOCK
    td_anchor = build_anchor_td(td_full)
    graph_tds = {"heterogeneous": td_full, "anchor_only": td_anchor}
    n_positive = len(data["positive_ids"])
    positive_counts = {pct: max(1, min(n_positive, int(math.floor(n_positive * pct / 100 + 0.5)))) for pct in FRACTIONS}
    if args.dry_run:
        model = base.GraphProspectivityModel(td_full["config"], "hybrid").to(DEVICE)
        print(json.dumps({
            "device": torch.cuda.get_device_name(0), "model_config": td_full["config"],
            "groups": GROUPS, "fractions": list(FRACTIONS), "positive_counts": positive_counts,
            "seeds": [BASE_SEED + repeat for repeat in REPEATS], "fine_tuning_runs": len(GROUPS) * len(FRACTIONS) * len(REPEATS),
            "pretraining_runs": sum(item["pretrain"] for item in GROUPS.values()) * len(REPEATS),
            "max_pretraining_epochs": PRETRAIN_MAX_EPOCHS, "max_finetuning_epochs": FINETUNE_MAX_EPOCHS,
            "early_stopping": {"monitor_window": MOVING_WINDOW, "minimum_epochs": MIN_EPOCHS, "patience": PATIENCE, "min_delta": MIN_DELTA},
            "anchor_only_nodes": td_anchor["n_nodes"], "heterogeneous_nodes": td_full["n_nodes"],
            "output_root": str(OUTPUT_ROOT),
        }, ensure_ascii=False, indent=2))
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    model_config = td_full["config"]
    group_edges = {"anchor_only": int(td_anchor["edge_index"].shape[1]), "heterogeneous": int(td_full["edge_index"].shape[1])}
    graph_hash = base.sha256(base.GRAPH_SOURCE)
    manifest = {
        "experiment": "Seven-way two-layer GCN and GCN-Transformer comparison with early stopping",
        "source_main_experiment": ".", "graph_file": base.GRAPH_SOURCE.as_posix(), "graph_sha256": graph_hash,
        "device": torch.cuda.get_device_name(0), "model_config": model_config, "groups": GROUPS,
        "graph_edge_counts": group_edges,
        "pretraining": {"maximum_epochs": PRETRAIN_MAX_EPOCHS, "groups": [g for g, s in GROUPS.items() if s["pretrain"]], "reuse": "one checkpoint per group and seed reused at all ten positive fractions"},
        "finetuning_maximum_epochs": FINETUNE_MAX_EPOCHS,
        "early_stopping": {
            "pretraining_monitor": "sum of fixed held-out relation reconstruction cross entropy and feature reconstruction loss; held-out edges excluded from validation message passing",
            "finetuning_monitor": "five-epoch moving mean of training nnPU risk; not a generalization validation metric",
            "patience": PATIENCE, "min_delta": MIN_DELTA, "minimum_epochs": MIN_EPOCHS,
        },
        "fractions_percent": list(FRACTIONS), "sampled_positive_counts": positive_counts,
        "seeds": [BASE_SEED + repeat for repeat in REPEATS], "repeats": len(REPEATS),
        "sampling": "Same permutation and nested positive subsets are shared across all seven methods for each seed.",
        "metrics": ["resubstitution ROC-AUC", "tie-aware success-rate curve AUC", "known-deposit capture at 5%, 10%, and 20% of study area"],
        "evaluation_scope": "All 36 known deposits are evaluated in every run, including those sampled for training; in-sample ranking/capture, not independent spatial validation.",
        "status": "running",
    }
    write_json(OUTPUT_ROOT / "experiment_manifest.json", manifest)

    pretrain_groups = {g: s for g, s in GROUPS.items() if s["pretrain"]}
    pretraining_rows = []
    checkpoints = {}
    for repeat in REPEATS:
        for group, settings in pretrain_groups.items():
            folder = OUTPUT_ROOT / "pretraining" / group / f"repeat_{repeat:02d}"
            checkpoint = folder / "pretrained.pt"
            if checkpoint.is_file():
                saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                if saved.get("model_config") != model_config or saved.get("group") != group or saved.get("seed") != BASE_SEED + repeat:
                    raise ValueError(f"Pretraining checkpoint metadata mismatch: {checkpoint}")
                row = {"group": group, "repeat": repeat, "seed": saved["seed"], "best_epoch": saved["best_epoch"],
                       "stopped_epoch": saved["stopped_epoch"], "best_validation_loss": saved["best_validation_loss"], "checkpoint": str(checkpoint)}
            else:
                row = pretrain_one(data, td_full, repeat, group, settings, folder)
            checkpoints[(group, repeat)] = checkpoint
            pretraining_rows.append(row)
            pd.DataFrame(pretraining_rows).to_csv(OUTPUT_ROOT / "pretraining_summary.csv", index=False, encoding="utf-8-sig")
            write_json(OUTPUT_ROOT / "status.json", {
                "status": "running", "completed_pretraining_runs": len(checkpoints),
                "expected_pretraining_runs": len(pretrain_groups) * len(REPEATS),
                "completed_finetuning_runs": 0, "expected_finetuning_runs": len(GROUPS) * len(FRACTIONS) * len(REPEATS),
            })

    metrics_path = OUTPUT_ROOT / "run_metrics.csv"
    existing = pd.read_csv(metrics_path).to_dict("records") if metrics_path.is_file() else []
    completed = {(row["group"], int(row["fraction_percent"]), int(row["repeat"])) for row in existing}
    results = list(existing)
    permutations = {repeat: np.random.default_rng(BASE_SEED + repeat).permutation(data["positive_ids"]) for repeat in REPEATS}
    expected_runs = len(GROUPS) * len(FRACTIONS) * len(REPEATS)
    write_json(OUTPUT_ROOT / "status.json", {
        "status": "running", "completed_pretraining_runs": len(checkpoints),
        "expected_pretraining_runs": len(pretrain_groups) * len(REPEATS),
        "completed_finetuning_runs": len(completed), "expected_finetuning_runs": expected_runs,
    })
    for group, settings in GROUPS.items():
        td = graph_tds[settings["graph"]]
        for fraction in FRACTIONS:
            count = positive_counts[fraction]
            for repeat in REPEATS:
                key = (group, fraction, repeat)
                if key in completed:
                    continue
                selected_ids = permutations[repeat][:count]
                checkpoint = checkpoints.get((group, repeat))
                run_dir = OUTPUT_ROOT / "runs" / group / f"positive_{fraction:03d}" / f"repeat_{repeat:02d}"
                started = time.time()
                checkpoint_output = None
                if group == "GCN-Transformer_SSL_soft_hard" and fraction == 100:
                    checkpoint_output = OUTPUT_ROOT / "checkpoints" / group / f"positive_{fraction:03d}" / f"repeat_{repeat:02d}" / "finetuned.pt"
                result = finetune_one(
                    data, td, group, settings, repeat, fraction, selected_ids,
                    checkpoint, run_dir, checkpoint_output=checkpoint_output,
                )
                result["elapsed_seconds"] = time.time() - started
                results.append(result)
                completed.add(key)
                pd.DataFrame([result]).to_csv(metrics_path, mode="a", header=not metrics_path.exists(), index=False,
                                               encoding="utf-8-sig" if not metrics_path.exists() else "utf-8")
                write_json(OUTPUT_ROOT / "status.json", {
                    "status": "running", "completed_pretraining_runs": len(checkpoints),
                    "expected_pretraining_runs": len(pretrain_groups) * len(REPEATS),
                    "completed_finetuning_runs": len(completed), "expected_finetuning_runs": expected_runs,
                })

    raw = pd.DataFrame(results)
    if len(raw) != expected_runs or raw.duplicated(["group", "fraction_percent", "repeat"]).any():
        raise RuntimeError(f"Expected {expected_runs} unique fine-tuning runs, found {len(raw)}")
    for group in GROUPS:
        rows = raw[raw.group == group]
        if len(rows) != len(FRACTIONS) * len(REPEATS) or (rows.groupby("fraction_percent").size() != len(REPEATS)).any():
            raise RuntimeError(f"Incomplete fraction/seed pairs for {group}")
    summarize_results(raw)
    manifest["status"] = "complete"
    write_json(OUTPUT_ROOT / "experiment_manifest.json", manifest)
    write_json(OUTPUT_ROOT / "status.json", {
        "status": "complete", "completed_pretraining_runs": len(checkpoints),
        "expected_pretraining_runs": len(pretrain_groups) * len(REPEATS),
        "completed_finetuning_runs": len(raw), "expected_finetuning_runs": expected_runs,
    })
    print(pd.read_csv(OUTPUT_ROOT / "comparison_summary.csv").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
