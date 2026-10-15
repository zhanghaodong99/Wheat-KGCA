from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.convnextv2_tiny import build_model  # noqa: E402


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "src_model_training"
    / "src_single_convnextv2_tiny_20260713_092041"
    / "convnextv2_tiny"
    / "best_model.pth"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "paper_full_experiments" / "image_ablation_04_expert_prototype"

EXPERT_FEATURE_NAMES = [
    "leaf_emergence",
    "tiller_density",
    "stem_elongation",
    "flag_leaf_visibility",
    "boot_swelling",
    "spike_emergence",
    "spike_visibility",
    "anther_visibility",
    "flowering_intensity",
    "grain_development",
    "grain_fullness",
    "plant_green_degree",
    "leaf_senescence",
    "spike_yellowing",
    "dry_down_degree",
    "background_noise_tolerance",
]

EXPERT_STAGE_PROTOTYPES = {
    "seedling": [1.00, 0.15, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.95, 0.05, 0.00, 0.00, 0.25],
    "tillering": [0.75, 1.00, 0.20, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.95, 0.05, 0.00, 0.00, 0.30],
    "stem_elongation": [0.55, 0.70, 1.00, 0.30, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.90, 0.08, 0.00, 0.00, 0.35],
    "flag_leaf_booting": [0.35, 0.45, 0.85, 1.00, 0.95, 0.15, 0.10, 0.00, 0.00, 0.00, 0.00, 0.85, 0.10, 0.00, 0.00, 0.35],
    "heading": [0.20, 0.30, 0.75, 0.75, 0.40, 1.00, 0.95, 0.15, 0.05, 0.00, 0.00, 0.80, 0.12, 0.05, 0.00, 0.40],
    "flowering": [0.15, 0.25, 0.65, 0.65, 0.20, 0.95, 1.00, 1.00, 1.00, 0.10, 0.05, 0.75, 0.15, 0.08, 0.00, 0.45],
    "grain_filling": [0.08, 0.15, 0.50, 0.40, 0.05, 0.80, 0.90, 0.20, 0.10, 1.00, 0.85, 0.55, 0.35, 0.35, 0.20, 0.50],
    "maturity": [0.02, 0.05, 0.30, 0.15, 0.00, 0.60, 0.80, 0.02, 0.00, 0.75, 1.00, 0.10, 0.95, 1.00, 1.00, 0.55],
}


def image_key(path_value: str) -> str:
    return PureWindowsPath(str(path_value)).name.lower()


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def build_transforms(input_size: int) -> tuple[Any, Any]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.15, 0.15, 0.10, 0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.10), ratio=(0.4, 2.5)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def balanced_limit_manifest(manifest: pd.DataFrame, limit: int) -> pd.DataFrame:
    if limit <= 0 or len(manifest) <= limit:
        return manifest.copy()
    class_names = sorted(manifest["stage_id"].astype(str).unique().tolist())
    per_class = max(1, int(np.ceil(limit / max(len(class_names), 1))))
    pieces = []
    for class_name in class_names:
        group = manifest[manifest["stage_id"].astype(str) == class_name]
        pieces.append(group.head(per_class))
    return pd.concat(pieces, axis=0).sort_index().head(limit).copy()


def build_stage_prototypes(class_names: list[str], mode: str, seed: int) -> tuple[np.ndarray, list[str]]:
    rows = []
    for class_name in class_names:
        if class_name not in EXPERT_STAGE_PROTOTYPES:
            raise ValueError(f"Missing expert prototype for class: {class_name}")
        rows.append(EXPERT_STAGE_PROTOTYPES[class_name])
    prototypes = np.asarray(rows, dtype=np.float32)
    rng = np.random.default_rng(seed)
    if mode == "expert":
        pass
    elif mode == "shuffled":
        prototypes = prototypes[rng.permutation(len(class_names))]
    elif mode == "random":
        prototypes = rng.uniform(0.0, 1.0, size=prototypes.shape).astype(np.float32)
    else:
        raise ValueError(f"Unknown prototype_mode={mode}")
    return prototypes, EXPERT_FEATURE_NAMES


class ManifestImageDataset(Dataset):
    def __init__(self, manifest_path: Path, class_to_idx: dict[str, int], transform: Any, limit: int):
        frame = pd.read_csv(manifest_path)
        frame = balanced_limit_manifest(frame, limit)
        missing_labels = sorted(set(frame["stage_id"].astype(str)) - set(class_to_idx))
        if missing_labels:
            raise ValueError(f"Manifest contains labels missing from checkpoint class_names: {missing_labels}")
        if "image_key" not in frame.columns:
            frame["image_key"] = frame["image_path"].map(image_key)
        frame["label_idx"] = frame["stage_id"].astype(str).map(class_to_idx).astype(int)
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(Path(str(row["image_path"]))) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, int(row["label_idx"]), index


class PrototypeWheatSAGE(nn.Module):
    def __init__(
        self,
        visual_model: nn.Module,
        prototypes: np.ndarray,
        hidden_dim: int,
        max_confidence: float,
        max_margin: float,
        eligibility_sharpness: float,
        gate_init_bias: float,
        min_effective_gate: float,
        semantic_logit_scale: float,
        delta_scale: float,
        disable_prototype: bool,
        disable_gate: bool,
    ) -> None:
        super().__init__()
        self.visual = visual_model
        self.max_confidence = max_confidence
        self.max_margin = max_margin
        self.eligibility_sharpness = eligibility_sharpness
        self.min_effective_gate = min_effective_gate
        self.semantic_logit_scale = semantic_logit_scale
        self.delta_scale = delta_scale
        self.disable_prototype = disable_prototype
        self.disable_gate = disable_gate
        self.register_buffer("prototype_bank", torch.as_tensor(prototypes, dtype=torch.float32))

        feature_dim = int(self.visual.feature_dim)
        prototype_dim = int(prototypes.shape[1])
        num_classes = int(prototypes.shape[0])
        self.visual_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.prototype_encoder = nn.Sequential(
            nn.Linear(prototype_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.visual_match = nn.Linear(hidden_dim, hidden_dim)
        self.prototype_match = nn.Linear(hidden_dim, hidden_dim)
        self.film = nn.Linear(hidden_dim, hidden_dim * 2)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, num_classes),
            nn.Tanh(),
        )
        self.logit_temperature = nn.Parameter(torch.tensor(1.5))
        nn.init.constant_(self.gate[-1].bias, gate_init_bias)
        nn.init.zeros_(self.delta_head[-2].weight)
        nn.init.zeros_(self.delta_head[-2].bias)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        visual_logits, raw_feat = self.visual(images)
        visual_feature = self.visual._pool_features(raw_feat)
        visual_token = self.visual_proj(visual_feature)
        prototype_tokens = self.prototype_encoder(self.prototype_bank)
        if self.disable_prototype:
            prototype_tokens = torch.zeros_like(prototype_tokens)

        visual_match = F.normalize(self.visual_match(visual_token), dim=1)
        prototype_match = F.normalize(self.prototype_match(prototype_tokens), dim=1)
        temperature = torch.exp(self.logit_temperature.clamp(-2.0, 3.0))
        semantic_logits = temperature * visual_match @ prototype_match.t()
        prototype_attention = F.softmax(semantic_logits, dim=1)
        semantic_context = prototype_attention @ prototype_tokens

        gamma, beta = self.film(semantic_context).chunk(2, dim=1)
        guided_visual = visual_token * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * torch.tanh(beta)

        visual_probs = F.softmax(visual_logits.detach(), dim=1)
        top2 = torch.topk(visual_probs, k=2, dim=1).values
        max_conf = top2[:, 0:1]
        margin = top2[:, 0:1] - top2[:, 1:2]
        entropy = -(visual_probs.clamp_min(1e-8) * visual_probs.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
        uncertainty = torch.cat([max_conf, margin, entropy], dim=1)

        gate_input = torch.cat([visual_token, semantic_context, uncertainty], dim=1)
        raw_gate = torch.sigmoid(self.gate(gate_input))
        low_conf = torch.sigmoid((self.max_confidence - max_conf) * self.eligibility_sharpness)
        low_margin = torch.sigmoid((self.max_margin - margin) * self.eligibility_sharpness)
        eligibility = torch.maximum(low_conf, low_margin)
        effective_gate = raw_gate * eligibility
        if self.min_effective_gate > 0:
            effective_gate = torch.maximum(effective_gate, self.min_effective_gate * eligibility)
        if self.disable_gate:
            effective_gate = torch.ones_like(effective_gate)

        residual_input = torch.cat([guided_visual, semantic_context, uncertainty], dim=1)
        learned_delta = self.delta_scale * self.delta_head(residual_input)
        prototype_delta = self.semantic_logit_scale * semantic_logits
        delta_logits = prototype_delta + learned_delta
        fused_logits = visual_logits + effective_gate * delta_logits
        return {
            "fused_logits": fused_logits,
            "visual_logits": visual_logits,
            "visual_probs": visual_probs,
            "semantic_logits": semantic_logits,
            "prototype_attention": prototype_attention,
            "raw_gate": raw_gate,
            "effective_gate": effective_gate,
            "eligibility": eligibility,
            "delta_logits": delta_logits,
        }


def set_visual_trainability(model: PrototypeWheatSAGE, freeze_visual: str) -> None:
    for param in model.visual.parameters():
        param.requires_grad = True
    if freeze_visual == "all":
        for param in model.visual.parameters():
            param.requires_grad = False
    elif freeze_visual == "encoder":
        for param in model.visual.encoder.parameters():
            param.requires_grad = False
        for param in model.visual.head.parameters():
            param.requires_grad = True
    elif freeze_visual == "none":
        return
    else:
        raise ValueError(f"Unknown freeze_visual={freeze_visual}")


def high_confidence_mask(probs: torch.Tensor, confidence_threshold: float, margin_threshold: float) -> torch.Tensor:
    top2 = torch.topk(probs, k=2, dim=1).values
    max_conf = top2[:, 0:1]
    margin = top2[:, 0:1] - top2[:, 1:2]
    return ((max_conf >= confidence_threshold) & (margin >= margin_threshold)).float()


def weighted_ce(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    losses = F.cross_entropy(logits, labels, reduction="none")
    weights = weights.squeeze(1)
    return (losses * weights).sum() / weights.sum().clamp_min(1e-6)


def preserve_margin_loss(
    fused_logits: torch.Tensor,
    visual_logits: torch.Tensor,
    high_conf: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    visual_pred = visual_logits.detach().argmax(dim=1)
    visual_score = fused_logits.gather(1, visual_pred[:, None])
    mask = torch.ones_like(fused_logits, dtype=torch.bool)
    mask.scatter_(1, visual_pred[:, None], False)
    competitor_score = fused_logits.masked_fill(~mask, -1e9).max(dim=1, keepdim=True).values
    losses = F.relu(margin - (visual_score - competitor_score))
    return (losses * high_conf).sum() / high_conf.sum().clamp_min(1.0)


def compute_loss(outputs: dict[str, torch.Tensor], labels: torch.Tensor, args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float]]:
    visual_probs = outputs["visual_probs"]
    high_conf = high_confidence_mask(visual_probs, args.high_confidence_threshold, args.high_margin_threshold)
    sample_weights = 1.0 + args.low_conf_focus_weight * outputs["eligibility"].detach()
    fused_ce = weighted_ce(outputs["fused_logits"], labels, sample_weights)
    visual_aux = F.cross_entropy(outputs["visual_logits"], labels)
    semantic_ce = F.cross_entropy(outputs["semantic_logits"], labels)
    gate_loss = (outputs["raw_gate"] * high_conf).sum() / high_conf.sum().clamp_min(1.0)
    gate_open_target = args.gate_open_target * outputs["eligibility"].detach()
    gate_open_loss = F.relu(gate_open_target - outputs["effective_gate"]).pow(2).sum() / (
        outputs["eligibility"].detach().sum().clamp_min(1.0)
    )
    anchor_loss = (
        F.kl_div(F.log_softmax(outputs["fused_logits"], dim=1), visual_probs, reduction="none").sum(dim=1, keepdim=True)
        * high_conf
    ).sum() / high_conf.sum().clamp_min(1.0)
    switch_loss = preserve_margin_loss(outputs["fused_logits"], outputs["visual_logits"], high_conf, args.preserve_margin)
    delta_loss = (
        outputs["delta_logits"].pow(2).mean(dim=1, keepdim=True) * outputs["effective_gate"] * high_conf
    ).sum() / high_conf.sum().clamp_min(1.0)
    loss = (
        fused_ce
        + args.aux_weight * visual_aux
        + args.semantic_ce_weight * semantic_ce
        + args.gate_weight * gate_loss
        + args.gate_open_weight * gate_open_loss
        + args.anchor_weight * anchor_loss
        + args.switch_weight * switch_loss
        + args.delta_weight * delta_loss
    )
    stats = {
        "loss": float(loss.detach().item()),
        "fused_ce": float(fused_ce.detach().item()),
        "visual_aux": float(visual_aux.detach().item()),
        "semantic_ce": float(semantic_ce.detach().item()),
        "gate": float(gate_loss.detach().item()),
        "gate_open": float(gate_open_loss.detach().item()),
        "anchor": float(anchor_loss.detach().item()),
        "switch": float(switch_loss.detach().item()),
        "delta": float(delta_loss.detach().item()),
        "raw_gate_mean": float(outputs["raw_gate"].detach().mean().item()),
        "effective_gate_mean": float(outputs["effective_gate"].detach().mean().item()),
        "eligibility_mean": float(outputs["eligibility"].detach().mean().item()),
    }
    return loss, stats


def make_grad_scaler(enabled: bool, device: torch.device) -> Any:
    if not enabled or device.type != "cuda":
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda")
        except TypeError:
            return torch.amp.GradScaler()
    return torch.cuda.amp.GradScaler()


def autocast_context(device: torch.device) -> Any:
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type)
    if device.type == "cuda":
        return torch.cuda.amp.autocast()
    return nullcontext()


def train_one_epoch(
    model: PrototypeWheatSAGE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.train()
    totals: dict[str, list[float]] = {}
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with autocast_context(device):
                outputs = model(images)
                loss, stats = compute_loss(outputs, labels, args)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss, stats = compute_loss(outputs, labels, args)
            loss.backward()
            optimizer.step()
        for key, value in stats.items():
            totals.setdefault(key, []).append(value)
    return {key: float(np.mean(values)) for key, values in totals.items()}


def balanced_accuracy_all_classes(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> float:
    recalls: list[float] = []
    for label in labels:
        mask = y_true == label
        if mask.sum() == 0:
            continue
        recalls.append(float((y_pred[mask] == label).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


@torch.no_grad()
def evaluate(
    model: PrototypeWheatSAGE,
    loader: DataLoader,
    frame: pd.DataFrame,
    class_names: list[str],
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model.eval()
    labels_all: list[int] = []
    visual_preds: list[int] = []
    fused_preds: list[int] = []
    semantic_preds: list[int] = []
    visual_conf: list[float] = []
    fused_conf: list[float] = []
    semantic_conf: list[float] = []
    raw_gates: list[float] = []
    effective_gates: list[float] = []
    eligibilities: list[float] = []
    row_indices: list[int] = []
    for images, labels, indices in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        visual_prob = F.softmax(outputs["visual_logits"], dim=1)
        fused_prob = F.softmax(outputs["fused_logits"], dim=1)
        semantic_prob = F.softmax(outputs["semantic_logits"], dim=1)
        labels_all.extend(labels.numpy().astype(int).tolist())
        visual_preds.extend(visual_prob.argmax(dim=1).cpu().numpy().astype(int).tolist())
        fused_preds.extend(fused_prob.argmax(dim=1).cpu().numpy().astype(int).tolist())
        semantic_preds.extend(semantic_prob.argmax(dim=1).cpu().numpy().astype(int).tolist())
        visual_conf.extend(visual_prob.max(dim=1).values.cpu().numpy().astype(float).tolist())
        fused_conf.extend(fused_prob.max(dim=1).values.cpu().numpy().astype(float).tolist())
        semantic_conf.extend(semantic_prob.max(dim=1).values.cpu().numpy().astype(float).tolist())
        raw_gates.extend(outputs["raw_gate"].cpu().numpy().reshape(-1).astype(float).tolist())
        effective_gates.extend(outputs["effective_gate"].cpu().numpy().reshape(-1).astype(float).tolist())
        eligibilities.extend(outputs["eligibility"].cpu().numpy().reshape(-1).astype(float).tolist())
        row_indices.extend(indices.numpy().astype(int).tolist())

    labels_np = np.array(labels_all, dtype=int)
    visual_np = np.array(visual_preds, dtype=int)
    fused_np = np.array(fused_preds, dtype=int)
    semantic_np = np.array(semantic_preds, dtype=int)
    visual_correct = visual_np == labels_np
    fused_correct = fused_np == labels_np
    switched = visual_np != fused_np
    metric_labels = list(range(len(class_names)))
    metrics = {
        "visual_accuracy": float(accuracy_score(labels_np, visual_np)),
        "visual_macro_f1": float(f1_score(labels_np, visual_np, labels=metric_labels, average="macro", zero_division=0)),
        "visual_balanced_accuracy": balanced_accuracy_all_classes(labels_np, visual_np, metric_labels),
        "semantic_accuracy": float(accuracy_score(labels_np, semantic_np)),
        "semantic_macro_f1": float(f1_score(labels_np, semantic_np, labels=metric_labels, average="macro", zero_division=0)),
        "fused_accuracy": float(accuracy_score(labels_np, fused_np)),
        "fused_macro_f1": float(f1_score(labels_np, fused_np, labels=metric_labels, average="macro", zero_division=0)),
        "fused_balanced_accuracy": balanced_accuracy_all_classes(labels_np, fused_np, metric_labels),
        "raw_gate_mean": float(np.mean(raw_gates)),
        "effective_gate_mean": float(np.mean(effective_gates)),
        "eligibility_mean": float(np.mean(eligibilities)),
        "switch_count": int(switched.sum()),
        "corrected_count": int((switched & ~visual_correct & fused_correct).sum()),
        "introduced_error_count": int((switched & visual_correct & ~fused_correct).sum()),
        "wrong_to_wrong_switch_count": int((switched & ~visual_correct & ~fused_correct).sum()),
        "net_correct_change_vs_visual_branch": int(fused_correct.sum() - visual_correct.sum()),
    }

    pred_frame = frame.iloc[row_indices][["image_id", "image_key", "split", "stage_id", "source_type", "image_path"]].copy()
    pred_frame["true_label"] = [class_names[i] for i in labels_np]
    pred_frame["visual_pred_label"] = [class_names[i] for i in visual_np]
    pred_frame["semantic_pred_label"] = [class_names[i] for i in semantic_np]
    pred_frame["fused_pred_label"] = [class_names[i] for i in fused_np]
    pred_frame["visual_confidence"] = visual_conf
    pred_frame["semantic_confidence"] = semantic_conf
    pred_frame["fused_confidence"] = fused_conf
    pred_frame["raw_gate"] = raw_gates
    pred_frame["effective_gate"] = effective_gates
    pred_frame["eligibility"] = eligibilities
    pred_frame["visual_correct"] = visual_correct
    pred_frame["fused_correct"] = fused_correct
    pred_frame["switched"] = switched
    return metrics, pred_frame


def make_optimizer(model: PrototypeWheatSAGE, args: argparse.Namespace) -> torch.optim.Optimizer:
    visual_head_params = [p for p in model.visual.head.parameters() if p.requires_grad]
    encoder_params = [p for p in model.visual.encoder.parameters() if p.requires_grad]
    fusion_params = [
        p
        for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("visual.head.") and not name.startswith("visual.encoder.")
    ]
    groups = []
    if encoder_params:
        groups.append({"params": encoder_params, "lr": args.lr * args.encoder_lr_scale})
    if visual_head_params:
        groups.append({"params": visual_head_params, "lr": args.lr * args.visual_head_lr_scale})
    groups.append({"params": fusion_params, "lr": args.lr})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def selection_key(metrics: dict[str, Any], mode: str) -> tuple[Any, ...]:
    if mode == "conservative":
        return (
            metrics["fused_macro_f1"],
            metrics["fused_accuracy"],
            metrics["net_correct_change_vs_visual_branch"],
            -metrics["introduced_error_count"],
            -metrics["switch_count"],
        )
    if mode == "fused":
        return (
            metrics["fused_accuracy"],
            metrics["fused_macro_f1"],
            metrics["net_correct_change_vs_visual_branch"],
            -metrics["introduced_error_count"],
        )
    if mode == "switch_gain":
        return (
            metrics["net_correct_change_vs_visual_branch"],
            metrics["corrected_count"],
            -metrics["introduced_error_count"],
            metrics["fused_accuracy"],
            metrics["fused_macro_f1"],
            -metrics["switch_count"],
        )
    raise ValueError(f"Unknown selection_mode={mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train no-leak expert prototype Wheat-SAGE on full images.")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prototype-mode", choices=["expert", "shuffled", "random"], default="expert")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--visual-head-lr-scale", type=float, default=0.3)
    parser.add_argument("--encoder-lr-scale", type=float, default=0.05)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--freeze-visual", choices=["all", "encoder", "none"], default="encoder")
    parser.add_argument("--aux-weight", type=float, default=0.15)
    parser.add_argument("--semantic-ce-weight", type=float, default=0.20)
    parser.add_argument("--gate-weight", type=float, default=0.0)
    parser.add_argument("--gate-open-weight", type=float, default=0.05)
    parser.add_argument("--gate-open-target", type=float, default=0.15)
    parser.add_argument("--anchor-weight", type=float, default=0.05)
    parser.add_argument("--switch-weight", type=float, default=0.02)
    parser.add_argument("--delta-weight", type=float, default=0.0)
    parser.add_argument("--low-conf-focus-weight", type=float, default=2.0)
    parser.add_argument("--max-confidence", type=float, default=0.97)
    parser.add_argument("--max-margin", type=float, default=0.75)
    parser.add_argument("--high-confidence-threshold", type=float, default=0.985)
    parser.add_argument("--high-margin-threshold", type=float, default=0.80)
    parser.add_argument("--preserve-margin", type=float, default=0.03)
    parser.add_argument("--eligibility-sharpness", type=float, default=10.0)
    parser.add_argument("--semantic-logit-scale", type=float, default=0.35)
    parser.add_argument("--delta-scale", type=float, default=2.0)
    parser.add_argument("--gate-init-bias", type=float, default=0.0)
    parser.add_argument("--min-effective-gate", type=float, default=0.05)
    parser.add_argument("--disable-prototype", action="store_true")
    parser.add_argument("--disable-gate", action="store_true")
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    parser.add_argument(
        "--selection-mode",
        choices=["conservative", "fused", "switch_gain"],
        default="switch_gain",
    )
    return parser.parse_args()


def json_safe_args(args: argparse.Namespace) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in vars(args).items():
        safe[key] = str(value) if isinstance(value, Path) else value
    return safe


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(args.checkpoint, device)
    class_names = checkpoint.get("class_names")
    if not class_names:
        raise RuntimeError(f"Checkpoint does not contain class_names: {args.checkpoint}")
    class_names = [str(name) for name in class_names]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    input_size = int(checkpoint.get("input_size", 224))
    train_transform, eval_transform = build_transforms(input_size)
    prototypes, prototype_feature_names = build_stage_prototypes(class_names, args.prototype_mode, args.seed)

    train_dataset = ManifestImageDataset(args.train_manifest, class_to_idx, train_transform, args.limit_train)
    val_dataset = ManifestImageDataset(args.val_manifest, class_to_idx, eval_transform, args.limit_val)
    test_dataset = ManifestImageDataset(args.test_manifest, class_to_idx, eval_transform, args.limit_test)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    visual_model = build_model(num_classes=len(class_names), pretrained=False)
    visual_model.load_state_dict(checkpoint["model_state_dict"])
    model = PrototypeWheatSAGE(
        visual_model=visual_model,
        prototypes=prototypes,
        hidden_dim=args.hidden_dim,
        max_confidence=args.max_confidence,
        max_margin=args.max_margin,
        eligibility_sharpness=args.eligibility_sharpness,
        gate_init_bias=args.gate_init_bias,
        min_effective_gate=args.min_effective_gate,
        semantic_logit_scale=args.semantic_logit_scale,
        delta_scale=args.delta_scale,
        disable_prototype=args.disable_prototype,
        disable_gate=args.disable_gate,
    ).to(device)
    set_visual_trainability(model, args.freeze_visual)
    optimizer = make_optimizer(model, args)
    scaler = make_grad_scaler(args.amp, device)

    config = {
        "method": "prototype_wheat_sage_no_label_leak",
        "label_leakage_guard": "Dataset returns image and label only; every sample receives the same all-class prototype bank.",
        "train_manifest": str(args.train_manifest),
        "val_manifest": str(args.val_manifest),
        "test_manifest": str(args.test_manifest),
        "checkpoint": str(args.checkpoint),
        "class_names": class_names,
        "prototype_mode": args.prototype_mode,
        "prototype_feature_names": prototype_feature_names,
        "prototype_bank": prototypes.tolist(),
        "input_size": input_size,
        "prototype_dim": int(prototypes.shape[1]),
        "train_rows": int(len(train_dataset)),
        "val_rows": int(len(val_dataset)),
        "test_rows": int(len(test_dataset)),
        "args": json_safe_args(args),
    }
    (args.out_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    best_state = None
    best_key = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, scaler, device, args)
        val_metrics, _ = evaluate(model, val_loader, val_dataset.frame, class_names, device)
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_stats.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        key = selection_key(val_metrics, args.selection_mode)
        if best_key is None or key > best_key:
            best_key = key
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            torch.save(
                {
                    "model_state_dict": best_state,
                    "class_names": class_names,
                    "prototype_feature_names": prototype_feature_names,
                    "prototype_bank": prototypes.tolist(),
                    "prototype_mode": args.prototype_mode,
                    "input_size": input_size,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "args": json_safe_args(args),
                },
                args.out_dir / "best_model.pth",
            )
        pd.DataFrame(history).to_csv(args.out_dir / "history.csv", index=False, encoding="utf-8-sig")
        print(
            f"epoch={epoch:03d} loss={train_stats['loss']:.4f} "
            f"val_visual_acc={val_metrics['visual_accuracy']:.4f} "
            f"val_sem_acc={val_metrics['semantic_accuracy']:.4f} "
            f"val_fused_acc={val_metrics['fused_accuracy']:.4f} "
            f"gate={val_metrics['effective_gate_mean']:.4f} "
            f"switch={val_metrics['switch_count']} net={val_metrics['net_correct_change_vs_visual_branch']}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    val_metrics, val_predictions = evaluate(model, val_loader, val_dataset.frame, class_names, device)
    test_metrics, test_predictions = evaluate(model, test_loader, test_dataset.frame, class_names, device)
    val_predictions.to_csv(args.out_dir / "val_predictions.csv", index=False, encoding="utf-8-sig")
    test_predictions.to_csv(args.out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(args.out_dir / "history.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(
        confusion_matrix(
            test_predictions["true_label"].map(class_to_idx).to_numpy(dtype=int),
            test_predictions["fused_pred_label"].map(class_to_idx).to_numpy(dtype=int),
            labels=list(range(len(class_names))),
        ),
        index=class_names,
        columns=class_names,
    ).to_csv(args.out_dir / "test_fused_confusion_matrix.csv", encoding="utf-8-sig")

    summary = {
        "method": "prototype_wheat_sage_no_label_leak",
        "prototype_mode": args.prototype_mode,
        "selection": f"best validation checkpoint by selection_mode={args.selection_mode}",
        "freeze_visual": args.freeze_visual,
        "semantic_dim": int(prototypes.shape[1]),
        "input_size": input_size,
        "class_names": class_names,
        "best_val": val_metrics,
        "test": test_metrics,
        "test_fused_change_vs_visual_branch": float(test_metrics["fused_accuracy"] - test_metrics["visual_accuracy"]),
        "outputs": {
            "history": str(args.out_dir / "history.csv"),
            "test_predictions": str(args.out_dir / "test_predictions.csv"),
            "best_model": str(args.out_dir / "best_model.pth"),
        },
    }
    (args.out_dir / "image_level_wheat_sage_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.out_dir / "prototype_feature_names.json").write_text(
        json.dumps(prototype_feature_names, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Output: {args.out_dir}")


if __name__ == "__main__":
    main()
