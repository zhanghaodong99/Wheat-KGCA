from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion.stage2_cam.generate_convnext_gradcam import DEFAULT_CHECKPOINT, load_checkpoint  # noqa: E402
from src.fusion.stage4_discrepancy_gated.train_discrepancy_gated_fusion import (  # noqa: E402
    DEFAULT_TEST_MANIFEST,
    DEFAULT_TRAIN_MANIFEST,
    DEFAULT_VAL_MANIFEST,
    FusionImageDataset,
    GRID_NAMES,
    GRID_TO_INDEX,
    build_evidence_lookup,
    discrepancy_gated_alignment_loss,
    prepare_split,
    to_nchw,
)
from src.fusion.stage5_attention_fusion.train_attention_fusion import (  # noqa: E402
    DEFAULT_TEST_SEMANTIC,
    DEFAULT_TRAIN_EVIDENCE,
    DEFAULT_TRAIN_SEMANTIC,
    DEFAULT_VAL_EVIDENCE,
    DEFAULT_VAL_SEMANTIC,
)
from src.fusion.train_semantic_visual_module import SemanticVectorizer  # noqa: E402
from src.models.convnextv2_tiny import build_model  # noqa: E402


AUX_PAIRS = [
    ("seedling", "tillering"),
    ("heading", "flowering"),
    ("flowering", "grain_filling"),
]
DEFAULT_EXPERT_REGION_PRIOR = PROJECT_ROOT / "outputs" / "paper_full_experiments" / "expert_spatial_priors" / "expert_spatial_prior_config.json"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
    }


def limit_split(
    frame: pd.DataFrame,
    semantic: np.ndarray,
    max_samples: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    if max_samples <= 0 or len(frame) <= max_samples:
        return frame, semantic
    frame = frame.reset_index(drop=True)
    class_count = max(int(frame["stage_id"].nunique()), 1)
    per_class = max(1, math.ceil(max_samples / class_count))
    positions: list[int] = []
    for _, group in frame.groupby("stage_id", sort=False):
        positions.extend([int(index) for index in group.index[:per_class]])
    positions = sorted(positions[:max_samples])
    return frame.iloc[positions].reset_index(drop=True), semantic[np.array(positions, dtype=np.int64)]


def load_expert_region_prior_matrix(path: Path, class_names: list[str]) -> torch.Tensor:
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    matrix = np.zeros((len(class_names), len(GRID_NAMES)), dtype=np.float32)
    for class_index, class_name in enumerate(class_names):
        for item in config.get(class_name, []):
            grid = str(item.get("grid", "")).strip()
            if grid not in GRID_TO_INDEX:
                continue
            confidence = min(max(float(item.get("confidence", 0.0)), 0.0), 1.0)
            visibility_weight = 1.0 if str(item.get("visibility", "")).strip() == "yes" else 0.55
            matrix[class_index, GRID_TO_INDEX[grid]] = max(
                matrix[class_index, GRID_TO_INDEX[grid]],
                confidence * visibility_weight,
            )
        row_sum = float(matrix[class_index].sum())
        if row_sum <= 0:
            matrix[class_index] = 1.0 / len(GRID_NAMES)
        else:
            matrix[class_index] /= row_sum
    return torch.from_numpy(matrix)


def pair_id(pair: tuple[str, str]) -> str:
    return f"{pair[0]}__vs__{pair[1]}"


def build_pair_index(
    class_to_idx: dict[str, int],
    aux_pairs: list[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    specs = []
    selected_pairs = aux_pairs if aux_pairs is not None else AUX_PAIRS
    for pair_index, (left, right) in enumerate(selected_pairs):
        if left not in class_to_idx or right not in class_to_idx:
            raise RuntimeError(f"Aux pair has class missing from checkpoint: {left}, {right}")
        specs.append(
            {
                "pair_index": pair_index,
                "pair_id": pair_id((left, right)),
                "left": left,
                "right": right,
                "left_class_idx": class_to_idx[left],
                "right_class_idx": class_to_idx[right],
            }
        )
    return specs


def parse_pair_name(pair_name: str) -> tuple[str, str] | None:
    parts = [part.strip() for part in str(pair_name).split("__vs__") if part.strip()]
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def make_soft_target(
    class_to_idx: dict[str, int],
    primary_label: str,
    secondary_label: str,
    primary_weight: float,
    secondary_weight: float,
) -> list[float] | None:
    if primary_label not in class_to_idx:
        return None
    total = max(float(primary_weight) + float(secondary_weight), 1e-8)
    target = [0.0 for _ in class_to_idx]
    target[class_to_idx[primary_label]] = float(primary_weight) / total
    if secondary_label in class_to_idx and float(secondary_weight) > 0:
        target[class_to_idx[secondary_label]] = float(secondary_weight) / total
    return target


def load_expert_rule_config(path: Path | None, class_to_idx: dict[str, int]) -> dict[str, Any]:
    if path is None:
        return {"aux_pairs": None, "soft_rules": {}, "soft_rules_by_label_set": {}, "summary": {}}
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"rule_id", "rule_type", "pair_id", "primary_label", "secondary_label", "primary_weight", "secondary_weight"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Expert rules csv missing columns: {missing}")
    soft_rows = frame[frame["rule_type"].astype(str).str.strip().eq("soft_label")].copy()
    aux_pairs: list[tuple[str, str]] = []
    soft_rules: dict[tuple[str, str], dict[str, Any]] = {}
    soft_rules_by_label_set: dict[frozenset[str], dict[str, dict[str, Any]]] = {}
    for _, row in soft_rows.iterrows():
        pair_name = str(row["pair_id"]).strip()
        parsed_pair = parse_pair_name(pair_name)
        if parsed_pair is None:
            continue
        left, right = parsed_pair
        if left not in class_to_idx or right not in class_to_idx:
            continue
        if (left, right) not in aux_pairs:
            aux_pairs.append((left, right))
        primary_label = str(row["primary_label"]).strip()
        secondary_label = str(row["secondary_label"]).strip()
        target = make_soft_target(
            class_to_idx,
            primary_label,
            secondary_label,
            float(row["primary_weight"]),
            float(row["secondary_weight"]),
        )
        if target is None:
            continue
        rule = {
            "rule_id": str(row["rule_id"]).strip(),
            "pair_id": pair_name,
            "primary_label": primary_label,
            "secondary_label": secondary_label,
            "primary_weight": float(row["primary_weight"]),
            "secondary_weight": float(row["secondary_weight"]),
            "target": target,
        }
        soft_rules[(pair_name, primary_label)] = rule
        label_set = frozenset([primary_label, secondary_label])
        soft_rules_by_label_set.setdefault(label_set, {})[primary_label] = rule
    return {
        "aux_pairs": aux_pairs,
        "soft_rules": soft_rules,
        "soft_rules_by_label_set": soft_rules_by_label_set,
        "summary": {
            "expert_rules_csv": str(path),
            "soft_rule_count": len(soft_rules),
            "aux_pairs": [pair_id(pair) for pair in aux_pairs],
        },
    }


def resolve_review_soft_rule(row: pd.Series, expert_rule_config: dict[str, Any]) -> dict[str, Any] | None:
    soft_rules: dict[tuple[str, str], dict[str, Any]] = expert_rule_config["soft_rules"]
    soft_rules_by_label_set: dict[frozenset[str], dict[str, dict[str, Any]]] = expert_rule_config[
        "soft_rules_by_label_set"
    ]
    review_label = str(row.get("review_label", "")).strip()
    labels = [part.strip() for part in review_label.split("|") if part.strip()]
    pair_name = str(row.get("pair_id", "")).strip()
    primary_label = str(row.get("pair_training_label", "")).strip()
    stage_label = str(row.get("stage_id", "")).strip()
    if not primary_label and stage_label in labels:
        primary_label = stage_label
    if not primary_label and labels:
        primary_label = labels[0]
    if pair_name and primary_label:
        rule = soft_rules.get((pair_name, primary_label))
        if rule is not None:
            return rule
    if len(labels) >= 2 and primary_label:
        return soft_rules_by_label_set.get(frozenset(labels), {}).get(primary_label)
    return None


def resolve_explicit_soft_target(row: pd.Series, class_to_idx: dict[str, int]) -> dict[str, Any] | None:
    primary_label = str(row.get("soft_primary_label", "")).strip()
    secondary_label = str(row.get("soft_secondary_label", "")).strip()
    if not primary_label:
        primary_label = str(row.get("primary_label_user", "")).strip()
    if not secondary_label:
        secondary_label = str(row.get("secondary_label_user", "")).strip()
    primary_weight_raw = row.get("soft_primary_weight", row.get("primary_weight_user", ""))
    secondary_weight_raw = row.get("soft_secondary_weight", row.get("secondary_weight_user", ""))
    if primary_label not in class_to_idx:
        return None
    try:
        primary_weight = float(primary_weight_raw)
    except (TypeError, ValueError):
        primary_weight = 1.0
    try:
        secondary_weight = float(secondary_weight_raw)
    except (TypeError, ValueError):
        secondary_weight = 0.0
    target = make_soft_target(class_to_idx, primary_label, secondary_label, primary_weight, secondary_weight)
    if target is None:
        return None
    return {
        "rule_id": str(row.get("rule_id", "explicit_review")).strip(),
        "primary_label": primary_label,
        "secondary_label": secondary_label,
        "primary_weight": primary_weight,
        "secondary_weight": secondary_weight,
        "target": target,
    }


def load_boundary_review(
    path: Path | None,
    class_to_idx: dict[str, int],
    allowed_splits: set[str] | None = None,
    expert_rule_config: dict[str, Any] | None = None,
    boundary_soft_main_weight: float = 1.0,
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    if path is None:
        return {}, {}
    frame = pd.read_csv(path)
    main_lookup: dict[str, dict[str, Any]] = {}
    pair_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in frame.iterrows():
        if allowed_splits is not None:
            review_split = str(row.get("review_split", "")).strip()
            if review_split not in allowed_splits:
                continue
        image_id = str(row["image_id"])
        status = str(row.get("review_status", "")).strip()
        review_label = str(row.get("review_label", "")).strip()
        weight = float(row.get("boundary_weight_clean", row.get("boundary_weight", 1.0)))
        usable = str(row.get("usable_for_pair_training", "False")).lower() in ["true", "1", "yes"]
        pair_name = str(row.get("pair_id", "")).strip()
        pair_training_label = str(row.get("pair_training_label", "")).strip()
        use_for_main = str(row.get("use_for_main_training", "True")).lower() in ["true", "1", "yes"]
        if use_for_main:
            current = main_lookup.setdefault(
                image_id,
                {"label_idx": None, "soft_target": None, "weight": 1.0, "status": status, "soft_rule_id": ""},
            )
            if status == "clear_corrected" and review_label in class_to_idx:
                current["label_idx"] = class_to_idx[review_label]
                current["weight"] = min(float(current["weight"]), 1.0)
                current["status"] = status
            elif status == "clear_original" and review_label in class_to_idx:
                current["label_idx"] = class_to_idx[review_label]
                current["weight"] = min(float(current["weight"]), 1.0)
                current["status"] = status
            elif status == "boundary":
                current["weight"] = min(float(current["weight"]), weight)
                current["status"] = status
                soft_rule = resolve_explicit_soft_target(row, class_to_idx)
                if soft_rule is None and expert_rule_config is not None:
                    soft_rule = resolve_review_soft_rule(row, expert_rule_config)
                if soft_rule is not None:
                    current["soft_target"] = soft_rule["target"]
                    current["label_idx"] = class_to_idx[soft_rule["primary_label"]]
                    current["weight"] = float(boundary_soft_main_weight)
                    current["soft_rule_id"] = soft_rule["rule_id"]
            elif status == "uncertain":
                current["weight"] = 0.0
                current["status"] = status
        if pair_name:
            pair_lookup[(image_id, pair_name)] = {
                "usable": usable,
                "label": pair_training_label,
                "weight": float(row.get("pair_training_weight", weight if usable else 0.0)),
                "status": status,
            }
    return main_lookup, pair_lookup


class ExpertRegionMatrix(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        prior_matrix: torch.Tensor,
        region_logit_weight: float,
        trainable_matrix: bool = True,
    ) -> None:
        super().__init__()
        if tuple(prior_matrix.shape) != (num_classes, len(GRID_NAMES)):
            raise ValueError(
                f"Expected expert region matrix shape {(num_classes, len(GRID_NAMES))}, "
                f"got {tuple(prior_matrix.shape)}"
            )
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.region_logit_weight = float(region_logit_weight)
        self.register_buffer("expert_prior", prior_matrix.float())
        if trainable_matrix:
            self.matrix_delta = nn.Parameter(torch.zeros_like(self.expert_prior))
        else:
            self.register_parameter("matrix_delta", None)
        self.class_region_weight = nn.Parameter(torch.zeros(num_classes, feature_dim))
        self.class_region_bias = nn.Parameter(torch.zeros(num_classes))

    def current_matrix(self) -> torch.Tensor:
        logits = self.expert_prior.clamp_min(1e-6).log()
        if self.matrix_delta is not None:
            logits = logits + self.matrix_delta
        return torch.softmax(logits, dim=1)

    def forward(self, raw_features: torch.Tensor) -> torch.Tensor:
        features = to_nchw(raw_features, self.feature_dim)
        region_tokens = F.adaptive_avg_pool2d(features, (3, 3)).flatten(2).transpose(1, 2)
        matrix = self.current_matrix().to(device=region_tokens.device, dtype=region_tokens.dtype)
        class_region_tokens = torch.einsum("cr,brd->bcd", matrix, region_tokens)
        class_weight = self.class_region_weight.to(dtype=region_tokens.dtype)
        region_logits = (class_region_tokens * class_weight.unsqueeze(0)).sum(dim=2)
        region_logits = region_logits / math.sqrt(float(self.feature_dim))
        region_logits = region_logits + self.class_region_bias.to(dtype=region_tokens.dtype)
        return self.region_logit_weight * region_logits


class MultiScaleAttentionRefiner(nn.Module):
    def __init__(self, feature_dim: int, dropout: float = 0.05) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.local_branch = nn.Conv2d(feature_dim, feature_dim, kernel_size=1, groups=1, bias=False)
        self.mid_branch = nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, groups=feature_dim, bias=False)
        self.wide_branch = nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=2, dilation=2, groups=feature_dim, bias=False)
        self.mix = nn.Sequential(
            nn.Conv2d(feature_dim * 3, feature_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, feature_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )
        hidden_dim = max(feature_dim // 16, 16)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, feature_dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )
        self.output = nn.Conv2d(feature_dim, feature_dim, kernel_size=1, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, raw_features: torch.Tensor) -> torch.Tensor:
        features = to_nchw(raw_features, self.feature_dim)
        multi_scale = torch.cat(
            [
                self.local_branch(features),
                self.mid_branch(features),
                self.wide_branch(features),
            ],
            dim=1,
        )
        mixed = self.mix(multi_scale)
        channel_weight = self.channel_gate(mixed)
        spatial_input = torch.cat([mixed.mean(dim=1, keepdim=True), mixed.amax(dim=1, keepdim=True)], dim=1)
        spatial_weight = self.spatial_gate(spatial_input)
        refined_delta = self.output(mixed * channel_weight * spatial_weight)
        return features + refined_delta


class HierarchicalAuxFusion(nn.Module):
    def __init__(
        self,
        visual_model: nn.Module,
        semantic_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_aux_heads: int = 3,
        expert_region_prior: torch.Tensor | None = None,
        region_logit_weight: float = 0.0,
        trainable_expert_region_matrix: bool = True,
        selective_region_gate: bool = False,
        region_gate_margin: float = 0.20,
        region_gate_temperature: float = 12.0,
        use_multiscale_attention: bool = False,
        multiscale_logit_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.visual = visual_model
        self.semantic_encoder = nn.Sequential(
            nn.Linear(semantic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.semantic_delta = nn.Linear(hidden_dim, num_classes)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim + num_classes + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.aux_heads = nn.ModuleList([nn.Linear(hidden_dim + num_classes + 3, 2) for _ in range(num_aux_heads)])
        self.expert_region = None
        self.selective_region_gate = bool(selective_region_gate)
        self.region_gate_margin = float(region_gate_margin)
        self.region_gate_temperature = float(region_gate_temperature)
        self.multiscale_attention = (
            MultiScaleAttentionRefiner(int(visual_model.feature_dim)) if use_multiscale_attention else None
        )
        self.multiscale_logit_weight = float(multiscale_logit_weight)
        if expert_region_prior is not None and region_logit_weight > 0:
            self.expert_region = ExpertRegionMatrix(
                feature_dim=int(visual_model.feature_dim),
                num_classes=num_classes,
                prior_matrix=expert_region_prior,
                region_logit_weight=region_logit_weight,
                trainable_matrix=trainable_expert_region_matrix,
            )
        nn.init.zeros_(self.semantic_delta.weight)
        nn.init.zeros_(self.semantic_delta.bias)
        nn.init.constant_(self.gate[-1].bias, -3.0)

    def forward(self, images: torch.Tensor, semantic: torch.Tensor):
        visual_logits, raw_features = self.visual(images)
        if self.multiscale_attention is not None and self.multiscale_logit_weight > 0:
            refined_features = self.multiscale_attention(raw_features)
            refined_logits = self.visual.head(refined_features.mean(dim=[-2, -1]))
            visual_logits = visual_logits + self.multiscale_logit_weight * (refined_logits - visual_logits)
        semantic_token = self.semantic_encoder(semantic)
        probabilities = torch.softmax(visual_logits, dim=1)
        top_two = torch.topk(probabilities, k=2, dim=1).values
        uncertainty = torch.cat(
            [
                top_two[:, :1],
                top_two[:, :1] - top_two[:, 1:2],
                -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=1, keepdim=True),
            ],
            dim=1,
        )
        shared = torch.cat([semantic_token, visual_logits, uncertainty], dim=1)
        gate = torch.sigmoid(self.gate(shared))
        fused_logits = visual_logits + gate * self.semantic_delta(semantic_token)
        if self.expert_region is not None:
            region_logits = self.expert_region(raw_features)
            if self.selective_region_gate:
                margin_gate = torch.sigmoid(
                    (
                        self.region_gate_margin
                        - (top_two[:, :1] - top_two[:, 1:2])
                    )
                    * self.region_gate_temperature
                )
                region_logits = margin_gate * region_logits
            fused_logits = fused_logits + region_logits
        aux_logits = [head(shared) for head in self.aux_heads]
        return fused_logits, visual_logits, raw_features, gate, aux_logits


def aux_pair_losses(
    aux_logits: list[torch.Tensor],
    labels: torch.Tensor,
    pair_specs: list[dict[str, Any]],
    image_ids: tuple[str, ...] | list[str],
    pair_review_lookup: dict[tuple[str, str], dict[str, Any]],
    class_to_idx: dict[str, int],
    use_boundary_pair_weights: bool,
    sample_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, int]]:
    losses = []
    counts: dict[str, int] = {}
    image_ids_list = [str(value) for value in image_ids]
    for spec, logits in zip(pair_specs, aux_logits):
        left_idx = int(spec["left_class_idx"])
        right_idx = int(spec["right_class_idx"])
        selected_indices = []
        targets = []
        weights = []
        pair_name = str(spec["pair_id"])
        for batch_index, image_id in enumerate(image_ids_list):
            if sample_mask is not None and not bool(sample_mask[batch_index].item()):
                continue
            label_value = int(labels[batch_index].item())
            if label_value not in [left_idx, right_idx]:
                continue
            target_label_idx = label_value
            sample_weight = 1.0
            if use_boundary_pair_weights:
                review = pair_review_lookup.get((image_id, pair_name))
                if review is not None:
                    if not bool(review["usable"]):
                        continue
                    review_label = str(review["label"])
                    if review_label in class_to_idx:
                        target_label_idx = class_to_idx[review_label]
                    sample_weight = float(review["weight"])
                    if sample_weight <= 0:
                        continue
            selected_indices.append(batch_index)
            targets.append(1 if target_label_idx == right_idx else 0)
            weights.append(sample_weight)
        count = len(selected_indices)
        counts[f"{spec['pair_id']}_samples"] = count
        if count == 0:
            continue
        index_tensor = torch.tensor(selected_indices, device=labels.device, dtype=torch.long)
        binary_target = torch.tensor(targets, device=labels.device, dtype=torch.long)
        weight_tensor = torch.tensor(weights, device=labels.device, dtype=logits.dtype)
        per_sample = F.cross_entropy(logits[index_tensor], binary_target, reduction="none")
        losses.append((per_sample * weight_tensor).sum() / weight_tensor.sum().clamp_min(1e-8))
    if not losses:
        return labels.new_zeros((), dtype=torch.float32), counts
    return torch.stack(losses).mean(), counts


def main_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    image_ids: tuple[str, ...] | list[str],
    main_review_lookup: dict[str, dict[str, Any]],
    use_boundary_main_labels: bool,
    use_rule_soft_labels: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not use_boundary_main_labels and not use_rule_soft_labels:
        target_probs = F.one_hot(labels, num_classes=logits.shape[1]).to(dtype=logits.dtype)
        weights = torch.ones_like(labels, dtype=logits.dtype)
        return F.cross_entropy(logits, labels), labels, target_probs, weights
    effective_labels = labels.clone()
    weights = torch.ones_like(labels, dtype=logits.dtype)
    target_probs = F.one_hot(labels, num_classes=logits.shape[1]).to(dtype=logits.dtype)
    for batch_index, image_id_value in enumerate(image_ids):
        review = main_review_lookup.get(str(image_id_value))
        if review is None:
            continue
        if use_rule_soft_labels and review.get("soft_target") is not None:
            soft_target = torch.tensor(review["soft_target"], device=logits.device, dtype=logits.dtype)
            target_probs[batch_index] = soft_target
            effective_labels[batch_index] = int(torch.argmax(soft_target).item())
        elif use_boundary_main_labels and review["label_idx"] is not None:
            effective_labels[batch_index] = int(review["label_idx"])
            target_probs[batch_index] = F.one_hot(
                effective_labels[batch_index],
                num_classes=logits.shape[1],
            ).to(dtype=logits.dtype)
        weights[batch_index] = float(review["weight"])
    log_probs = F.log_softmax(logits, dim=1)
    per_sample = -(target_probs * log_probs).sum(dim=1)
    loss = (per_sample * weights).sum() / weights.sum().clamp_min(1e-8)
    return loss, effective_labels, target_probs, weights


def build_selective_boundary_mask(
    visual_logits: torch.Tensor,
    labels: torch.Tensor,
    image_ids: tuple[str, ...] | list[str],
    pair_specs: list[dict[str, Any]],
    main_review_lookup: dict[str, dict[str, Any]],
    margin_threshold: float,
) -> torch.Tensor:
    probabilities = torch.softmax(visual_logits.detach(), dim=1)
    top = torch.topk(probabilities, k=2, dim=1)
    top_indices = top.indices
    top_margin = top.values[:, 0] - top.values[:, 1]
    adjacent_mask = torch.zeros(labels.shape[0], device=labels.device, dtype=torch.bool)
    for spec in pair_specs:
        left_idx = int(spec["left_class_idx"])
        right_idx = int(spec["right_class_idx"])
        pair_mask = (
            ((top_indices[:, 0] == left_idx) & (top_indices[:, 1] == right_idx))
            | ((top_indices[:, 0] == right_idx) & (top_indices[:, 1] == left_idx))
        )
        adjacent_mask |= pair_mask
    low_margin_mask = top_margin <= float(margin_threshold)
    review_mask = torch.tensor(
        [
            str(main_review_lookup.get(str(image_id), {}).get("status", "")).strip()
            in {"boundary", "clear_corrected"}
            for image_id in image_ids
        ],
        device=labels.device,
        dtype=torch.bool,
    )
    return (low_margin_mask & adjacent_mask) | review_mask


def selective_boundary_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    image_ids: tuple[str, ...] | list[str],
    main_review_lookup: dict[str, dict[str, Any]],
    active_mask: torch.Tensor,
    use_boundary_main_labels: bool,
    use_rule_soft_labels: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    effective_labels = labels.clone()
    target_probs = F.one_hot(labels, num_classes=logits.shape[1]).to(dtype=logits.dtype)
    weights = active_mask.to(dtype=logits.dtype)
    for batch_index, image_id_value in enumerate(image_ids):
        if not bool(active_mask[batch_index].item()):
            continue
        review = main_review_lookup.get(str(image_id_value))
        if review is None:
            continue
        if use_rule_soft_labels and review.get("soft_target") is not None:
            soft_target = torch.tensor(review["soft_target"], device=logits.device, dtype=logits.dtype)
            target_probs[batch_index] = soft_target
            effective_labels[batch_index] = int(torch.argmax(soft_target).item())
        elif use_boundary_main_labels and review["label_idx"] is not None:
            effective_labels[batch_index] = int(review["label_idx"])
            target_probs[batch_index] = F.one_hot(
                effective_labels[batch_index],
                num_classes=logits.shape[1],
            ).to(dtype=logits.dtype)
        weights[batch_index] = float(review["weight"])
    if not active_mask.any():
        return logits.new_zeros(()), effective_labels, target_probs, weights
    log_probs = F.log_softmax(logits, dim=1)
    per_sample = -(target_probs * log_probs).sum(dim=1)
    active_weights = weights * active_mask.to(dtype=weights.dtype)
    loss = (per_sample * active_weights).sum() / active_weights.sum().clamp_min(1e-8)
    return loss, effective_labels, target_probs, active_weights


def ordered_stage_indices(class_names: list[str]) -> list[int]:
    canonical_order = [
        "seedling",
        "tillering",
        "stem_elongation",
        "flag_leaf_booting",
        "heading",
        "flowering",
        "grain_filling",
        "maturity",
    ]
    missing = [name for name in canonical_order if name not in class_names]
    if missing:
        raise RuntimeError(f"Ordinal stage order is missing labels: {missing}")
    return [class_names.index(name) for name in canonical_order]


def ordinal_stage_loss(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
    sample_weights: torch.Tensor,
    class_names: list[str],
    sample_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    order = ordered_stage_indices(class_names)
    ordered_probabilities = torch.softmax(logits, dim=1)[:, order]
    ordered_targets = target_probs[:, order]
    # P(stage > threshold) for thresholds between consecutive growth stages.
    cumulative_probabilities = torch.flip(
        torch.cumsum(torch.flip(ordered_probabilities, dims=[1]), dim=1),
        dims=[1],
    )[:, :-1]
    cumulative_targets = torch.flip(
        torch.cumsum(torch.flip(ordered_targets, dims=[1]), dim=1),
        dims=[1],
    )[:, :-1]
    per_sample = F.binary_cross_entropy(
        cumulative_probabilities.clamp(1e-6, 1.0 - 1e-6),
        cumulative_targets,
        reduction="none",
    ).mean(dim=1)
    if sample_mask is not None:
        sample_weights = sample_weights * sample_mask.to(dtype=sample_weights.dtype)
    return (per_sample * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)


@torch.no_grad()
def evaluate(
    model: HierarchicalAuxFusion,
    loader: DataLoader,
    device: torch.device,
    pair_specs: list[dict[str, Any]],
    aux_refine_weight: float,
    aux_refine_margin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, int]]:
    model.eval()
    labels_all = []
    predictions_all = []
    refined_predictions_all = []
    gates = []
    refine_counts = {"eligible": 0, "changed": 0}
    for images, semantic, labels, _, _, _, _ in loader:
        fused_logits, _, _, gate, aux_logits = model(images.to(device), semantic.to(device))
        labels_np = labels.numpy()
        logits = fused_logits.cpu()
        refined = logits.clone()
        probabilities = torch.softmax(logits, dim=1)
        top = torch.topk(probabilities, k=2, dim=1)
        top_indices = top.indices
        top_margin = top.values[:, 0] - top.values[:, 1]
        for spec, pair_aux_logits in zip(pair_specs, aux_logits):
            left_idx = int(spec["left_class_idx"])
            right_idx = int(spec["right_class_idx"])
            pair_set = {left_idx, right_idx}
            pair_top = [
                index
                for index in range(len(labels_np))
                if set(top_indices[index].tolist()) == pair_set and float(top_margin[index].item()) <= aux_refine_margin
            ]
            if not pair_top:
                continue
            refine_counts["eligible"] += len(pair_top)
            aux_prob = torch.softmax(pair_aux_logits.detach().cpu()[pair_top], dim=1)
            delta = aux_refine_weight * (aux_prob[:, 1] - aux_prob[:, 0])
            refined[pair_top, right_idx] += delta
            refined[pair_top, left_idx] -= delta
        pred = logits.argmax(dim=1).numpy()
        refined_pred = refined.argmax(dim=1).numpy()
        refine_counts["changed"] += int((pred != refined_pred).sum())
        labels_all.append(labels_np)
        predictions_all.append(pred)
        refined_predictions_all.append(refined_pred)
        gates.append(gate.cpu().numpy())
    return (
        np.concatenate(labels_all),
        np.concatenate(predictions_all),
        np.concatenate(refined_predictions_all),
        float(np.concatenate(gates).mean()),
        refine_counts,
    )


@torch.no_grad()
def evaluate_visual(model: HierarchicalAuxFusion, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels_all = []
    predictions_all = []
    for images, _, labels, _, _, _, _ in loader:
        logits, _ = model.visual(images.to(device))
        labels_all.append(labels.numpy())
        predictions_all.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(labels_all), np.concatenate(predictions_all)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train unified Wheat-SAGE-CAM with adjacent-stage auxiliary heads.")
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--val-manifest", type=Path, default=DEFAULT_VAL_MANIFEST)
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_TEST_MANIFEST)
    parser.add_argument("--semantic-train", type=Path, default=DEFAULT_TRAIN_SEMANTIC)
    parser.add_argument("--semantic-val", type=Path, default=DEFAULT_VAL_SEMANTIC)
    parser.add_argument("--semantic-test", type=Path, default=DEFAULT_TEST_SEMANTIC)
    parser.add_argument("--train-evidence", type=Path, default=DEFAULT_TRAIN_EVIDENCE)
    parser.add_argument("--val-evidence", type=Path, default=DEFAULT_VAL_EVIDENCE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-align", type=float, default=0.0025)
    parser.add_argument("--lambda-pair", type=float, default=0.2)
    parser.add_argument("--lambda-ordinal", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--aux-refine-weight", type=float, default=0.0)
    parser.add_argument("--aux-refine-margin", type=float, default=0.20)
    parser.add_argument("--selection-target", choices=["auto", "main", "refined"], default="auto")
    parser.add_argument("--primary-output", choices=["main", "refined"], default="main")
    parser.add_argument("--use-expert-region-matrix", action="store_true")
    parser.add_argument("--expert-region-prior-json", type=Path, default=DEFAULT_EXPERT_REGION_PRIOR)
    parser.add_argument("--region-logit-weight", type=float, default=0.2)
    parser.add_argument("--freeze-expert-region-matrix", action="store_true")
    parser.add_argument("--selective-boundary-gate", action="store_true")
    parser.add_argument("--boundary-gate-margin", type=float, default=0.20)
    parser.add_argument("--region-gate-temperature", type=float, default=12.0)
    parser.add_argument("--lambda-boundary", type=float, default=0.5)
    parser.add_argument("--use-multiscale-attention", action="store_true")
    parser.add_argument("--multiscale-logit-weight", type=float, default=1.0)
    parser.add_argument("--boundary-review-csv", type=Path, default=None)
    parser.add_argument("--expert-rules-csv", type=Path, default=None)
    parser.add_argument("--aux-pairs-source", choices=["default", "expert_soft_rules"], default="default")
    parser.add_argument(
        "--boundary-review-splits",
        type=str,
        default="",
        help="Comma-separated review splits allowed for boundary supervision, e.g. train,val. Empty means no filter.",
    )
    parser.add_argument("--use-boundary-pair-weights", action="store_true")
    parser.add_argument("--use-boundary-main-labels", action="store_true")
    parser.add_argument("--use-rule-soft-labels", action="store_true")
    parser.add_argument("--boundary-soft-main-weight", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--unfreeze-encoder", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(args.checkpoint, device)
    class_names = [str(value) for value in checkpoint["class_names"]]
    class_to_idx = {name: index for index, name in enumerate(class_names)}
    expert_rule_config = load_expert_rule_config(args.expert_rules_csv, class_to_idx)
    aux_pairs = expert_rule_config["aux_pairs"] if args.aux_pairs_source == "expert_soft_rules" else None
    pair_specs = build_pair_index(class_to_idx, aux_pairs)
    allowed_review_splits = (
        {split.strip() for split in args.boundary_review_splits.split(",") if split.strip()}
        if args.boundary_review_splits.strip()
        else None
    )
    main_review_lookup, pair_review_lookup = load_boundary_review(
        args.boundary_review_csv,
        class_to_idx,
        allowed_review_splits,
        expert_rule_config,
        args.boundary_soft_main_weight,
    )

    vectorizer = SemanticVectorizer()
    train_frame, train_semantic = prepare_split(
        args.train_manifest, args.semantic_train, vectorizer, True, class_to_idx
    )
    val_frame, val_semantic = prepare_split(args.val_manifest, args.semantic_val, vectorizer, False, class_to_idx)
    test_frame, test_semantic = prepare_split(args.test_manifest, args.semantic_test, vectorizer, False, class_to_idx)
    train_frame, train_semantic = limit_split(train_frame, train_semantic, args.max_train_samples)
    val_frame, val_semantic = limit_split(val_frame, val_semantic, args.max_val_samples)
    test_frame, test_semantic = limit_split(test_frame, test_semantic, args.max_test_samples)

    train_labels = {str(row["image_id"]): str(row["stage_id"]) for _, row in train_frame.iterrows()}
    val_labels = {str(row["image_id"]): str(row["stage_id"]) for _, row in val_frame.iterrows()}
    train_evidence = build_evidence_lookup(args.train_evidence, train_labels, class_to_idx)
    val_evidence = build_evidence_lookup(args.val_evidence, val_labels, class_to_idx)
    if args.lambda_align > 0 and not train_evidence:
        raise RuntimeError("CAM alignment requires train evidence, but none was loaded.")
    expert_region_prior = None
    if args.use_expert_region_matrix:
        expert_region_prior = load_expert_region_prior_matrix(args.expert_region_prior_json, class_names)

    train_dataset = FusionImageDataset(train_frame, train_semantic, class_to_idx, train_evidence)
    val_dataset = FusionImageDataset(val_frame, val_semantic, class_to_idx, {})
    test_dataset = FusionImageDataset(test_frame, test_semantic, class_to_idx, {})
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(
        f"[startup] device={device} train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)} "
        f"batch_size={args.batch_size} train_batches={len(train_loader)} "
        f"boundary_main={len(main_review_lookup)} boundary_pair={len(pair_review_lookup)} "
        f"aux_pairs={[spec['pair_id'] for spec in pair_specs]} "
        f"expert_region_matrix={args.use_expert_region_matrix}",
        flush=True,
    )

    visual_model = build_model(num_classes=len(class_names), pretrained=False)
    visual_model.load_state_dict(checkpoint["model_state_dict"])
    if not args.unfreeze_encoder:
        visual_model.encoder.requires_grad_(False)
    model = HierarchicalAuxFusion(
        visual_model=visual_model,
        semantic_dim=train_semantic.shape[1],
        num_classes=len(class_names),
        hidden_dim=args.hidden_dim,
        num_aux_heads=len(pair_specs),
        expert_region_prior=expert_region_prior,
        region_logit_weight=args.region_logit_weight,
        trainable_expert_region_matrix=not args.freeze_expert_region_matrix,
        selective_region_gate=args.selective_boundary_gate,
        region_gate_margin=args.boundary_gate_margin,
        region_gate_temperature=args.region_gate_temperature,
        use_multiscale_attention=args.use_multiscale_attention,
        multiscale_logit_weight=args.multiscale_logit_weight,
    ).to(device)

    print("[startup] evaluating visual baseline before fusion training...", flush=True)
    baseline_labels, baseline_predictions = evaluate_visual(model, test_loader, device)
    baseline_metrics = metrics(baseline_labels, baseline_predictions)
    print(f"[startup] visual_baseline_acc={baseline_metrics['accuracy']:.4f}", flush=True)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    best_val_f1 = -1.0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        if not args.unfreeze_encoder:
            model.visual.encoder.eval()
        totals: dict[str, float] = {
            "loss": 0.0,
            "classification": 0.0,
            "alignment": 0.0,
            "pair": 0.0,
            "ordinal": 0.0,
            "boundary_active": 0.0,
            "active": 0.0,
            "mismatch": 0.0,
            "gain": 0.0,
        }
        pair_seen = {f"{spec['pair_id']}_samples": 0 for spec in pair_specs}
        batches = 0
        for batch_index, (images, semantic, labels, priors, targets, active, image_ids) in enumerate(train_loader, start=1):
            images = images.to(device)
            semantic = semantic.to(device)
            labels = labels.to(device)
            priors = priors.to(device)
            targets = targets.to(device)
            active = active.to(device)

            fused_logits, visual_logits, raw_features, _, aux_logits = model(images, semantic)
            if args.selective_boundary_gate:
                boundary_mask = build_selective_boundary_mask(
                    visual_logits=visual_logits,
                    labels=labels,
                    image_ids=image_ids,
                    pair_specs=pair_specs,
                    main_review_lookup=main_review_lookup,
                    margin_threshold=args.boundary_gate_margin,
                )
                base_classification_loss = F.cross_entropy(fused_logits, labels)
                boundary_loss, effective_labels, target_probs, sample_weights = selective_boundary_loss(
                    fused_logits,
                    labels,
                    image_ids,
                    main_review_lookup,
                    boundary_mask,
                    args.use_boundary_main_labels,
                    args.use_rule_soft_labels,
                )
                classification_loss = base_classification_loss + args.lambda_boundary * boundary_loss
            else:
                boundary_mask = torch.ones(labels.shape[0], device=labels.device, dtype=torch.bool)
                classification_loss, effective_labels, target_probs, sample_weights = main_classification_loss(
                    fused_logits,
                    labels,
                    image_ids,
                    main_review_lookup,
                    args.use_boundary_main_labels,
                    args.use_rule_soft_labels,
                )
            ordinal_loss = ordinal_stage_loss(
                fused_logits,
                target_probs,
                sample_weights,
                class_names,
                sample_mask=boundary_mask if args.selective_boundary_gate else None,
            )
            alignment_loss, active_count, mismatch_count, alignment_gain = discrepancy_gated_alignment_loss(
                model.visual, raw_features, priors, targets, active
            )
            pair_loss, counts = aux_pair_losses(
                aux_logits,
                effective_labels,
                pair_specs,
                image_ids,
                pair_review_lookup,
                class_to_idx,
                args.use_boundary_pair_weights,
                sample_mask=boundary_mask if args.selective_boundary_gate else None,
            )
            loss = (
                classification_loss
                + args.lambda_align * alignment_loss
                + args.lambda_pair * pair_loss
                + args.lambda_ordinal * ordinal_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            totals["loss"] += float(loss.item())
            totals["classification"] += float(classification_loss.item())
            totals["alignment"] += float(alignment_loss.item())
            totals["pair"] += float(pair_loss.item())
            totals["ordinal"] += float(ordinal_loss.item())
            totals["boundary_active"] += float(boundary_mask.sum().item())
            totals["active"] += active_count
            totals["mismatch"] += mismatch_count
            totals["gain"] += alignment_gain
            for key, value in counts.items():
                pair_seen[key] += value
            batches += 1
            if args.log_interval > 0 and (batch_index == 1 or batch_index % args.log_interval == 0):
                print(
                    f"epoch={epoch:02d} batch={batch_index:04d}/{len(train_loader)} "
                    f"loss={totals['loss'] / max(batches, 1):.4f} "
                    f"cls={totals['classification'] / max(batches, 1):.4f} "
                    f"align={totals['alignment'] / max(batches, 1):.4f} "
                    f"pair={totals['pair'] / max(batches, 1):.4f} "
                    f"ordinal={totals['ordinal'] / max(batches, 1):.4f} "
                    f"boundary={int(totals['boundary_active'])}",
                    flush=True,
                )

        val_labels_np, val_predictions, val_refined, val_gate, val_refine = evaluate(
            model, val_loader, device, pair_specs, args.aux_refine_weight, args.aux_refine_margin
        )
        val_metrics = metrics(val_labels_np, val_predictions)
        val_refined_metrics = metrics(val_labels_np, val_refined)
        row: dict[str, Any] = {
            "epoch": epoch,
            "loss": totals["loss"] / max(batches, 1),
            "classification_loss": totals["classification"] / max(batches, 1),
            "alignment_loss": totals["alignment"] / max(batches, 1),
            "pair_loss": totals["pair"] / max(batches, 1),
            "ordinal_loss": totals["ordinal"] / max(batches, 1),
            "boundary_active_samples": int(totals["boundary_active"]),
            "active_region_samples_seen": int(totals["active"]),
            "mismatch_samples_seen": int(totals["mismatch"]),
            "mean_alignment_gain": totals["gain"] / max(batches, 1),
            "val_gate_mean": val_gate,
            "val_refine_eligible": val_refine["eligible"],
            "val_refine_changed": val_refine["changed"],
            **{f"train_{key}": value for key, value in pair_seen.items()},
            **{f"val_{name}": value for name, value in val_metrics.items()},
            **{f"val_refined_{name}": value for name, value in val_refined_metrics.items()},
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} loss={row['loss']:.4f} cls={row['classification_loss']:.4f} "
            f"align={row['alignment_loss']:.4f} pair={row['pair_loss']:.4f} "
            f"ordinal={row['ordinal_loss']:.4f} "
            f"val_acc={row['val_accuracy']:.4f} val_f1={row['val_macro_f1']:.4f} "
            f"refined_acc={row['val_refined_accuracy']:.4f} gate={val_gate:.3f}",
            flush=True,
        )
        if args.selection_target == "main":
            select_f1 = val_metrics["macro_f1"]
        elif args.selection_target == "refined":
            select_f1 = val_refined_metrics["macro_f1"]
        else:
            select_f1 = val_refined_metrics["macro_f1"] if args.aux_refine_weight > 0 else val_metrics["macro_f1"]
        if select_f1 > best_val_f1:
            best_val_f1 = select_f1
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_labels, test_predictions, test_refined, test_gate, test_refine = evaluate(
        model, test_loader, device, pair_specs, args.aux_refine_weight, args.aux_refine_margin
    )
    final_metrics = metrics(test_labels, test_predictions)
    final_refined_metrics = metrics(test_labels, test_refined)
    primary_predictions = test_predictions if args.primary_output == "main" else test_refined
    primary_metrics = final_metrics if args.primary_output == "main" else final_refined_metrics
    prediction_frame = test_frame[["image_id", "image_path", "stage_id", "source_type"]].copy()
    prediction_frame["pred_label"] = [class_names[index] for index in test_predictions]
    prediction_frame["refined_pred_label"] = [class_names[index] for index in test_refined]
    prediction_frame["primary_pred_label"] = [class_names[index] for index in primary_predictions]
    prediction_frame["correct"] = prediction_frame["stage_id"] == prediction_frame["pred_label"]
    prediction_frame["refined_correct"] = prediction_frame["stage_id"] == prediction_frame["refined_pred_label"]
    prediction_frame["primary_correct"] = prediction_frame["stage_id"] == prediction_frame["primary_pred_label"]
    prediction_frame.to_csv(args.out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(args.out_dir / "history.csv", index=False, encoding="utf-8-sig")
    if model.expert_region is not None:
        region_matrix = model.expert_region.current_matrix().detach().cpu().numpy()
        pd.DataFrame(region_matrix, index=class_names, columns=GRID_NAMES).to_csv(
            args.out_dir / "expert_region_matrix.csv",
            encoding="utf-8-sig",
        )
    pd.DataFrame(
        confusion_matrix(prediction_frame["stage_id"], prediction_frame["pred_label"], labels=class_names),
        index=class_names,
        columns=class_names,
    ).to_csv(args.out_dir / "test_confusion_matrix.csv", encoding="utf-8-sig")
    pd.DataFrame(
        confusion_matrix(prediction_frame["stage_id"], prediction_frame["refined_pred_label"], labels=class_names),
        index=class_names,
        columns=class_names,
    ).to_csv(args.out_dir / "test_refined_confusion_matrix.csv", encoding="utf-8-sig")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "semantic_columns": vectorizer.columns,
            "pair_specs": pair_specs,
            "lambda_align": args.lambda_align,
            "lambda_pair": args.lambda_pair,
            "lambda_ordinal": args.lambda_ordinal,
            "aux_refine_weight": args.aux_refine_weight,
            "aux_refine_margin": args.aux_refine_margin,
            "selection_target": args.selection_target,
            "primary_output": args.primary_output,
            "boundary_review_csv": str(args.boundary_review_csv) if args.boundary_review_csv else None,
            "expert_rules_csv": str(args.expert_rules_csv) if args.expert_rules_csv else None,
            "aux_pairs_source": args.aux_pairs_source,
            "boundary_review_splits": sorted(allowed_review_splits) if allowed_review_splits is not None else [],
            "use_boundary_pair_weights": args.use_boundary_pair_weights,
            "use_boundary_main_labels": args.use_boundary_main_labels,
            "use_rule_soft_labels": args.use_rule_soft_labels,
            "boundary_soft_main_weight": args.boundary_soft_main_weight,
            "use_expert_region_matrix": args.use_expert_region_matrix,
            "expert_region_prior_json": str(args.expert_region_prior_json) if args.use_expert_region_matrix else None,
            "region_logit_weight": args.region_logit_weight if args.use_expert_region_matrix else 0.0,
            "freeze_expert_region_matrix": args.freeze_expert_region_matrix,
            "selective_boundary_gate": args.selective_boundary_gate,
            "boundary_gate_margin": args.boundary_gate_margin,
            "region_gate_temperature": args.region_gate_temperature,
            "lambda_boundary": args.lambda_boundary,
            "use_multiscale_attention": args.use_multiscale_attention,
            "multiscale_logit_weight": args.multiscale_logit_weight if args.use_multiscale_attention else 0.0,
            "freeze_encoder": not args.unfreeze_encoder,
        },
        args.out_dir / "best_model.pth",
    )
    summary = {
        "device": str(device),
        "freeze_encoder": not args.unfreeze_encoder,
        "semantic_dim": int(train_semantic.shape[1]),
        "active_train_priors": len(train_evidence),
        "active_val_priors_for_audit": len(val_evidence),
        "lambda_align": args.lambda_align,
        "lambda_pair": args.lambda_pair,
        "lambda_ordinal": args.lambda_ordinal,
        "aux_refine_weight": args.aux_refine_weight,
        "aux_refine_margin": args.aux_refine_margin,
        "selection_target": args.selection_target,
        "primary_output": args.primary_output,
        "boundary_review_csv": str(args.boundary_review_csv) if args.boundary_review_csv else None,
        "expert_rules_csv": str(args.expert_rules_csv) if args.expert_rules_csv else None,
        "aux_pairs_source": args.aux_pairs_source,
        "expert_rule_summary": expert_rule_config["summary"],
        "boundary_review_splits": sorted(allowed_review_splits) if allowed_review_splits is not None else [],
        "use_boundary_pair_weights": args.use_boundary_pair_weights,
        "use_boundary_main_labels": args.use_boundary_main_labels,
        "use_rule_soft_labels": args.use_rule_soft_labels,
        "boundary_soft_main_weight": args.boundary_soft_main_weight,
        "use_expert_region_matrix": args.use_expert_region_matrix,
        "expert_region_prior_json": str(args.expert_region_prior_json) if args.use_expert_region_matrix else None,
        "region_logit_weight": args.region_logit_weight if args.use_expert_region_matrix else 0.0,
        "freeze_expert_region_matrix": args.freeze_expert_region_matrix,
        "selective_boundary_gate": args.selective_boundary_gate,
        "boundary_gate_margin": args.boundary_gate_margin,
        "region_gate_temperature": args.region_gate_temperature,
        "lambda_boundary": args.lambda_boundary,
        "use_multiscale_attention": args.use_multiscale_attention,
        "multiscale_logit_weight": args.multiscale_logit_weight if args.use_multiscale_attention else 0.0,
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "max_test_samples": args.max_test_samples,
        "boundary_main_review_samples": len(main_review_lookup),
        "boundary_pair_review_samples": len(pair_review_lookup),
        "pair_specs": pair_specs,
        "baseline_visual_test": baseline_metrics,
        "final_fusion_test": final_metrics,
        "final_refined_test": final_refined_metrics,
        "primary_test": primary_metrics,
        "test_accuracy_change_vs_visual": final_metrics["accuracy"] - baseline_metrics["accuracy"],
        "test_refined_accuracy_change_vs_visual": final_refined_metrics["accuracy"] - baseline_metrics["accuracy"],
        "primary_accuracy_change_vs_visual": primary_metrics["accuracy"] - baseline_metrics["accuracy"],
        "best_val_macro_f1": best_val_f1,
        "test_gate_mean": test_gate,
        "test_refine_eligible": test_refine["eligible"],
        "test_refine_changed": test_refine["changed"],
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "semantic_vector_columns.json").write_text(
        json.dumps(vectorizer.columns, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
