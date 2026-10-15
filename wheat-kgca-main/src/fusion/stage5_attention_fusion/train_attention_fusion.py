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
    build_evidence_lookup,
    discrepancy_gated_alignment_loss,
    prepare_split,
    to_nchw,
)
from src.fusion.train_semantic_visual_module import SemanticVectorizer  # noqa: E402
from src.models.convnextv2_tiny import build_model  # noqa: E402


DEFAULT_TRAIN_SEMANTIC = (
    PROJECT_ROOT / "outputs" / "paper_full_experiments" / "vlm_semantics" / "full_train_19877" / "parsed_results.csv"
)
DEFAULT_VAL_SEMANTIC = (
    PROJECT_ROOT / "outputs" / "paper_full_experiments" / "vlm_semantics" / "full_val_2485" / "parsed_results.csv"
)
DEFAULT_TEST_SEMANTIC = (
    PROJECT_ROOT / "outputs" / "paper_full_experiments" / "vlm_semantics" / "full_test_2485" / "parsed_results.csv"
)
DEFAULT_TRAIN_EVIDENCE = (
    PROJECT_ROOT / "outputs" / "paper_full_experiments" / "expert_spatial_priors" / "expert_spatial_prior_train.csv"
)
DEFAULT_VAL_EVIDENCE = (
    PROJECT_ROOT / "outputs" / "paper_full_experiments" / "expert_spatial_priors" / "expert_spatial_prior_val.csv"
)


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


def prior_to_distribution(priors_3x3: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    prior = F.interpolate(priors_3x3.view(-1, 1, 3, 3), size=size, mode="bilinear", align_corners=False)
    prior = prior.flatten(1)
    return prior / prior.sum(dim=1, keepdim=True).clamp_min(1e-8)


class CamGuidedCrossAttentionFusion(nn.Module):
    def __init__(
        self,
        visual_model: nn.Module,
        semantic_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        attention_dim: int = 128,
        num_heads: int = 4,
        use_prior_bias: bool = False,
        prior_bias_strength: float = 1.0,
    ) -> None:
        super().__init__()
        if attention_dim % num_heads != 0:
            raise ValueError("attention_dim must be divisible by num_heads")
        self.visual = visual_model
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.use_prior_bias = use_prior_bias
        self.prior_bias_strength = prior_bias_strength

        self.semantic_encoder = nn.Sequential(
            nn.Linear(semantic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.visual_proj = nn.Linear(visual_model.feature_dim, attention_dim)
        self.query_proj = nn.Linear(hidden_dim, attention_dim)
        self.key_proj = nn.Linear(attention_dim, attention_dim)
        self.value_proj = nn.Linear(attention_dim, attention_dim)
        self.context_proj = nn.Sequential(
            nn.LayerNorm(attention_dim),
            nn.Linear(attention_dim, hidden_dim),
            nn.GELU(),
        )
        self.attention_delta = nn.Linear(hidden_dim, num_classes)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + num_classes + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.attention_delta.weight)
        nn.init.zeros_(self.attention_delta.bias)
        nn.init.constant_(self.gate[-1].bias, -3.0)

    def forward(
        self,
        images: torch.Tensor,
        semantic: torch.Tensor,
        priors_3x3: torch.Tensor | None = None,
        prior_active: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
        visual_logits, raw_features = self.visual(images)
        features = to_nchw(raw_features, self.visual.feature_dim)
        batch_size, channels, height, width = features.shape
        visual_tokens = features.flatten(2).transpose(1, 2)
        visual_tokens = self.visual_proj(visual_tokens)

        semantic_token = self.semantic_encoder(semantic)
        query = self.query_proj(semantic_token).view(batch_size, self.num_heads, 1, self.head_dim)
        key = self.key_proj(visual_tokens).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_proj(visual_tokens).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.use_prior_bias and priors_3x3 is not None and prior_active is not None:
            prior_distribution = prior_to_distribution(priors_3x3, (height, width)).view(batch_size, 1, 1, height * width)
            prior_scale = prior_distribution * float(height * width)
            active = prior_active.view(batch_size, 1, 1, 1).to(dtype=scores.dtype)
            scores = scores + active * self.prior_bias_strength * prior_scale

        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, value).transpose(1, 2).contiguous().view(batch_size, 1, self.attention_dim)
        context_token = self.context_proj(context.squeeze(1))

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
        gate_input = torch.cat([semantic_token, context_token, visual_logits, uncertainty], dim=1)
        gate = torch.sigmoid(self.gate(gate_input))
        fused_logits = visual_logits + gate * self.attention_delta(context_token)
        return fused_logits, visual_logits, raw_features, gate, attention.squeeze(2), (height, width)


def attention_prior_loss(
    attention: torch.Tensor,
    priors_3x3: torch.Tensor,
    active: torch.Tensor,
    feature_hw: tuple[int, int],
) -> torch.Tensor:
    if not active.any():
        return attention.new_zeros(())
    attention_distribution = attention.mean(dim=1)[active]
    target_distribution = prior_to_distribution(priors_3x3[active], feature_hw)
    loss = -(target_distribution * attention_distribution.clamp_min(1e-8).log()).sum(dim=1)
    return loss.mean()


@torch.no_grad()
def evaluate(
    model: CamGuidedCrossAttentionFusion,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    model.eval()
    labels_all = []
    predictions_all = []
    gates = []
    entropies = []
    for images, semantic, labels, priors, _, active, _ in loader:
        fused_logits, _, _, gate, attention, _ = model(
            images.to(device),
            semantic.to(device),
            priors.to(device),
            active.to(device),
        )
        labels_all.append(labels.numpy())
        predictions_all.append(fused_logits.argmax(dim=1).cpu().numpy())
        gates.append(gate.cpu().numpy())
        mean_attention = attention.mean(dim=1)
        entropy = -(mean_attention * mean_attention.clamp_min(1e-8).log()).sum(dim=1)
        entropies.append(entropy.cpu().numpy())
    return (
        np.concatenate(labels_all),
        np.concatenate(predictions_all),
        float(np.concatenate(gates).mean()),
        float(np.concatenate(entropies).mean()),
    )


@torch.no_grad()
def evaluate_visual(
    model: CamGuidedCrossAttentionFusion,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels_all = []
    predictions_all = []
    for images, _, labels, _, _, _, _ in loader:
        logits, _ = model.visual(images.to(device))
        labels_all.append(labels.numpy())
        predictions_all.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(labels_all), np.concatenate(predictions_all)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CAM-guided cross-attention semantic fusion.")
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
    parser.add_argument("--lambda-align", type=float, default=0.0)
    parser.add_argument("--lambda-attn", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--attention-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--use-prior-bias", action="store_true")
    parser.add_argument("--prior-bias-strength", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--unfreeze-encoder", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(args.checkpoint, device)
    class_names = [str(value) for value in checkpoint["class_names"]]
    class_to_idx = {name: index for index, name in enumerate(class_names)}

    vectorizer = SemanticVectorizer()
    train_frame, train_semantic = prepare_split(
        args.train_manifest, args.semantic_train, vectorizer, True, class_to_idx
    )
    val_frame, val_semantic = prepare_split(args.val_manifest, args.semantic_val, vectorizer, False, class_to_idx)
    test_frame, test_semantic = prepare_split(args.test_manifest, args.semantic_test, vectorizer, False, class_to_idx)

    train_labels = {str(row["image_id"]): str(row["stage_id"]) for _, row in train_frame.iterrows()}
    val_labels = {str(row["image_id"]): str(row["stage_id"]) for _, row in val_frame.iterrows()}
    train_evidence = build_evidence_lookup(args.train_evidence, train_labels, class_to_idx)
    val_evidence = build_evidence_lookup(args.val_evidence, val_labels, class_to_idx)
    needs_priors = args.lambda_align > 0 or args.lambda_attn > 0 or args.use_prior_bias
    if needs_priors and not train_evidence:
        raise RuntimeError("This attention setting needs train region priors, but none were loaded.")

    train_dataset = FusionImageDataset(train_frame, train_semantic, class_to_idx, train_evidence)
    val_dataset = FusionImageDataset(val_frame, val_semantic, class_to_idx, {})
    test_dataset = FusionImageDataset(test_frame, test_semantic, class_to_idx, {})
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    visual_model = build_model(num_classes=len(class_names), pretrained=False)
    visual_model.load_state_dict(checkpoint["model_state_dict"])
    if not args.unfreeze_encoder:
        visual_model.encoder.requires_grad_(False)
    model = CamGuidedCrossAttentionFusion(
        visual_model=visual_model,
        semantic_dim=train_semantic.shape[1],
        num_classes=len(class_names),
        hidden_dim=args.hidden_dim,
        attention_dim=args.attention_dim,
        num_heads=args.num_heads,
        use_prior_bias=args.use_prior_bias,
        prior_bias_strength=args.prior_bias_strength,
    ).to(device)

    baseline_labels, baseline_predictions = evaluate_visual(model, test_loader, device)
    baseline_metrics = metrics(baseline_labels, baseline_predictions)
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
        totals = {
            "loss": 0.0,
            "classification": 0.0,
            "alignment": 0.0,
            "attention_prior": 0.0,
            "active": 0,
            "mismatch": 0,
            "gain": 0.0,
        }
        batches = 0
        for images, semantic, labels, priors, targets, active, _ in train_loader:
            images = images.to(device)
            semantic = semantic.to(device)
            labels = labels.to(device)
            priors = priors.to(device)
            targets = targets.to(device)
            active = active.to(device)

            fused_logits, _, raw_features, _, attention, feature_hw = model(images, semantic, priors, active)
            classification_loss = F.cross_entropy(fused_logits, labels)
            alignment_loss, active_count, mismatch_count, alignment_gain = discrepancy_gated_alignment_loss(
                model.visual, raw_features, priors, targets, active
            )
            attn_loss = attention_prior_loss(attention, priors, active, feature_hw)
            loss = classification_loss + args.lambda_align * alignment_loss + args.lambda_attn * attn_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            totals["loss"] += float(loss.item())
            totals["classification"] += float(classification_loss.item())
            totals["alignment"] += float(alignment_loss.item())
            totals["attention_prior"] += float(attn_loss.item())
            totals["active"] += active_count
            totals["mismatch"] += mismatch_count
            totals["gain"] += alignment_gain
            batches += 1

        val_labels_np, val_predictions, val_gate, val_attention_entropy = evaluate(model, val_loader, device)
        val_metrics = metrics(val_labels_np, val_predictions)
        row = {
            "epoch": epoch,
            "loss": totals["loss"] / max(batches, 1),
            "classification_loss": totals["classification"] / max(batches, 1),
            "alignment_loss": totals["alignment"] / max(batches, 1),
            "attention_prior_loss": totals["attention_prior"] / max(batches, 1),
            "active_region_samples_seen": totals["active"],
            "mismatch_samples_seen": totals["mismatch"],
            "mean_alignment_gain": totals["gain"] / max(batches, 1),
            "val_gate_mean": val_gate,
            "val_attention_entropy": val_attention_entropy,
            **{f"val_{name}": value for name, value in val_metrics.items()},
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} loss={row['loss']:.4f} cls={row['classification_loss']:.4f} "
            f"align={row['alignment_loss']:.4f} attn={row['attention_prior_loss']:.4f} "
            f"mismatch={row['mismatch_samples_seen']} val_acc={row['val_accuracy']:.4f} "
            f"val_f1={row['val_macro_f1']:.4f} gate={val_gate:.3f} attn_entropy={val_attention_entropy:.3f}"
        )
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_labels, test_predictions, test_gate, test_attention_entropy = evaluate(model, test_loader, device)
    final_metrics = metrics(test_labels, test_predictions)
    prediction_frame = test_frame[["image_id", "image_path", "stage_id"]].copy()
    prediction_frame["pred_label"] = [class_names[index] for index in test_predictions]
    prediction_frame["correct"] = prediction_frame["stage_id"] == prediction_frame["pred_label"]
    prediction_frame.to_csv(args.out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(args.out_dir / "history.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        confusion_matrix(prediction_frame["stage_id"], prediction_frame["pred_label"], labels=class_names),
        index=class_names,
        columns=class_names,
    ).to_csv(args.out_dir / "test_confusion_matrix.csv", encoding="utf-8-sig")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "semantic_columns": vectorizer.columns,
            "lambda_align": args.lambda_align,
            "lambda_attn": args.lambda_attn,
            "use_prior_bias": args.use_prior_bias,
            "prior_bias_strength": args.prior_bias_strength,
            "freeze_encoder": not args.unfreeze_encoder,
            "active_train_priors": len(train_evidence),
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
        "lambda_attn": args.lambda_attn,
        "use_prior_bias": args.use_prior_bias,
        "prior_bias_strength": args.prior_bias_strength,
        "attention_dim": args.attention_dim,
        "num_heads": args.num_heads,
        "baseline_visual_test": baseline_metrics,
        "final_fusion_test": final_metrics,
        "test_accuracy_change_vs_visual": final_metrics["accuracy"] - baseline_metrics["accuracy"],
        "test_macro_f1_change_vs_visual": final_metrics["macro_f1"] - baseline_metrics["macro_f1"],
        "best_val_macro_f1": best_val_f1,
        "test_gate_mean": test_gate,
        "test_attention_entropy": test_attention_entropy,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "semantic_vector_columns.json").write_text(
        json.dumps(vectorizer.columns, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
