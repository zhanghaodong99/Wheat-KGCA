from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion.stage2_cam.generate_convnext_gradcam import DEFAULT_CHECKPOINT, load_checkpoint  # noqa: E402
from src.fusion.train_semantic_visual_module import SemanticVectorizer  # noqa: E402
from src.models.convnextv2_tiny import build_model  # noqa: E402


DEFAULT_MANIFEST_DIR = PROJECT_ROOT / "outputs" / "fusion" / "manifests"
DEFAULT_TRAIN_MANIFEST = DEFAULT_MANIFEST_DIR / "fusion_train_800.csv"
DEFAULT_VAL_MANIFEST = DEFAULT_MANIFEST_DIR / "fusion_val_400.csv"
DEFAULT_TEST_MANIFEST = DEFAULT_MANIFEST_DIR / "fusion_test_800.csv"
DEFAULT_TRAIN_SEMANTIC = (
    PROJECT_ROOT
    / "outputs"
    / "fusion"
    / "vlm_semantics"
    / "stage3_20260715_111841"
    / "gpt-5.5"
    / "P7"
    / "gpt-5.5_P7_fusion_train_800"
    / "parsed_results.csv"
)
DEFAULT_VAL_SEMANTIC = (
    PROJECT_ROOT
    / "outputs"
    / "fusion"
    / "vlm_semantics"
    / "stage3_20260715_150354"
    / "gpt-5.5"
    / "P7"
    / "gpt-5.5_P7_fusion_val_400"
    / "parsed_results.csv"
)
DEFAULT_TEST_SEMANTIC = (
    PROJECT_ROOT
    / "outputs"
    / "fusion"
    / "vlm_semantics"
    / "stage3_20260715_173238"
    / "gpt-5.5"
    / "P7"
    / "gpt-5.5_P7_fusion_test_800"
    / "parsed_results.csv"
)
DEFAULT_TRAIN_EVIDENCE = (
    PROJECT_ROOT
    / "outputs"
    / "fusion"
    / "stage2_cam"
    / "contrast_evidence"
    / "stage3_train_contrast_160"
    / "contrast_evidence_results.csv"
)
DEFAULT_VAL_EVIDENCE = (
    PROJECT_ROOT
    / "outputs"
    / "fusion"
    / "stage2_cam"
    / "contrast_evidence"
    / "stage3_val_contrast_80"
    / "contrast_evidence_results.csv"
)

GRID_NAMES = [
    "upper_left",
    "upper_center",
    "upper_right",
    "center_left",
    "center",
    "center_right",
    "lower_left",
    "lower_center",
    "lower_right",
]
GRID_TO_INDEX = {name: index for index, name in enumerate(GRID_NAMES)}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def parse_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def regional_prior(items: list[dict[str, Any]]) -> np.ndarray:
    prior = np.zeros(9, dtype=np.float32)
    for item in items:
        grid = str(item.get("grid", "")).strip()
        if grid not in GRID_TO_INDEX:
            continue
        confidence = min(max(float(item.get("confidence", 0.0)), 0.0), 1.0)
        visibility_weight = 1.0 if str(item.get("visibility", "")).strip() == "yes" else 0.55
        prior[GRID_TO_INDEX[grid]] = max(prior[GRID_TO_INDEX[grid]], confidence * visibility_weight)
    return prior


def build_evidence_lookup(
    evidence_path: Path,
    labels_by_id: dict[str, str],
    class_to_idx: dict[str, int],
) -> dict[str, tuple[np.ndarray, int]]:
    frame = pd.read_csv(evidence_path)
    frame = frame[(frame["status"] == "ok") & (frame["schema_valid"])].copy()
    lookup: dict[str, tuple[np.ndarray, int]] = {}
    for _, row in frame.iterrows():
        image_id = str(row["image_id"])
        candidate_b = str(row["candidate_b"])
        if labels_by_id.get(image_id) != candidate_b or candidate_b not in class_to_idx:
            continue
        prior = regional_prior(parse_items(row["support_b_json"]))
        if prior.sum() > 0:
            lookup[image_id] = (prior, class_to_idx[candidate_b])
    return lookup


def read_semantic(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[(frame["status"] == "ok") & (frame["schema_valid"])].copy()
    if "confidence" in frame.columns:
        frame = frame.rename(columns={"confidence": "gpt_confidence"})
    return frame


def prepare_split(
    manifest_path: Path,
    semantic_path: Path,
    vectorizer: SemanticVectorizer,
    fit_vectorizer: bool,
    class_to_idx: dict[str, int],
) -> tuple[pd.DataFrame, np.ndarray]:
    manifest = pd.read_csv(manifest_path)
    semantic = read_semantic(semantic_path)
    merged = manifest.merge(semantic, on="image_id", how="inner", validate="one_to_one", suffixes=("", "_semantic"))
    if len(merged) != len(manifest):
        missing = len(manifest) - len(merged)
        raise RuntimeError(f"{missing} manifest rows have no valid VLM semantic output: {semantic_path}")
    semantic_vector = vectorizer.fit_transform(merged) if fit_vectorizer else vectorizer.transform(merged)
    missing_labels = set(merged["stage_id"].astype(str)) - set(class_to_idx)
    if missing_labels:
        raise RuntimeError(f"Unknown labels in manifest: {sorted(missing_labels)}")
    return merged, semantic_vector


class FusionImageDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        semantic: np.ndarray,
        class_to_idx: dict[str, int],
        evidence: dict[str, tuple[np.ndarray, int]],
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.semantic = semantic.astype(np.float32)
        self.class_to_idx = class_to_idx
        self.evidence = evidence
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image_id = str(row["image_id"])
        with Image.open(str(row["image_path"])) as image:
            image_tensor = self.transform(image.convert("RGB"))
        prior, target = self.evidence.get(image_id, (np.zeros(9, dtype=np.float32), -1))
        return (
            image_tensor,
            torch.from_numpy(self.semantic[index]),
            torch.tensor(self.class_to_idx[str(row["stage_id"])], dtype=torch.long),
            torch.from_numpy(prior),
            torch.tensor(target, dtype=torch.long),
            torch.tensor(target >= 0, dtype=torch.bool),
            image_id,
        )


class DiscrepancyGatedFusion(nn.Module):
    def __init__(self, visual_model: nn.Module, semantic_dim: int, num_classes: int, hidden_dim: int = 128) -> None:
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
        nn.init.zeros_(self.semantic_delta.weight)
        nn.init.zeros_(self.semantic_delta.bias)
        nn.init.constant_(self.gate[-1].bias, -3.0)

    def forward(self, images: torch.Tensor, semantic: torch.Tensor):
        visual_logits, raw_features = self.visual(images)
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
        gate = torch.sigmoid(self.gate(torch.cat([semantic_token, visual_logits, uncertainty], dim=1)))
        fused_logits = visual_logits + gate * self.semantic_delta(semantic_token)
        return fused_logits, visual_logits, raw_features, gate

    def encode_semantic(self, semantic: torch.Tensor) -> torch.Tensor:
        return self.semantic_encoder(semantic)


def to_nchw(raw_features: torch.Tensor, feature_dim: int) -> torch.Tensor:
    if raw_features.dim() != 4:
        raise ValueError(f"Expected 4D raw feature map, got {tuple(raw_features.shape)}")
    if raw_features.shape[1] == feature_dim:
        return raw_features
    return raw_features.permute(0, 3, 1, 2).contiguous()


def discrepancy_gated_alignment_loss(
    visual_model: nn.Module,
    raw_features: torch.Tensor,
    priors_3x3: torch.Tensor,
    cam_targets: torch.Tensor,
    active: torch.Tensor,
) -> tuple[torch.Tensor, int, int, float]:
    if not active.any():
        return raw_features.new_zeros(()), 0, 0, 0.0
    features = to_nchw(raw_features, visual_model.feature_dim)[active]
    targets = cam_targets[active]
    prior = priors_3x3[active].view(-1, 1, 3, 3)
    weights = visual_model.head.weight[targets].view(-1, visual_model.feature_dim, 1, 1)
    cam = F.relu((features * weights).sum(dim=1, keepdim=True))
    cam_distribution = cam.flatten(1)
    cam_distribution = cam_distribution / cam_distribution.sum(dim=1, keepdim=True).clamp_min(1e-8)
    target_distribution = F.interpolate(prior, size=cam.shape[-2:], mode="bilinear", align_corners=False).flatten(1)
    target_distribution = target_distribution / target_distribution.sum(dim=1, keepdim=True).clamp_min(1e-8)

    # Uniform attention has alignment gain 1.0. Only worse-than-uniform CAM maps are corrected.
    alignment_gain = cam_distribution.shape[1] * (cam_distribution * target_distribution).sum(dim=1)
    mismatch_weight = F.relu(1.0 - alignment_gain.detach())
    active_mismatch = mismatch_weight > 0
    if not active_mismatch.any():
        return raw_features.new_zeros(()), int(active.sum().item()), 0, float(alignment_gain.mean().item())

    cross_entropy = -(target_distribution * cam_distribution.clamp_min(1e-8).log()).sum(dim=1)
    loss = (cross_entropy * mismatch_weight).sum() / mismatch_weight.sum().clamp_min(1e-8)
    return (
        loss,
        int(active.sum().item()),
        int(active_mismatch.sum().item()),
        float(alignment_gain.mean().item()),
    )


def build_semantic_anchor_targets(
    semantic: np.ndarray,
    labels: pd.Series,
    class_to_idx: dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    num_classes = len(class_to_idx)
    label_idx = labels.astype(str).map(class_to_idx).to_numpy(dtype=np.int64)
    anchors = np.zeros((num_classes, semantic.shape[1]), dtype=np.float32)
    for class_index in range(num_classes):
        mask = label_idx == class_index
        if not mask.any():
            raise RuntimeError(f"No training semantic samples for class index={class_index}")
        anchors[class_index] = semantic[mask].mean(axis=0)
    return torch.from_numpy(anchors).float().to(device)


def semantic_anchor_loss(
    model: DiscrepancyGatedFusion,
    semantic: torch.Tensor,
    labels: torch.Tensor,
    anchor_vectors: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    sample_tokens = F.normalize(model.encode_semantic(semantic), dim=1)
    anchor_tokens = F.normalize(model.encode_semantic(anchor_vectors), dim=1)
    logits = sample_tokens @ anchor_tokens.t()
    logits = logits / max(float(temperature), 1e-6)
    return F.cross_entropy(logits, labels)


@torch.no_grad()
def evaluate(model: DiscrepancyGatedFusion, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    labels_all = []
    predictions_all = []
    gates = []
    for images, semantic, labels, _, _, _, _ in loader:
        fused_logits, _, _, gate = model(images.to(device), semantic.to(device))
        labels_all.append(labels.numpy())
        predictions_all.append(fused_logits.argmax(dim=1).cpu().numpy())
        gates.append(gate.cpu().numpy())
    return np.concatenate(labels_all), np.concatenate(predictions_all), float(np.concatenate(gates).mean())


@torch.no_grad()
def evaluate_visual(model: DiscrepancyGatedFusion, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels_all = []
    predictions_all = []
    for images, _, labels, _, _, _, _ in loader:
        logits, _ = model.visual(images.to(device))
        labels_all.append(labels.numpy())
        predictions_all.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(labels_all), np.concatenate(predictions_all)


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train image-level semantic fusion with discrepancy-gated regional CAM alignment."
    )
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
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-align", type=float, default=0.005)
    parser.add_argument("--lambda-sem", type=float, default=0.0)
    parser.add_argument("--semantic-temperature", type=float, default=0.20)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260718)
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
    if not train_evidence:
        raise RuntimeError("No usable train region priors. Do not run without verified support_b evidence.")

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
    model = DiscrepancyGatedFusion(
        visual_model=visual_model,
        semantic_dim=train_semantic.shape[1],
        num_classes=len(class_names),
        hidden_dim=args.hidden_dim,
    ).to(device)
    semantic_anchor_vectors = build_semantic_anchor_targets(
        train_semantic,
        train_frame["stage_id"],
        class_to_idx,
        device,
    )

    baseline_labels, baseline_predictions = evaluate_visual(model, test_loader, device)
    baseline_metrics = metrics(baseline_labels, baseline_predictions)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    best_val_f1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        if not args.unfreeze_encoder:
            model.visual.encoder.eval()
        totals = {
            "loss": 0.0,
            "classification": 0.0,
            "alignment": 0.0,
            "semantic_anchor": 0.0,
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
            fused_logits, _, raw_features, _ = model(images, semantic)
            classification_loss = F.cross_entropy(fused_logits, labels)
            alignment_loss, active_count, mismatch_count, alignment_gain = discrepancy_gated_alignment_loss(
                model.visual, raw_features, priors, targets, active
            )
            if args.lambda_sem > 0:
                sem_loss = semantic_anchor_loss(
                    model,
                    semantic,
                    labels,
                    semantic_anchor_vectors,
                    args.semantic_temperature,
                )
            else:
                sem_loss = fused_logits.new_zeros(())
            loss = classification_loss + args.lambda_align * alignment_loss + args.lambda_sem * sem_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals["loss"] += float(loss.item())
            totals["classification"] += float(classification_loss.item())
            totals["alignment"] += float(alignment_loss.item())
            totals["semantic_anchor"] += float(sem_loss.item())
            totals["active"] += active_count
            totals["mismatch"] += mismatch_count
            totals["gain"] += alignment_gain
            batches += 1

        val_labels_np, val_predictions, val_gate = evaluate(model, val_loader, device)
        val_metrics = metrics(val_labels_np, val_predictions)
        row = {
            "epoch": epoch,
            "loss": totals["loss"] / max(batches, 1),
            "classification_loss": totals["classification"] / max(batches, 1),
            "alignment_loss": totals["alignment"] / max(batches, 1),
            "semantic_anchor_loss": totals["semantic_anchor"] / max(batches, 1),
            "active_region_samples_seen": totals["active"],
            "mismatch_samples_seen": totals["mismatch"],
            "mean_alignment_gain": totals["gain"] / max(batches, 1),
            "val_gate_mean": val_gate,
            **{f"val_{name}": value for name, value in val_metrics.items()},
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} loss={row['loss']:.4f} cls={row['classification_loss']:.4f} "
            f"align={row['alignment_loss']:.4f} sem={row['semantic_anchor_loss']:.4f} "
            f"mismatch={row['mismatch_samples_seen']} "
            f"val_acc={row['val_accuracy']:.4f} val_f1={row['val_macro_f1']:.4f} gate={val_gate:.3f}"
        )
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_labels, test_predictions, test_gate = evaluate(model, test_loader, device)
    final_metrics = metrics(test_labels, test_predictions)
    prediction_frame = test_frame[["image_id", "image_path", "stage_id"]].copy()
    prediction_frame["pred_label"] = [class_names[index] for index in test_predictions]
    prediction_frame["correct"] = prediction_frame["stage_id"] == prediction_frame["pred_label"]
    prediction_frame.to_csv(args.out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(args.out_dir / "history.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        confusion_matrix(
            prediction_frame["stage_id"],
            prediction_frame["pred_label"],
            labels=class_names,
        ),
        index=class_names,
        columns=class_names,
    ).to_csv(args.out_dir / "test_confusion_matrix.csv", encoding="utf-8-sig")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "semantic_columns": vectorizer.columns,
            "lambda_align": args.lambda_align,
            "lambda_sem": args.lambda_sem,
            "semantic_temperature": args.semantic_temperature,
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
        "lambda_sem": args.lambda_sem,
        "semantic_temperature": args.semantic_temperature,
        "baseline_visual_test": baseline_metrics,
        "final_fusion_test": final_metrics,
        "test_accuracy_change_vs_visual": final_metrics["accuracy"] - baseline_metrics["accuracy"],
        "test_macro_f1_change_vs_visual": final_metrics["macro_f1"] - baseline_metrics["macro_f1"],
        "best_val_macro_f1": best_val_f1,
        "test_gate_mean": test_gate,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "semantic_vector_columns.json").write_text(
        json.dumps(vectorizer.columns, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
