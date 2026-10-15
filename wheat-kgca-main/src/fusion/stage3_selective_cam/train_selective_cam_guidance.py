from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion.stage2_cam.generate_convnext_gradcam import DEFAULT_CHECKPOINT, load_checkpoint  # noqa: E402
from src.models.convnextv2_tiny import build_model  # noqa: E402


DEFAULT_MANIFEST_DIR = PROJECT_ROOT / "outputs" / "fusion" / "manifests"
DEFAULT_TRAIN_MANIFEST = DEFAULT_MANIFEST_DIR / "fusion_train_800.csv"
DEFAULT_VAL_MANIFEST = DEFAULT_MANIFEST_DIR / "fusion_val_400.csv"
DEFAULT_TEST_MANIFEST = DEFAULT_MANIFEST_DIR / "fusion_test_800.csv"

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
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else "cpu" if value == "auto" else value)


def parse_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def grid_prior(items: list[dict[str, Any]]) -> np.ndarray:
    prior = np.zeros(9, dtype=np.float32)
    for item in items:
        grid = str(item.get("grid", "")).strip()
        if grid not in GRID_TO_INDEX:
            continue
        confidence = min(max(float(item.get("confidence", 0.0)), 0.0), 1.0)
        visibility_weight = 1.0 if str(item.get("visibility", "")) == "yes" else 0.55
        prior[GRID_TO_INDEX[grid]] = max(prior[GRID_TO_INDEX[grid]], confidence * visibility_weight)
    return prior


def build_evidence_lookup(
    path: Path,
    class_to_idx: dict[str, int],
    training_labels: dict[str, str],
) -> dict[str, tuple[np.ndarray, int]]:
    evidence = pd.read_csv(path)
    evidence = evidence[(evidence["status"] == "ok") & (evidence["schema_valid"])].copy()
    lookup: dict[str, tuple[np.ndarray, int]] = {}
    for _, row in evidence.iterrows():
        image_id = str(row["image_id"])
        candidate_b = str(row["candidate_b"])
        if candidate_b not in class_to_idx or candidate_b != training_labels.get(image_id):
            continue
        prior = grid_prior(parse_items(row["support_b_json"]))
        if prior.sum() <= 0:
            continue
        lookup[image_id] = (prior, class_to_idx[candidate_b])
    return lookup


@dataclass
class Sample:
    image_path: str
    label: int
    image_id: str
    prior: np.ndarray
    cam_target: int
    cam_active: bool


class WheatDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, class_to_idx: dict[str, int], evidence: dict[str, tuple[np.ndarray, int]]):
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.samples: list[Sample] = []
        for _, row in manifest.iterrows():
            image_id = str(row["image_id"])
            prior, target = evidence.get(image_id, (np.zeros(9, dtype=np.float32), -1))
            self.samples.append(
                Sample(
                    image_path=str(row["image_path"]),
                    label=class_to_idx[str(row["stage_id"])],
                    image_id=image_id,
                    prior=prior,
                    cam_target=target,
                    cam_active=target >= 0,
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return (
            tensor,
            torch.tensor(sample.label, dtype=torch.long),
            torch.from_numpy(sample.prior),
            torch.tensor(sample.cam_target, dtype=torch.long),
            torch.tensor(sample.cam_active, dtype=torch.bool),
            sample.image_id,
        )


def to_nchw(raw_features: torch.Tensor, feature_dim: int) -> torch.Tensor:
    if raw_features.dim() != 4:
        raise ValueError(f"Expected 4D feature map, got {tuple(raw_features.shape)}")
    return raw_features if raw_features.shape[1] == feature_dim else raw_features.permute(0, 3, 1, 2).contiguous()


def selective_cam_loss(
    model: nn.Module,
    raw_features: torch.Tensor,
    priors_3x3: torch.Tensor,
    cam_targets: torch.Tensor,
    active: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    if not active.any():
        return raw_features.new_zeros(()), 0
    features = to_nchw(raw_features, model.feature_dim)
    chosen_features = features[active]
    chosen_targets = cam_targets[active]
    chosen_prior = priors_3x3[active].view(-1, 1, 3, 3)
    class_weights = model.head.weight[chosen_targets].view(-1, model.feature_dim, 1, 1)
    cam = F.relu((chosen_features * class_weights).sum(dim=1, keepdim=True))
    cam_distribution = cam.flatten(1)
    cam_distribution = cam_distribution / cam_distribution.sum(dim=1, keepdim=True).clamp_min(1e-8)
    prior = F.interpolate(chosen_prior, size=cam.shape[-2:], mode="bilinear", align_corners=False).flatten(1)
    prior = prior / prior.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return -(prior * cam_distribution.clamp_min(1e-8).log()).sum(dim=1).mean(), int(active.sum().item())


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_labels = []
    all_preds = []
    for images, labels, _, _, _, _ in loader:
        logits, _ = model(images.to(device))
        all_labels.append(labels.numpy())
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(all_labels), np.concatenate(all_preds)


def metric_dict(labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune ConvNeXt with a selective, label-reliable CAM prior.")
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--val-manifest", type=Path, default=DEFAULT_VAL_MANIFEST)
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_TEST_MANIFEST)
    parser.add_argument("--train-evidence", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-cam", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint(args.checkpoint, device)
    class_names = [str(value) for value in checkpoint["class_names"]]
    class_to_idx = {name: index for index, name in enumerate(class_names)}

    train_manifest = pd.read_csv(args.train_manifest)
    val_manifest = pd.read_csv(args.val_manifest)
    test_manifest = pd.read_csv(args.test_manifest)
    training_labels = {
        str(row["image_id"]): str(row["stage_id"])
        for _, row in train_manifest.iterrows()
    }
    evidence = build_evidence_lookup(args.train_evidence, class_to_idx, training_labels)
    if not evidence:
        raise RuntimeError("No usable support_b evidence matched training labels. Do not train without active CAM priors.")

    train_dataset = WheatDataset(train_manifest, class_to_idx, evidence)
    val_dataset = WheatDataset(val_manifest, class_to_idx, {})
    test_dataset = WheatDataset(test_manifest, class_to_idx, {})
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    baseline_labels, baseline_preds = evaluate(model, test_loader, device)
    baseline_metrics = metric_dict(baseline_labels, baseline_preds)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_val_f1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_cam = 0.0
        active_count = 0
        batches = 0
        for images, labels, priors, targets, active, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            priors = priors.to(device)
            targets = targets.to(device)
            active = active.to(device)
            logits, raw_features = model(images)
            ce = F.cross_entropy(logits, labels)
            cam_loss, active_in_batch = selective_cam_loss(model, raw_features, priors, targets, active)
            loss = ce + args.lambda_cam * cam_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            total_ce += float(ce.item())
            total_cam += float(cam_loss.item())
            active_count += active_in_batch
            batches += 1

        val_labels, val_preds = evaluate(model, val_loader, device)
        val_metrics = metric_dict(val_labels, val_preds)
        row = {
            "epoch": epoch,
            "loss": total_loss / max(batches, 1),
            "classification_loss": total_ce / max(batches, 1),
            "cam_loss": total_cam / max(batches, 1),
            "active_cam_samples_seen": active_count,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} loss={row['loss']:.4f} ce={row['classification_loss']:.4f} "
            f"cam={row['cam_loss']:.4f} val_acc={row['val_accuracy']:.4f} val_f1={row['val_macro_f1']:.4f}"
        )
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_labels, test_preds = evaluate(model, test_loader, device)
    test_metrics = metric_dict(test_labels, test_preds)
    test_labels_text = [class_names[index] for index in test_labels]
    test_preds_text = [class_names[index] for index in test_preds]
    predictions = test_manifest[["image_id", "image_path", "stage_id"]].copy()
    predictions["pred_label"] = test_preds_text
    predictions["correct"] = predictions["stage_id"] == predictions["pred_label"]
    predictions.to_csv(args.out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(args.out_dir / "history.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        confusion_matrix(test_labels_text, test_preds_text, labels=class_names),
        index=class_names,
        columns=class_names,
    ).to_csv(args.out_dir / "test_confusion_matrix.csv", encoding="utf-8-sig")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "input_size": 224,
            "lambda_cam": args.lambda_cam,
            "active_training_samples": len(evidence),
        },
        args.out_dir / "best_model.pth",
    )
    summary = {
        "device": str(device),
        "lambda_cam": args.lambda_cam,
        "active_training_samples": len(evidence),
        "baseline_test": baseline_metrics,
        "final_test": test_metrics,
        "test_accuracy_change": test_metrics["accuracy"] - baseline_metrics["accuracy"],
        "test_macro_f1_change": test_metrics["macro_f1"] - baseline_metrics["macro_f1"],
        "best_val_macro_f1": best_val_f1,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
