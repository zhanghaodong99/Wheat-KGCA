from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import torch
from PIL import Image
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
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "fusion" / "convnext_features"


class ManifestImageDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, class_to_idx: dict[str, int], input_size: int):
        self.manifest = manifest.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int):
        row = self.manifest.iloc[index]
        image_path = Path(str(row["image_path"]))
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        label = self.class_to_idx[str(row["stage_id"])]
        return tensor, label, index


def image_key(path_value: str) -> str:
    return PureWindowsPath(str(path_value)).name.lower()


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def extract_features(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = load_checkpoint(args.checkpoint, device)
    class_names = checkpoint.get("class_names") or sorted(pd.read_csv(args.manifest)["stage_id"].unique().tolist())
    input_size = int(checkpoint.get("input_size", 224))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    manifest = pd.read_csv(args.manifest)
    if args.limit > 0:
        manifest = manifest.head(args.limit).copy()
    missing = sorted(set(manifest["stage_id"].astype(str)) - set(class_to_idx))
    if missing:
        raise ValueError(f"Manifest contains labels missing from checkpoint class_names: {missing}")

    dataset = ManifestImageDataset(manifest, class_to_idx=class_to_idx, input_size=input_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    all_features = []
    all_logits = []
    all_labels = []
    all_indices = []

    with torch.no_grad():
        for images, labels, indices in loader:
            images = images.to(device)
            logits, raw_feat = model(images)
            features = model._pool_features(raw_feat)
            all_features.append(features.detach().cpu().numpy())
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labels.numpy())
            all_indices.append(indices.numpy())

    features_np = np.concatenate(all_features, axis=0)
    logits_np = np.concatenate(all_logits, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    indices_np = np.concatenate(all_indices, axis=0)
    probs_np = torch.softmax(torch.from_numpy(logits_np), dim=1).numpy()
    pred_idx = probs_np.argmax(axis=1)

    out_name = args.manifest.stem if args.limit <= 0 else f"{args.manifest.stem}_limit{args.limit}"
    out_dir = args.out_dir / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    ordered_manifest = manifest.iloc[indices_np].reset_index(drop=True).copy()
    ordered_manifest["image_key"] = ordered_manifest["image_path"].map(image_key)
    ordered_manifest["true_idx"] = labels_np
    ordered_manifest["pred_idx"] = pred_idx
    ordered_manifest["pred_label"] = [class_names[i] for i in pred_idx]
    ordered_manifest["confidence"] = probs_np.max(axis=1)
    ordered_manifest["correct"] = ordered_manifest["stage_id"].astype(str).to_numpy() == ordered_manifest["pred_label"].astype(str).to_numpy()

    pred_cols = [
        "image_id",
        "image_key",
        "split",
        "stage_id",
        "source_type",
        "image_path",
        "true_idx",
        "pred_idx",
        "pred_label",
        "confidence",
        "correct",
    ]
    ordered_manifest[[c for c in pred_cols if c in ordered_manifest.columns]].to_csv(
        out_dir / "convnext_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    np.savez_compressed(
        out_dir / "convnext_features.npz",
        features=features_np,
        logits=logits_np,
        probs=probs_np,
        labels=labels_np,
        image_id=ordered_manifest["image_id"].astype(str).to_numpy(),
        image_key=ordered_manifest["image_key"].astype(str).to_numpy(),
        class_names=np.array(class_names, dtype=object),
    )

    metrics = {
        "manifest": str(args.manifest.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "rows": int(len(ordered_manifest)),
        "feature_dim": int(features_np.shape[1]),
        "accuracy": float(ordered_manifest["correct"].mean()),
        "device": str(device),
    }
    (out_dir / "convnext_feature_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Output: {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract ConvNeXt logits and pooled features for fusion manifests.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extract_features(args)


if __name__ == "__main__":
    main()
