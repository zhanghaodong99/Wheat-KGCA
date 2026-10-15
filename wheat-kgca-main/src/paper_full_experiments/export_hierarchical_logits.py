from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion.stage4_discrepancy_gated.train_discrepancy_gated_fusion import FusionImageDataset, prepare_split  # noqa: E402
from src.fusion.stage6_hierarchical_heads.train_hierarchical_aux_heads import HierarchicalAuxFusion  # noqa: E402
from src.fusion.train_semantic_visual_module import SemanticVectorizer  # noqa: E402
from src.models.convnextv2_tiny import build_model  # noqa: E402


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


@torch.no_grad()
def collect_logits(model: HierarchicalAuxFusion, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    fused_logits_all = []
    visual_logits_all = []
    for images, semantic, _, _, _, _, _ in loader:
        fused_logits, visual_logits, _, _, _ = model(images.to(device), semantic.to(device))
        fused_logits_all.append(fused_logits.cpu().numpy())
        visual_logits_all.append(visual_logits.cpu().numpy())
    return np.concatenate(fused_logits_all), np.concatenate(visual_logits_all)


def export_split(
    split_name: str,
    manifest: Path,
    semantic_csv: Path,
    model: HierarchicalAuxFusion,
    vectorizer: SemanticVectorizer,
    class_to_idx: dict[str, int],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    out_dir: Path,
) -> None:
    frame, semantic = prepare_split(manifest, semantic_csv, vectorizer, False, class_to_idx)
    dataset = FusionImageDataset(frame, semantic, class_to_idx, {})
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    fused_logits, visual_logits = collect_logits(model, loader, device)
    labels = frame["stage_id"].map(class_to_idx).to_numpy(dtype=np.int64)
    meta = frame[["image_id", "image_path", "stage_id", "source_type"]].copy()
    meta.to_csv(out_dir / f"{split_name}_meta.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        out_dir / f"{split_name}_logits.npz",
        fused_logits=fused_logits.astype(np.float32),
        visual_logits=visual_logits.astype(np.float32),
        labels=labels,
        image_id=frame["image_id"].astype(str).to_numpy(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export split logits from a hierarchical fusion checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, default=None)
    parser.add_argument("--val-manifest", type=Path, default=None)
    parser.add_argument("--test-manifest", type=Path, default=None)
    parser.add_argument("--semantic-train", type=Path, default=None)
    parser.add_argument("--semantic-val", type=Path, default=None)
    parser.add_argument("--semantic-test", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    requested = [
        (args.train_manifest, args.semantic_train),
        (args.val_manifest, args.semantic_val),
        (args.test_manifest, args.semantic_test),
    ]
    if not any(manifest is not None and semantic is not None for manifest, semantic in requested):
        raise RuntimeError("At least one split manifest/semantic pair must be provided.")
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    class_names = [str(value) for value in checkpoint["class_names"]]
    class_to_idx = {name: index for index, name in enumerate(class_names)}
    semantic_columns = [str(value) for value in checkpoint["semantic_columns"]]
    pair_specs = checkpoint.get("pair_specs", [])

    visual_model = build_model(num_classes=len(class_names), pretrained=False)
    model = HierarchicalAuxFusion(
        visual_model=visual_model,
        semantic_dim=len(semantic_columns),
        num_classes=len(class_names),
        hidden_dim=128,
        num_aux_heads=len(pair_specs),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    vectorizer = SemanticVectorizer()
    vectorizer.columns = semantic_columns
    for split_name, manifest, semantic_csv in [
        ("train", args.train_manifest, args.semantic_train),
        ("val", args.val_manifest, args.semantic_val),
        ("test", args.test_manifest, args.semantic_test),
    ]:
        if manifest is None or semantic_csv is None:
            continue
        export_split(
            split_name,
            manifest,
            semantic_csv,
            model,
            vectorizer,
            class_to_idx,
            args.batch_size,
            args.num_workers,
            device,
            args.out_dir,
        )
    (args.out_dir / "export_config.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "class_names": class_names,
                "semantic_columns": semantic_columns,
                "pair_specs": pair_specs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(args.out_dir), "class_names": class_names}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
