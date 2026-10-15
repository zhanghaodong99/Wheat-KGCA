from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion.stage2_cam.generate_convnext_gradcam import load_checkpoint, to_nchw  # noqa: E402
from src.fusion.stage4_discrepancy_gated.train_discrepancy_gated_fusion import prepare_split  # noqa: E402
from src.fusion.stage6_hierarchical_heads.train_hierarchical_aux_heads import HierarchicalAuxFusion  # noqa: E402
from src.fusion.train_semantic_visual_module import SemanticVectorizer  # noqa: E402
from src.models.convnextv2_tiny import build_model  # noqa: E402


DEFAULT_BASELINE_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "src_model_training"
    / "src_single_convnextv2_tiny_20260713_092041"
    / "convnextv2_tiny"
    / "best_model.pth"
)
DEFAULT_H03_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "paper_full_experiments"
    / "hierarchical_03_aux_heads_cam_valref_select_main_output"
    / "best_model.pth"
)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def preprocess(path: Path, input_size: int) -> tuple[torch.Tensor, Image.Image]:
    image = Image.open(path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transform(image).unsqueeze(0), image


def normalize_cam(cam: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=size, mode="bilinear", align_corners=False)
    arr = cam[0, 0].detach().cpu().numpy()
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    return arr


def baseline_cam(model, image_tensor: torch.Tensor, target_idx: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.zero_grad(set_to_none=True)
    image_tensor = image_tensor.to(device)
    logits, raw_features = model(image_tensor)
    raw_features.retain_grad()
    score = logits[:, target_idx].sum()
    score.backward()
    features = to_nchw(raw_features.detach(), model.feature_dim)
    grads = to_nchw(raw_features.grad.detach(), model.feature_dim)
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * features).sum(dim=1, keepdim=True)
    return normalize_cam(cam, image_tensor.shape[-2:]), logits.detach().cpu().numpy()[0]


def h03_cam(
    model: HierarchicalAuxFusion,
    image_tensor: torch.Tensor,
    semantic_tensor: torch.Tensor,
    target_idx: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.zero_grad(set_to_none=True)
    image_tensor = image_tensor.to(device)
    semantic_tensor = semantic_tensor.to(device)
    fused_logits, visual_logits, raw_features, gate, _ = model(image_tensor, semantic_tensor)
    raw_features.retain_grad()
    score = fused_logits[:, target_idx].sum()
    score.backward()
    features = to_nchw(raw_features.detach(), model.visual.feature_dim)
    grads = to_nchw(raw_features.grad.detach(), model.visual.feature_dim)
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * features).sum(dim=1, keepdim=True)
    return normalize_cam(cam, image_tensor.shape[-2:]), fused_logits.detach().cpu().numpy()[0], gate.detach().cpu().numpy()[0]


def overlay_axis(ax, image: Image.Image, cam: np.ndarray | None, title: str) -> None:
    image_resized = image.resize((224, 224))
    ax.imshow(np.asarray(image_resized).astype(np.float32) / 255.0)
    if cam is not None:
        ax.imshow(cam, cmap="jet", alpha=0.42, vmin=0, vmax=1)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def save_triptych(
    image: Image.Image,
    base_cam_arr: np.ndarray,
    kgca_cam_arr: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    overlay_axis(axes[0], image, None, "Original")
    overlay_axis(axes[1], image, base_cam_arr, "ConvNeXtV2 CAM")
    overlay_axis(axes[2], image, kgca_cam_arr, "Wheat-KGCA CAM")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate baseline vs Wheat-KGCA CAM comparison figures.")
    parser.add_argument("--samples", type=Path, default=Path("outputs/paper_full_experiments/wheat_kgca_final/wheat_kgca_selected_samples.csv"))
    parser.add_argument("--semantic-test", type=Path, default=Path("outputs/paper_full_experiments/vlm_semantics/full_test_2485/parsed_results.csv"))
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE_CHECKPOINT)
    parser.add_argument("--h03-checkpoint", type=Path, default=DEFAULT_H03_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/paper_full_experiments/wheat_kgca_final/cam_compare"))
    parser.add_argument("--target", choices=["true", "baseline_pred", "wheat_kgca_pred"], default="true")
    parser.add_argument("--limit", type=int, default=48)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    baseline_ckpt = load_checkpoint(args.baseline_checkpoint, device)
    class_names = [str(v) for v in baseline_ckpt["class_names"]]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    input_size = int(baseline_ckpt.get("input_size", 224))

    baseline_model = build_model(num_classes=len(class_names), pretrained=False)
    baseline_model.load_state_dict(baseline_ckpt["model_state_dict"])
    baseline_model.to(device).eval()

    h03_ckpt = torch.load(args.h03_checkpoint, map_location=device, weights_only=False)
    semantic_columns = [str(v) for v in h03_ckpt["semantic_columns"]]
    pair_specs = h03_ckpt.get("pair_specs", [])
    h03_visual = build_model(num_classes=len(class_names), pretrained=False)
    h03_model = HierarchicalAuxFusion(
        visual_model=h03_visual,
        semantic_dim=len(semantic_columns),
        num_classes=len(class_names),
        hidden_dim=128,
        num_aux_heads=len(pair_specs),
    ).to(device)
    h03_model.load_state_dict(h03_ckpt["model_state_dict"], strict=True)
    h03_model.eval()

    samples = pd.read_csv(args.samples).head(args.limit).copy()
    vectorizer = SemanticVectorizer()
    vectorizer.columns = semantic_columns
    frame, semantic = prepare_split(args.samples, args.semantic_test, vectorizer, False, class_to_idx)
    semantic_by_id = {
        str(row["image_id"]): torch.from_numpy(semantic[idx : idx + 1].astype(np.float32))
        for idx, row in frame.reset_index(drop=True).iterrows()
    }

    rows = []
    for _, row in samples.iterrows():
        image_id = str(row["image_id"])
        image_path = Path(str(row["image_path"]))
        if image_id not in semantic_by_id:
            continue
        image_tensor, image = preprocess(image_path, input_size)
        target_label = str(row["stage_id"] if args.target == "true" else row[args.target])
        if target_label not in class_to_idx:
            continue
        target_idx = class_to_idx[target_label]

        base_arr, base_logits = baseline_cam(baseline_model, image_tensor, target_idx, device)
        kgca_arr, kgca_logits, gate = h03_cam(h03_model, image_tensor, semantic_by_id[image_id], target_idx, device)
        base_pred = class_names[int(np.argmax(base_logits))]
        kgca_pred = class_names[int(np.argmax(kgca_logits))]

        reason = str(row.get("selection_reason", "selected"))
        out_path = args.out_dir / reason / f"{image_id}_cam_compare_{args.target}_{target_label}.png"
        title = f"{image_id} | true={row['stage_id']} | baseline={base_pred} | KGCA={kgca_pred} | target={target_label}"
        save_triptych(image, base_arr, kgca_arr, out_path, title)
        rows.append(
            {
                "image_id": image_id,
                "selection_reason": reason,
                "image_path": str(image_path),
                "true_label": row["stage_id"],
                "target_label": target_label,
                "baseline_pred_from_cam_run": base_pred,
                "wheat_kgca_pred_from_cam_run": kgca_pred,
                "wheat_kgca_gate": float(np.ravel(gate)[0]),
                "cam_compare_path": str(out_path),
            }
        )

    index = pd.DataFrame(rows)
    index.to_csv(args.out_dir / f"cam_compare_index_{args.target}.csv", index=False, encoding="utf-8-sig")
    (args.out_dir / f"cam_compare_config_{args.target}.json").write_text(
        json.dumps(
            {
                "samples": str(args.samples),
                "semantic_test": str(args.semantic_test),
                "baseline_checkpoint": str(args.baseline_checkpoint),
                "h03_checkpoint": str(args.h03_checkpoint),
                "target": args.target,
                "limit": args.limit,
                "generated": len(index),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Generated {len(index)} CAM comparison figures in {args.out_dir}")


if __name__ == "__main__":
    main()
