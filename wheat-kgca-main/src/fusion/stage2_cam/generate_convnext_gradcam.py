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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
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
DEFAULT_SAMPLES = PROJECT_ROOT / "outputs" / "fusion" / "stage2_cam" / "stage2_cam_selected_samples.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "fusion" / "stage2_cam" / "gradcam"


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def preprocess_image(path: Path, input_size: int) -> tuple[torch.Tensor, Image.Image]:
    image = Image.open(path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transform(image).unsqueeze(0), image


def to_nchw(feat: torch.Tensor, feature_dim: int) -> torch.Tensor:
    if feat.dim() != 4:
        raise ValueError(f"Grad-CAM expects 4D features, got shape={tuple(feat.shape)}")
    if feat.shape[1] != feature_dim:
        return feat.permute(0, 3, 1, 2).contiguous()
    return feat


def compute_cam(model, image_tensor: torch.Tensor, target_idx: int, device: torch.device) -> tuple[
    np.ndarray, np.ndarray]:
    model.zero_grad(set_to_none=True)
    image_tensor = image_tensor.to(device)
    logits, raw_feat = model(image_tensor)
    raw_feat.retain_grad()
    score = logits[:, target_idx].sum()
    score.backward()

    feat = to_nchw(raw_feat.detach(), model.feature_dim)
    grad = to_nchw(raw_feat.grad.detach(), model.feature_dim)
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * feat).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
    cam_np = cam[0, 0].detach().cpu().numpy()
    cam_np = cam_np - cam_np.min()
    if cam_np.max() > 0:
        cam_np = cam_np / cam_np.max()
    return cam_np, logits.detach().cpu().numpy()[0]


def save_overlay(image: Image.Image, cam: np.ndarray, out_path: Path, title: str) -> None:
    image_resized = image.resize((cam.shape[1], cam.shape[0]))
    image_np = np.asarray(image_resized).astype(np.float32) / 255.0
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image_np)
    ax.imshow(cam, cmap="jet", alpha=0.42, vmin=0, vmax=1)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ConvNeXt Grad-CAM overlays for Stage 2 selected samples.")
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--target", type=str, default="pred", choices=["pred", "true"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    class_names = checkpoint.get("class_names")
    if not class_names:
        raise RuntimeError("Checkpoint does not contain class_names.")
    input_size = int(checkpoint.get("input_size", 224))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    model = build_model(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    samples = pd.read_csv(args.samples)
    if args.limit > 0:
        samples = samples.head(args.limit).copy()

    rows = []
    for _, row in samples.iterrows():
        image_path = Path(str(row["image_path"]))
        image_tensor, image = preprocess_image(image_path, input_size)
        target_label = str(row["conv_pred"] if args.target == "pred" else row["true_label"])
        target_idx = class_to_idx[target_label]
        cam, logits = compute_cam(model, image_tensor, target_idx, device)
        pred_idx = int(np.argmax(logits))
        pred_label = str(class_names[pred_idx])
        image_id = str(row["image_id"])
        reason = str(row.get("selection_reason", "selected"))
        out_path = args.out_dir / reason / f"{image_id}_cam_{args.target}_{target_label}.png"
        title = f"{image_id} | true={row['true_label']} | conv={row['conv_pred']} | target={target_label}"
        save_overlay(image, cam, out_path, title)
        rows.append(
            {
                "image_id": image_id,
                "selection_reason": reason,
                "image_path": str(image_path),
                "true_label": row["true_label"],
                "conv_pred": row["conv_pred"],
                "target": args.target,
                "target_label": target_label,
                "model_pred_label": pred_label,
                "cam_path": str(out_path),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_dir / f"gradcam_index_{args.target}.csv", index=False, encoding="utf-8-sig")
    (args.out_dir / f"gradcam_config_{args.target}.json").write_text(
        json.dumps(
            {
                "samples": str(args.samples),
                "checkpoint": str(args.checkpoint),
                "target": args.target,
                "limit": args.limit,
                "device": str(device),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Generated {len(rows)} Grad-CAM overlays.")
    print(f"Output: {args.out_dir}")


if __name__ == "__main__":
    main()
