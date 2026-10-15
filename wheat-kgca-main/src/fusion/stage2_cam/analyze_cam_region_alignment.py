from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion.stage2_cam.generate_convnext_gradcam import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    compute_cam,
    load_checkpoint,
    preprocess_image,
    resolve_device,
)
from src.models.convnextv2_tiny import build_model  # noqa: E402

DEFAULT_SAMPLES = PROJECT_ROOT / "outputs" / "fusion" / "stage2_cam" / "stage2_cam_selected_samples.csv"
DEFAULT_EVIDENCE = (
    PROJECT_ROOT
    / "outputs"
    / "fusion"
    / "stage2_cam"
    / "region_evidence"
    / "stage2_region_evidence_82"
    / "region_evidence_results.csv"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "fusion" / "stage2_cam" / "cam_region_alignment"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2.3: quantify agreement between VLM 3x3 regional evidence and ConvNeXt Grad-CAM."
    )
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def parse_evidence(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def build_grid_priors(evidence: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Return positive-evidence and background-noise weights over the 3x3 grid."""
    positive = np.zeros(9, dtype=np.float64)
    noise = np.zeros(9, dtype=np.float64)
    for item in evidence:
        grid = str(item.get("grid", "")).strip()
        if grid not in GRID_TO_INDEX:
            continue
        confidence = float(item.get("confidence", 0.0))
        confidence = min(max(confidence, 0.0), 1.0)
        visibility = str(item.get("visibility", "")).strip()
        weight = confidence * (1.0 if visibility == "yes" else 0.55)
        if str(item.get("organ", "")).strip() == "background_noise":
            noise[GRID_TO_INDEX[grid]] = max(noise[GRID_TO_INDEX[grid]], weight)
        else:
            positive[GRID_TO_INDEX[grid]] = max(positive[GRID_TO_INDEX[grid]], weight)
    return positive, noise


def cam_grid_distribution(cam: np.ndarray) -> np.ndarray:
    height, width = cam.shape
    rows = np.array_split(np.arange(height), 3)
    cols = np.array_split(np.arange(width), 3)
    scores = []
    for row_indices in rows:
        for col_indices in cols:
            scores.append(float(cam[np.ix_(row_indices, col_indices)].mean()))
    values = np.asarray(scores, dtype=np.float64)
    return values / values.sum() if values.sum() > 0 else np.full(9, 1 / 9, dtype=np.float64)


def alignment_metrics(cam_grid: np.ndarray, positive: np.ndarray, noise: np.ndarray) -> dict[str, float]:
    positive_sum = float(positive.sum())
    positive_mask = positive > 0
    noise_mask = noise > 0
    if positive_sum > 0:
        prior = positive / positive_sum
        # A uniform CAM has score 1.0. Values above 1 indicate concentration on evidence regions.
        alignment_gain = float(9.0 * np.dot(cam_grid, prior))
        cosine = float(np.dot(cam_grid, prior) / (np.linalg.norm(cam_grid) * np.linalg.norm(prior) + 1e-12))
        evidence_coverage = float(cam_grid[positive_mask].sum())
        uniform_coverage = float(positive_mask.mean())
    else:
        alignment_gain = np.nan
        cosine = np.nan
        evidence_coverage = np.nan
        uniform_coverage = np.nan

    return {
        "alignment_gain_vs_uniform": alignment_gain,
        "alignment_cosine": cosine,
        "evidence_grid_coverage": evidence_coverage,
        "evidence_grid_uniform_coverage": uniform_coverage,
        "noise_grid_attention": float(cam_grid[noise_mask].sum()) if noise_mask.any() else 0.0,
    }


def summarize(frame: pd.DataFrame) -> dict[str, Any]:
    metric_columns = [
        "alignment_gain_vs_uniform",
        "alignment_cosine",
        "evidence_grid_coverage",
        "noise_grid_attention",
    ]
    result: dict[str, Any] = {
        "total_rows": int(len(frame)),
        "unique_images": int(frame["image_id"].nunique()),
        "by_target": {},
        "by_selection_reason": {},
        "paired_wrong_predictions": {},
    }
    for target, group in frame.groupby("target_kind"):
        result["by_target"][str(target)] = {
            "n": int(len(group)),
            **{column: float(group[column].mean()) for column in metric_columns},
        }
    for reason, group in frame.groupby("selection_reason"):
        result["by_selection_reason"][str(reason)] = {
            "n": int(group["image_id"].nunique()),
            **{
                f"{target}_{column}": float(target_group[column].mean())
                for target, target_group in group.groupby("target_kind")
                for column in metric_columns
            },
        }

    wrong = frame[frame["conv_pred"] != frame["true_label"]]
    paired = wrong.pivot_table(
        index=["image_id", "selection_reason"],
        columns="target_kind",
        values=metric_columns,
        aggfunc="first",
    )
    if ("alignment_gain_vs_uniform", "pred") in paired and ("alignment_gain_vs_uniform", "true") in paired:
        difference = (
            paired[("alignment_gain_vs_uniform", "true")]
            - paired[("alignment_gain_vs_uniform", "pred")]
        ).dropna()
        result["paired_wrong_predictions"] = {
            "n": int(len(difference)),
            "mean_true_minus_pred_alignment_gain": float(difference.mean()),
            "true_better_count": int((difference > 0).sum()),
            "pred_better_count": int((difference < 0).sum()),
            "tie_count": int((difference == 0).sum()),
        }
    return result


def main() -> None:
    args = parse_args()
    samples = pd.read_csv(args.samples)
    evidence_frame = pd.read_csv(args.evidence)
    evidence_frame = evidence_frame[evidence_frame["status"] == "ok"].copy()
    merged = samples.merge(
        evidence_frame[
            [
                "image_id",
                "region_evidence_json",
                "overall_localization_quality",
                "region_evidence_count",
            ]
        ],
        on="image_id",
        how="inner",
        validate="one_to_one",
    )
    if args.limit > 0:
        merged = merged.head(args.limit).copy()
    if len(merged) == 0:
        raise RuntimeError("No completed region-evidence rows matched the selected Stage 2 samples.")

    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    class_names = checkpoint.get("class_names")
    if not class_names:
        raise RuntimeError("Checkpoint does not contain class_names.")
    class_to_idx = {name: index for index, name in enumerate(class_names)}
    input_size = int(checkpoint.get("input_size", 224))

    model = build_model(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cam_dir = args.out_dir / "raw_cam"
    cam_dir.mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []

    for _, sample in merged.iterrows():
        image_id = str(sample["image_id"])
        evidence = parse_evidence(sample["region_evidence_json"])
        positive_prior, noise_prior = build_grid_priors(evidence)
        image_tensor, _ = preprocess_image(Path(str(sample["image_path"])), input_size)

        targets = {
            "pred": str(sample["conv_pred"]),
            "true": str(sample["true_label"]),
        }
        cached: dict[str, tuple[np.ndarray, np.ndarray, Path]] = {}
        for target_kind, target_label in targets.items():
            if target_label not in cached:
                cam, logits = compute_cam(model, image_tensor, class_to_idx[target_label], device)
                cam_path = cam_dir / f"{image_id}_{target_label}.npy"
                np.save(cam_path, cam.astype(np.float32))
                cached[target_label] = (cam, logits, cam_path)

            cam, logits, cam_path = cached[target_label]
            grid = cam_grid_distribution(cam)
            metrics = alignment_metrics(grid, positive_prior, noise_prior)
            rows.append(
                {
                    "image_id": image_id,
                    "selection_reason": str(sample["selection_reason"]),
                    "true_label": str(sample["true_label"]),
                    "conv_pred": str(sample["conv_pred"]),
                    "target_kind": target_kind,
                    "target_label": target_label,
                    "model_prediction": str(class_names[int(np.argmax(logits))]),
                    "localization_quality": str(sample["overall_localization_quality"]),
                    "region_evidence_count": int(sample["region_evidence_count"]),
                    "positive_prior_grids": "|".join(
                        GRID_NAMES[index] for index, value in enumerate(positive_prior) if value > 0
                    ),
                    "noise_prior_grids": "|".join(
                        GRID_NAMES[index] for index, value in enumerate(noise_prior) if value > 0
                    ),
                    "cam_path": str(cam_path),
                    **{f"cam_{name}": float(grid[index]) for index, name in enumerate(GRID_NAMES)},
                    **metrics,
                }
            )

    output = pd.DataFrame(rows)
    output.to_csv(args.out_dir / "cam_region_alignment.csv", index=False, encoding="utf-8-sig")
    summary = summarize(output)
    (args.out_dir / "cam_region_alignment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "samples": str(args.samples.resolve()),
                "evidence": str(args.evidence.resolve()),
                "checkpoint": str(args.checkpoint.resolve()),
                "device": str(device),
                "images": int(len(merged)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Output: {args.out_dir}")


if __name__ == "__main__":
    main()
