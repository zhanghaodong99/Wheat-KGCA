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


DEFAULT_EVIDENCE = (
    PROJECT_ROOT
    / "outputs"
    / "fusion"
    / "stage2_cam"
    / "contrast_evidence"
    / "stage2_contrast_evidence_18"
    / "contrast_evidence_results.csv"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "fusion" / "stage2_cam" / "cam_contrast_alignment_v2"
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
        description="Compare candidate-specific VLM evidence with candidate-specific ConvNeXt Grad-CAM."
    )
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def parse_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        result = json.loads(value)
    except json.JSONDecodeError:
        return []
    return result if isinstance(result, list) else []


def build_prior(items: list[dict[str, Any]]) -> np.ndarray:
    prior = np.zeros(9, dtype=np.float64)
    for item in items:
        grid = str(item.get("grid", "")).strip()
        if grid not in GRID_TO_INDEX:
            continue
        confidence = float(item.get("confidence", 0.0))
        confidence = min(max(confidence, 0.0), 1.0)
        visibility_weight = 1.0 if str(item.get("visibility", "")) == "yes" else 0.55
        prior[GRID_TO_INDEX[grid]] = max(prior[GRID_TO_INDEX[grid]], confidence * visibility_weight)
    return prior


def cam_grid_distribution(cam: np.ndarray) -> np.ndarray:
    rows = np.array_split(np.arange(cam.shape[0]), 3)
    columns = np.array_split(np.arange(cam.shape[1]), 3)
    grid = np.asarray(
        [float(cam[np.ix_(row_indices, col_indices)].mean()) for row_indices in rows for col_indices in columns],
        dtype=np.float64,
    )
    return grid / grid.sum() if grid.sum() > 0 else np.full(9, 1 / 9, dtype=np.float64)


def alignment_gain(cam_grid: np.ndarray, prior: np.ndarray) -> float:
    if prior.sum() <= 0:
        return np.nan
    return float(9.0 * np.dot(cam_grid, prior / prior.sum()))


def main() -> None:
    args = parse_args()
    evidence = pd.read_csv(args.evidence)
    evidence = evidence[(evidence["status"] == "ok") & evidence["schema_valid"]].copy()
    evidence = evidence[
        evidence.apply(
            lambda row: len(parse_items(row["support_a_json"])) > 0 or len(parse_items(row["support_b_json"])) > 0,
            axis=1,
        )
    ].copy()
    if args.limit > 0:
        evidence = evidence.head(args.limit).copy()
    if len(evidence) == 0:
        raise RuntimeError("No rows contain candidate-specific support evidence.")

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
    raw_dir = args.out_dir / "raw_cam"
    raw_dir.mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []
    for _, item in evidence.iterrows():
        candidate_a = str(item["candidate_a"])
        candidate_b = str(item["candidate_b"])
        prior_a = build_prior(parse_items(item["support_a_json"]))
        prior_b = build_prior(parse_items(item["support_b_json"]))
        contradiction_a = build_prior(parse_items(item["contradiction_a_json"]))
        image_tensor, _ = preprocess_image(Path(str(item["image_path"])), input_size)

        candidate_metrics: dict[str, dict[str, Any]] = {}
        for key, label, prior in (("a", candidate_a, prior_a), ("b", candidate_b, prior_b)):
            cam, logits = compute_cam(model, image_tensor, class_to_idx[label], device)
            cam_path = raw_dir / f"{item['image_id']}_{key}_{label}.npy"
            np.save(cam_path, cam.astype(np.float32))
            grid = cam_grid_distribution(cam)
            candidate_metrics[key] = {
                "label": label,
                "gain": alignment_gain(grid, prior),
                "grid": grid,
                "cam_path": str(cam_path),
                "model_prediction": str(class_names[int(np.argmax(logits))]),
            }

        true_label = str(item["true_label"])
        true_candidate = "a" if true_label == candidate_a else "b" if true_label == candidate_b else "neither"
        support_a_available = bool(prior_a.sum() > 0)
        support_b_available = bool(prior_b.sum() > 0)
        a_cam_on_a_evidence = alignment_gain(candidate_metrics["a"]["grid"], prior_a)
        b_cam_on_a_evidence = alignment_gain(candidate_metrics["b"]["grid"], prior_a)
        a_cam_on_b_evidence = alignment_gain(candidate_metrics["a"]["grid"], prior_b)
        b_cam_on_b_evidence = alignment_gain(candidate_metrics["b"]["grid"], prior_b)

        if true_candidate == "a" and support_a_available:
            true_cam_gain = a_cam_on_a_evidence
            wrong_cam_gain = b_cam_on_a_evidence
            evidence_source = "support_a"
        elif true_candidate == "b" and support_b_available:
            true_cam_gain = b_cam_on_b_evidence
            wrong_cam_gain = a_cam_on_b_evidence
            evidence_source = "support_b"
        else:
            true_cam_gain = np.nan
            wrong_cam_gain = np.nan
            evidence_source = ""

        rows.append(
            {
                "image_id": item["image_id"],
                "selection_reason": item["selection_reason"],
                "true_label": true_label,
                "candidate_a": candidate_a,
                "candidate_b": candidate_b,
                "true_candidate": true_candidate,
                "support_a_available": support_a_available,
                "support_b_available": support_b_available,
                "true_evidence_source": evidence_source,
                "a_cam_on_a_evidence_gain": a_cam_on_a_evidence,
                "b_cam_on_a_evidence_gain": b_cam_on_a_evidence,
                "a_cam_on_b_evidence_gain": a_cam_on_b_evidence,
                "b_cam_on_b_evidence_gain": b_cam_on_b_evidence,
                "true_cam_gain": true_cam_gain,
                "wrong_cam_gain": wrong_cam_gain,
                "true_minus_wrong_gain": true_cam_gain - wrong_cam_gain,
                "true_cam_better": bool(true_cam_gain > wrong_cam_gain) if not np.isnan(true_cam_gain) else False,
                "candidate_a_contradiction_attention": alignment_gain(
                    candidate_metrics["a"]["grid"], contradiction_a
                ),
                "candidate_b_contradiction_attention": alignment_gain(
                    candidate_metrics["b"]["grid"], contradiction_a
                ),
                "candidate_a_cam_path": candidate_metrics["a"]["cam_path"],
                "candidate_b_cam_path": candidate_metrics["b"]["cam_path"],
                **{f"a_cam_{name}": float(candidate_metrics["a"]["grid"][index]) for index, name in enumerate(GRID_NAMES)},
                **{f"b_cam_{name}": float(candidate_metrics["b"]["grid"][index]) for index, name in enumerate(GRID_NAMES)},
            }
        )

    result = pd.DataFrame(rows)
    eligible = result.dropna(subset=["true_cam_gain", "wrong_cam_gain"]).copy()
    by_true_candidate = {}
    for candidate, group in eligible.groupby("true_candidate"):
        by_true_candidate[str(candidate)] = {
            "n": int(len(group)),
            "true_cam_better_count": int(group["true_cam_better"].sum()),
            "true_cam_better_rate": float(group["true_cam_better"].mean()),
            "mean_true_minus_wrong_gain": float(group["true_minus_wrong_gain"].mean()),
        }
    summary = {
        "responses_with_any_candidate_support": int(len(result)),
        "support_a_available_count": int(result["support_a_available"].sum()),
        "support_b_available_count": int(result["support_b_available"].sum()),
        "eligible_truth_matched_evidence_count": int(len(eligible)),
        "true_cam_better_count": int(eligible["true_cam_better"].sum()),
        "true_cam_better_rate": float(eligible["true_cam_better"].mean()) if len(eligible) else 0.0,
        "mean_true_cam_gain": float(eligible["true_cam_gain"].mean()) if len(eligible) else 0.0,
        "mean_wrong_cam_gain": float(eligible["wrong_cam_gain"].mean()) if len(eligible) else 0.0,
        "mean_true_minus_wrong_gain": float(eligible["true_minus_wrong_gain"].mean()) if len(eligible) else 0.0,
        "by_true_candidate": by_true_candidate,
    }
    result.to_csv(args.out_dir / "cam_contrast_alignment.csv", index=False, encoding="utf-8-sig")
    (args.out_dir / "cam_contrast_alignment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Output: {args.out_dir}")


if __name__ == "__main__":
    main()
