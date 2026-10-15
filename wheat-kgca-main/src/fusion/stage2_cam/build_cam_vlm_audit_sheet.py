from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SAMPLES = PROJECT_ROOT / "outputs" / "fusion" / "stage2_cam" / "stage2_cam_selected_samples.csv"
DEFAULT_PRED_INDEX = PROJECT_ROOT / "outputs" / "fusion" / "stage2_cam" / "gradcam" / "gradcam_index_pred.csv"
DEFAULT_TRUE_INDEX = PROJECT_ROOT / "outputs" / "fusion" / "stage2_cam" / "gradcam" / "gradcam_index_true.csv"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "fusion" / "stage2_cam" / "cam_vlm_audit_sheet.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a manual CAM-VLM evidence audit sheet for Stage 2.")
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--pred-index", type=Path, default=DEFAULT_PRED_INDEX)
    parser.add_argument("--true-index", type=Path, default=DEFAULT_TRUE_INDEX)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = pd.read_csv(args.samples)
    pred = pd.read_csv(args.pred_index)[["image_id", "cam_path"]].rename(columns={"cam_path": "pred_cam_path"})
    true = pd.read_csv(args.true_index)[["image_id", "cam_path"]].rename(columns={"cam_path": "true_cam_path"})
    audit = samples.merge(pred, on="image_id", how="left").merge(true, on="image_id", how="left")

    keep = [
        "stage2_index",
        "image_id",
        "selection_reason",
        "source_type",
        "image_path",
        "true_label",
        "conv_pred",
        "fusion_pred",
        "vlm_pred",
        "conv_confidence",
        "vlm_confidence",
        "short_reason",
        "uncertainty_reason",
        "spike_visibility",
        "anther_visibility",
        "grain_fullness_score",
        "noise_degree",
        "pred_cam_path",
        "true_cam_path",
    ]
    audit = audit[[column for column in keep if column in audit.columns]].copy()
    audit["vlm_evidence_visible"] = ""
    audit["pred_cam_hits_evidence"] = ""
    audit["true_cam_hits_evidence"] = ""
    audit["pred_cam_background_distracted"] = ""
    audit["review_decision"] = ""
    audit["review_note"] = ""

    args.out.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(audit)} audit rows: {args.out}")


if __name__ == "__main__":
    main()
