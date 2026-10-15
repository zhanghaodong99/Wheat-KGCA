from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "fusion" / "stage3_selective_cam" / "manifests"

DEFAULT_TRAIN_CONV = PROJECT_ROOT / "outputs" / "fusion" / "convnext_features" / "fusion_train_800" / "convnext_predictions.csv"
DEFAULT_TRAIN_NPZ = PROJECT_ROOT / "outputs" / "fusion" / "convnext_features" / "fusion_train_800" / "convnext_features.npz"
DEFAULT_TRAIN_VLM = (
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
DEFAULT_VAL_CONV = PROJECT_ROOT / "outputs" / "fusion" / "convnext_features" / "fusion_val_400" / "convnext_predictions.csv"
DEFAULT_VAL_NPZ = PROJECT_ROOT / "outputs" / "fusion" / "convnext_features" / "fusion_val_400" / "convnext_features.npz"
DEFAULT_VAL_VLM = (
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


def second_competitor_by_image(npz_path: Path) -> dict[str, str]:
    data = np.load(npz_path, allow_pickle=True)
    class_names = [str(value) for value in data["class_names"].tolist()]
    image_ids = [str(value) for value in data["image_id"].tolist()]
    logits = data["logits"]
    result: dict[str, str] = {}
    for image_id, scores in zip(image_ids, logits, strict=True):
        order = np.argsort(scores)[::-1]
        if len(order) < 2:
            raise RuntimeError("At least two classes are required for contrastive candidates.")
        result[image_id] = class_names[int(order[1])]
    return result


def build_split(
    split_name: str,
    conv_path: Path,
    npz_path: Path,
    vlm_path: Path,
    per_stage: int,
    min_vlm_confidence: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    conv = pd.read_csv(conv_path)
    vlm = pd.read_csv(vlm_path)
    vlm = vlm[(vlm["status"] == "ok") & (vlm["schema_valid"])].copy()
    merged = conv.merge(
        vlm[["image_id", "predicted_stage", "confidence", "short_reason", "uncertainty_reason"]],
        on="image_id",
        how="inner",
        validate="one_to_one",
    )
    merged = merged.rename(columns={"confidence_x": "conv_confidence", "confidence_y": "vlm_confidence"})
    if "conv_confidence" not in merged.columns:
        merged = merged.rename(columns={"confidence": "conv_confidence"})
    competitor = second_competitor_by_image(npz_path)
    merged["candidate_a"] = merged["image_id"].map(competitor)
    merged["candidate_b"] = merged["stage_id"].astype(str)

    # This label check is allowed for supervised training-data construction.
    # It never affects test data and the label is never sent to the VLM prompt.
    reliable = merged[
        (merged["pred_label"] == merged["stage_id"])
        & (merged["predicted_stage"] == merged["stage_id"])
        & (merged["vlm_confidence"] >= min_vlm_confidence)
        & (merged["candidate_a"] != merged["candidate_b"])
    ].copy()
    reliable = reliable.sort_values(["stage_id", "conv_confidence", "image_id"], kind="stable")
    selected = (
        reliable.groupby("stage_id", group_keys=False)
        .head(per_stage)
        .copy()
    )
    selected["split_role"] = split_name
    selected["selection_rule"] = "conv_correct_and_vlm_correct_nearest_visual_competitor"
    selected["selected_rank_within_stage"] = selected.groupby("stage_id").cumcount() + 1
    columns = [
        "image_id",
        "image_key",
        "split",
        "stage_id",
        "source_type",
        "image_path",
        "candidate_a",
        "candidate_b",
        "pred_label",
        "predicted_stage",
        "conv_confidence",
        "vlm_confidence",
        "short_reason",
        "uncertainty_reason",
        "split_role",
        "selection_rule",
        "selected_rank_within_stage",
    ]
    summary = {
        "available_reliable": int(len(reliable)),
        "selected": int(len(selected)),
        "available_by_stage": {str(key): int(value) for key, value in reliable.groupby("stage_id").size().items()},
        "selected_by_stage": {str(key): int(value) for key, value in selected.groupby("stage_id").size().items()},
    }
    return selected[columns], summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build label-reliable train/val candidate manifests for selective CAM guidance."
    )
    parser.add_argument("--train-conv", type=Path, default=DEFAULT_TRAIN_CONV)
    parser.add_argument("--train-npz", type=Path, default=DEFAULT_TRAIN_NPZ)
    parser.add_argument("--train-vlm", type=Path, default=DEFAULT_TRAIN_VLM)
    parser.add_argument("--val-conv", type=Path, default=DEFAULT_VAL_CONV)
    parser.add_argument("--val-npz", type=Path, default=DEFAULT_VAL_NPZ)
    parser.add_argument("--val-vlm", type=Path, default=DEFAULT_VAL_VLM)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--train-per-stage", type=int, default=20)
    parser.add_argument("--val-per-stage", type=int, default=10)
    parser.add_argument("--min-vlm-confidence", type=float, default=0.70)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train, train_summary = build_split(
        "train",
        args.train_conv,
        args.train_npz,
        args.train_vlm,
        args.train_per_stage,
        args.min_vlm_confidence,
    )
    val, val_summary = build_split(
        "val",
        args.val_conv,
        args.val_npz,
        args.val_vlm,
        args.val_per_stage,
        args.min_vlm_confidence,
    )
    train_path = args.out_dir / f"stage3_train_candidates_{len(train)}.csv"
    val_path = args.out_dir / f"stage3_val_candidates_{len(val)}.csv"
    train.to_csv(train_path, index=False, encoding="utf-8-sig")
    val.to_csv(val_path, index=False, encoding="utf-8-sig")
    summary = {
        "selection": "conv_correct_and_vlm_correct_nearest_visual_competitor",
        "min_vlm_confidence": args.min_vlm_confidence,
        "train": train_summary,
        "val": val_summary,
        "train_manifest": str(train_path),
        "val_manifest": str(val_path),
    }
    (args.out_dir / "stage3_candidate_manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
