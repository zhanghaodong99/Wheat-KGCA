from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score


CLASS_NAMES = [
    "flag_leaf_booting",
    "flowering",
    "grain_filling",
    "heading",
    "maturity",
    "seedling",
    "stem_elongation",
    "tillering",
]


def yes(value: Any) -> bool:
    return str(value).strip().lower() in {"yes", "true", "1"}


def no(value: Any) -> bool:
    return str(value).strip().lower() in {"no", "false", "0"}


def numeric(row: pd.Series, name: str, default: float = 0.0) -> float:
    try:
        value = row.get(name, default)
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def allow_target(row: pd.Series, target: str, args: argparse.Namespace) -> bool:
    if not args.require_vlm_target:
        return True
    predicted_stage = str(row.get("predicted_stage", "")).strip()
    second_candidate = str(row.get("second_candidate", "")).strip()
    confidence = numeric(row, "confidence", 0.0)
    return (
        confidence >= args.vlm_confidence_threshold
        and (predicted_stage == target or (args.allow_vlm_second_candidate and second_candidate == target))
    )


def correct_one(row: pd.Series, args: argparse.Namespace) -> tuple[str, str]:
    pred = str(row["primary_pred_label"])
    anther_visible = yes(row.get("anther_visible", ""))
    anther_not_visible = no(row.get("anther_visible", ""))
    anther_score = numeric(row, "anther_visibility", 0.0)
    grain_score = numeric(row, "grain_fullness_score", 0.0)
    tillering_score = numeric(row, "tillering_strength", 0.0)
    tillering_yes = yes(row.get("tillering_clue", ""))
    tillering_no = no(row.get("tillering_clue", ""))
    spike_status = str(row.get("spike_status", "")).strip().lower()

    if pred == "heading" and allow_target(row, "flowering", args) and (
        anther_visible or anther_score >= args.anther_yes_threshold
    ):
        return "flowering", "heading_to_flowering_by_anther"
    if pred == "flowering" and anther_not_visible and anther_score <= args.anther_no_threshold:
        if allow_target(row, "heading", args) and (
            "emerged" in spike_status or "partly" in spike_status or "fully" in spike_status
        ):
            return "heading", "flowering_to_heading_no_anther"

    if (
        pred == "flowering"
        and allow_target(row, "grain_filling", args)
        and grain_score >= args.grain_yes_threshold
        and anther_score <= args.anther_no_threshold
    ):
        return "grain_filling", "flowering_to_grain_filling_by_grain"
    if pred == "grain_filling" and allow_target(row, "flowering", args) and (
        anther_visible or anther_score >= args.anther_yes_threshold
    ):
        return "flowering", "grain_filling_to_flowering_by_anther"

    if pred == "seedling" and allow_target(row, "tillering", args) and (
        tillering_yes or tillering_score >= args.tillering_yes_threshold
    ):
        return "tillering", "seedling_to_tillering_by_tiller"
    if pred == "tillering" and allow_target(row, "seedling", args) and tillering_no and tillering_score <= args.tillering_no_threshold:
        return "seedling", "tillering_to_seedling_no_tiller"

    return pred, ""


def metrics(labels: pd.Series, predictions: pd.Series) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply organ-evidence correction to adjacent-stage predictions.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--anther-yes-threshold", type=float, default=0.45)
    parser.add_argument("--anther-no-threshold", type=float, default=0.10)
    parser.add_argument("--grain-yes-threshold", type=float, default=0.70)
    parser.add_argument("--tillering-yes-threshold", type=float, default=0.65)
    parser.add_argument("--tillering-no-threshold", type=float, default=0.20)
    parser.add_argument("--require-vlm-target", action="store_true")
    parser.add_argument("--vlm-confidence-threshold", type=float, default=0.75)
    parser.add_argument("--allow-vlm-second-candidate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.predictions)
    semantic = pd.read_csv(args.semantic)
    merged = predictions.merge(semantic, on="image_id", how="left", suffixes=("", "_semantic"), validate="one_to_one")
    if len(merged) != len(predictions):
        raise RuntimeError("Prediction and semantic files did not merge one-to-one.")

    corrected = []
    reasons = []
    for _, row in merged.iterrows():
        label, reason = correct_one(row, args)
        corrected.append(label)
        reasons.append(reason)
    merged["organ_corrected_pred_label"] = corrected
    merged["organ_correction_reason"] = reasons
    merged["organ_corrected_correct"] = merged["stage_id"] == merged["organ_corrected_pred_label"]

    base_metrics = metrics(merged["stage_id"], merged["primary_pred_label"])
    corrected_metrics = metrics(merged["stage_id"], merged["organ_corrected_pred_label"])
    changed = merged["primary_pred_label"] != merged["organ_corrected_pred_label"]
    fixes = (~merged["primary_correct"]) & merged["organ_corrected_correct"]
    breaks = merged["primary_correct"] & (~merged["organ_corrected_correct"])

    summary = {
        "predictions": str(args.predictions),
        "semantic": str(args.semantic),
        "thresholds": {
            "anther_yes_threshold": args.anther_yes_threshold,
            "anther_no_threshold": args.anther_no_threshold,
            "grain_yes_threshold": args.grain_yes_threshold,
            "tillering_yes_threshold": args.tillering_yes_threshold,
            "tillering_no_threshold": args.tillering_no_threshold,
            "require_vlm_target": args.require_vlm_target,
            "vlm_confidence_threshold": args.vlm_confidence_threshold,
            "allow_vlm_second_candidate": args.allow_vlm_second_candidate,
        },
        "base_primary_test": base_metrics,
        "organ_corrected_test": corrected_metrics,
        "accuracy_change": corrected_metrics["accuracy"] - base_metrics["accuracy"],
        "changed_count": int(changed.sum()),
        "fix_count": int(fixes.sum()),
        "break_count": int(breaks.sum()),
        "net_fix": int(fixes.sum() - breaks.sum()),
        "reason_counts": merged.loc[changed, "organ_correction_reason"].value_counts().to_dict(),
    }

    merged.to_csv(args.out_dir / "test_predictions_organ_corrected.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        confusion_matrix(merged["stage_id"], merged["organ_corrected_pred_label"], labels=CLASS_NAMES),
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    ).to_csv(args.out_dir / "test_organ_corrected_confusion_matrix.csv", encoding="utf-8-sig")
    (args.out_dir / "organ_evidence_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
