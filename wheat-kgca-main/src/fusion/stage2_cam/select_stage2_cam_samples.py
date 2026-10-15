from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONV = PROJECT_ROOT / "outputs" / "fusion" / "convnext_features" / "fusion_test_800" / "convnext_predictions.csv"
DEFAULT_FUSION = PROJECT_ROOT / "outputs" / "fusion" / "semantic_visual_module" / "test_predictions.csv"
DEFAULT_VLM = (
    PROJECT_ROOT
    / "outputs"
    / "fusion"
    / "vlm_semantics"
    / "stage3_20260715_173238"
    / "gpt-5.5"
    / "P7"
    / "gpt-5.5_P7_fusion_test_800"
    / "parsed_results.csv"
)
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "fusion" / "stage2_cam" / "stage2_cam_selected_samples.csv"


ADJACENT_PAIRS = {
    ("seedling", "tillering"),
    ("tillering", "stem_elongation"),
    ("stem_elongation", "flag_leaf_booting"),
    ("flag_leaf_booting", "heading"),
    ("heading", "flowering"),
    ("flowering", "grain_filling"),
    ("grain_filling", "maturity"),
}


def is_adjacent(true_label: str, pred_label: str) -> bool:
    pair = (str(true_label), str(pred_label))
    return pair in ADJACENT_PAIRS or pair[::-1] in ADJACENT_PAIRS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select hard samples for Stage 2 CAM inspection.")
    parser.add_argument("--conv-pred", type=Path, default=DEFAULT_CONV)
    parser.add_argument("--fusion-pred", type=Path, default=DEFAULT_FUSION)
    parser.add_argument("--vlm-results", type=Path, default=DEFAULT_VLM)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--low-conf-threshold", type=float, default=0.90)
    parser.add_argument("--max-per-group", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conv = pd.read_csv(args.conv_pred)
    fusion = pd.read_csv(args.fusion_pred)
    vlm = pd.read_csv(args.vlm_results)

    df = conv[
        [
            "image_id",
            "image_key",
            "split",
            "stage_id",
            "source_type",
            "image_path",
            "pred_label",
            "confidence",
            "correct",
        ]
    ].rename(
        columns={
            "stage_id": "true_label",
            "pred_label": "conv_pred",
            "confidence": "conv_confidence",
            "correct": "conv_correct",
        }
    )
    df = df.merge(
        fusion[["image_id", "pred_label", "correct"]].rename(
            columns={"pred_label": "fusion_pred", "correct": "fusion_correct"}
        ),
        on="image_id",
        how="left",
    )
    df = df.merge(
        vlm[
            [
                "image_id",
                "predicted_stage",
                "confidence",
                "short_reason",
                "uncertainty_reason",
                "spike_visibility",
                "anther_visibility",
                "grain_fullness_score",
                "noise_degree",
            ]
        ].rename(columns={"predicted_stage": "vlm_pred", "confidence": "vlm_confidence"}),
        on="image_id",
        how="left",
    )

    df["conv_correct"] = df["conv_pred"].astype(str).eq(df["true_label"].astype(str))
    df["fusion_correct"] = df["fusion_pred"].astype(str).eq(df["true_label"].astype(str))
    df["vlm_correct"] = df["vlm_pred"].astype(str).eq(df["true_label"].astype(str))
    df["is_low_confidence"] = pd.to_numeric(df["conv_confidence"], errors="coerce") < args.low_conf_threshold
    df["is_conv_wrong"] = ~df["conv_correct"]
    df["is_fusion_corrected"] = df["is_conv_wrong"] & df["fusion_correct"]
    df["is_vlm_can_correct"] = df["is_conv_wrong"] & df["vlm_correct"]
    df["is_adjacent_confusion"] = [
        is_adjacent(t, p) for t, p in zip(df["true_label"], df["conv_pred"], strict=False)
    ]

    selected_parts = []
    selection_rules = [
        ("fusion_corrected", df["is_fusion_corrected"]),
        ("vlm_can_correct", df["is_vlm_can_correct"]),
        ("conv_wrong", df["is_conv_wrong"]),
        ("low_confidence", df["is_low_confidence"]),
        ("adjacent_confusion", df["is_adjacent_confusion"]),
    ]
    used_ids: set[str] = set()
    for tag, mask in selection_rules:
        part = df[mask].copy()
        part = part[~part["image_id"].isin(used_ids)]
        part = part.sort_values(["conv_confidence", "image_id"]).head(args.max_per_group)
        part["selection_reason"] = tag
        selected_parts.append(part)
        used_ids.update(part["image_id"].astype(str).tolist())

    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    selected = selected.drop_duplicates("image_id").reset_index(drop=True)
    selected["stage2_index"] = range(1, len(selected) + 1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"Selected {len(selected)} samples for Stage 2 CAM inspection.")
    print(selected["selection_reason"].value_counts().to_string())
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
