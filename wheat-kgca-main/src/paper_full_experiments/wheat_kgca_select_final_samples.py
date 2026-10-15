from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


TARGET_PAIRS = [
    ("seedling", "tillering"),
    ("heading", "flowering"),
    ("flowering", "grain_filling"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select final Wheat-KGCA samples for CAM and boundary-aware VLM evidence.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/paper_full_experiments/manifests/full_test_2485.csv"),
    )
    parser.add_argument(
        "--baseline-pred",
        type=Path,
        default=Path("outputs/paper_full_experiments/convnext_features/full_test_2485/convnext_predictions.csv"),
    )
    parser.add_argument(
        "--h03-pred",
        type=Path,
        default=Path("outputs/paper_full_experiments/hierarchical_03_aux_heads_cam_valref_select_main_output/test_predictions.csv"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/paper_full_experiments/wheat_kgca_final"))
    parser.add_argument("--max-per-reason", type=int, default=12)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.70)
    parser.add_argument("--max-total", type=int, default=96)
    return parser.parse_args()


def add_rows(pool: pd.DataFrame, reason: str, max_rows: int, rows: list[pd.DataFrame]) -> None:
    if pool.empty:
        return
    take = pool.copy().head(max_rows)
    take["selection_reason"] = reason
    rows.append(take)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    baseline = pd.read_csv(args.baseline_pred)
    h03 = pd.read_csv(args.h03_pred)

    base_cols = ["image_id", "pred_label", "confidence", "correct"]
    h03_cols = ["image_id", "primary_pred_label", "refined_pred_label", "primary_correct", "refined_correct"]
    merged = (
        manifest.merge(
            baseline[base_cols].rename(
                columns={
                    "pred_label": "baseline_pred",
                    "confidence": "baseline_confidence",
                    "correct": "baseline_correct",
                }
            ),
            on="image_id",
            how="inner",
        )
        .merge(h03[h03_cols], on="image_id", how="inner")
        .copy()
    )
    if len(merged) != len(manifest):
        raise RuntimeError(f"Merged rows {len(merged)} != manifest rows {len(manifest)}")

    rows: list[pd.DataFrame] = []

    corrected = merged[(~merged["baseline_correct"].astype(bool)) & (merged["primary_correct"].astype(bool))]
    corrected = corrected.sort_values(["baseline_confidence", "stage_id"], ascending=[True, True])
    add_rows(corrected, "baseline_wrong_wheat_kgca_correct", args.max_per_reason, rows)

    introduced = merged[(merged["baseline_correct"].astype(bool)) & (~merged["primary_correct"].astype(bool))]
    introduced = introduced.sort_values(["baseline_confidence", "stage_id"], ascending=[False, True])
    add_rows(introduced, "baseline_correct_wheat_kgca_wrong", min(args.max_per_reason, 6), rows)

    low_conf = merged[merged["baseline_confidence"] < args.low_confidence_threshold].sort_values("baseline_confidence")
    add_rows(low_conf, f"baseline_conf_lt_{args.low_confidence_threshold:.2f}", args.max_per_reason, rows)

    for left, right in TARGET_PAIRS:
        pair = merged[merged["stage_id"].isin([left, right])].copy()
        pair["pair_error_rank"] = (~pair["baseline_correct"].astype(bool)).astype(int) + pair["primary_correct"].astype(bool).astype(int)
        pair = pair.sort_values(["pair_error_rank", "baseline_confidence"], ascending=[False, True])
        add_rows(pair, f"target_boundary_{left}_vs_{right}", args.max_per_reason, rows)

    for stage, group in merged.sort_values("baseline_confidence").groupby("stage_id", sort=True):
        add_rows(group, f"stage_representative_{stage}", 2, rows)

    selected = pd.concat(rows, ignore_index=True)
    selected = selected.drop_duplicates("image_id", keep="first").head(args.max_total).copy()
    selected["wheat_kgca_pred"] = selected["primary_pred_label"]
    selected["wheat_kgca_correct"] = selected["primary_correct"]

    keep_cols = [
        "image_id",
        "split",
        "stage_id",
        "stage_cn",
        "source_type",
        "source_dir",
        "source_path",
        "target_path",
        "image_path",
        "original_filename",
        "link_mode",
        "selection_reason",
        "baseline_pred",
        "baseline_confidence",
        "baseline_correct",
        "wheat_kgca_pred",
        "wheat_kgca_correct",
        "refined_pred_label",
        "refined_correct",
    ]
    selected[keep_cols].to_csv(args.out_dir / "wheat_kgca_selected_samples.csv", index=False, encoding="utf-8-sig")

    # This manifest keeps the original manifest columns plus selection metadata; the existing VLM runner only needs
    # standard manifest columns and will ignore extras.
    selected[keep_cols].to_csv(args.out_dir / "boundary_aware_vlm_manifest.csv", index=False, encoding="utf-8-sig")

    summary = {
        "selected_count": int(len(selected)),
        "reason_counts": selected["selection_reason"].value_counts().to_dict(),
        "target_pairs": TARGET_PAIRS,
        "baseline_wrong_wheat_kgca_correct_available": int(len(corrected)),
        "baseline_correct_wheat_kgca_wrong_available": int(len(introduced)),
        "low_confidence_threshold": args.low_confidence_threshold,
        "low_confidence_available": int(len(low_conf)),
    }
    (args.out_dir / "selected_samples_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
