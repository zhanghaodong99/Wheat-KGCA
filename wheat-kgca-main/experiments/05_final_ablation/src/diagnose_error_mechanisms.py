from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.metrics import confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MANIFEST = PROJECT_ROOT / "outputs" / "paper_full_experiments" / "manifests" / "full_test_2485.csv"
DEFAULT_BASELINE = (
    PROJECT_ROOT / "outputs" / "paper_full_experiments" / "convnext_features" / "full_test_2485" / "convnext_predictions.csv"
)
DEFAULT_WHEAT_SAGE_CAM = (
    PROJECT_ROOT / "outputs" / "paper_full_experiments" / "expert_spatial_ablation_lambda00025" / "test_predictions.csv"
)
DEFAULT_ATTENTION15 = (
    PROJECT_ROOT
    / "outputs"
    / "paper_full_experiments"
    / "attention_15_cam_guided_cross_attention_attn_loss"
    / "test_predictions.csv"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "paper_full_experiments" / "error_diagnosis"

STAGE_ORDER = [
    "seedling",
    "tillering",
    "stem_elongation",
    "flag_leaf_booting",
    "heading",
    "flowering",
    "grain_filling",
    "maturity",
]

TARGET_REVIEW_PAIRS = [
    ("seedling", "tillering"),
    ("heading", "flowering"),
    ("flowering", "grain_filling"),
]


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    columns = [str(col) for col in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        values = []
        for col in frame.columns:
            value = row[col]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(value.replace("|", "/") for value in values) + " |")
    return "\n".join(lines)


def normalize_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def pair_key(left: str, right: str) -> str:
    return " / ".join(sorted([str(left), str(right)]))


def adjacent_pairs(stage_order: Iterable[str]) -> set[str]:
    stages = list(stage_order)
    return {pair_key(stages[index], stages[index + 1]) for index in range(len(stages) - 1)}


def read_prediction(path: Path, model_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    keep = ["image_id", "pred_label", "correct"]
    if "confidence" in frame.columns:
        keep.append("confidence")
    frame = frame[keep].copy()
    frame = frame.rename(
        columns={
            "pred_label": f"{model_name}_pred",
            "correct": f"{model_name}_correct",
            "confidence": f"{model_name}_confidence",
        }
    )
    frame[f"{model_name}_correct"] = normalize_bool(frame[f"{model_name}_correct"])
    return frame


def accuracy_rows(comparison: pd.DataFrame, models: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in models:
        correct_col = f"{model}_correct"
        pred_col = f"{model}_pred"
        rows.append(
            {
                "group": "overall",
                "value": "all",
                "model": model,
                "samples": len(comparison),
                "correct": int(comparison[correct_col].sum()),
                "accuracy": float(comparison[correct_col].mean()),
                "errors": int((~comparison[correct_col]).sum()),
            }
        )
        for source, group in comparison.groupby("source_type", dropna=False):
            rows.append(
                {
                    "group": "source_type",
                    "value": source,
                    "model": model,
                    "samples": len(group),
                    "correct": int(group[correct_col].sum()),
                    "accuracy": float(group[correct_col].mean()),
                    "errors": int((~group[correct_col]).sum()),
                }
            )
        for stage, group in comparison.groupby("stage_id", dropna=False):
            rows.append(
                {
                    "group": "stage_id",
                    "value": stage,
                    "model": model,
                    "samples": len(group),
                    "correct": int(group[correct_col].sum()),
                    "accuracy": float(group[correct_col].mean()),
                    "errors": int((~group[correct_col]).sum()),
                }
            )
    return rows


def write_confusions(comparison: pd.DataFrame, models: list[str], class_names: list[str], out_dir: Path) -> None:
    matrix_dir = out_dir / "source_confusion_matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    for model in models:
        pred_col = f"{model}_pred"
        matrix = pd.DataFrame(
            confusion_matrix(comparison["stage_id"], comparison[pred_col], labels=class_names),
            index=class_names,
            columns=class_names,
        )
        matrix.to_csv(matrix_dir / f"{model}_overall_confusion.csv", encoding="utf-8-sig")
        for source, group in comparison.groupby("source_type", dropna=False):
            safe_source = str(source).replace("\\", "_").replace("/", "_").replace(":", "_")
            matrix = pd.DataFrame(
                confusion_matrix(group["stage_id"], group[pred_col], labels=class_names),
                index=class_names,
                columns=class_names,
            )
            matrix.to_csv(matrix_dir / f"{model}_{safe_source}_confusion.csv", encoding="utf-8-sig")


def build_pair_summary(comparison: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    for model in models:
        wrong = comparison[~comparison[f"{model}_correct"]].copy()
        wrong["confusion_pair"] = wrong.apply(lambda row: pair_key(row["stage_id"], row[f"{model}_pred"]), axis=1)
        for pair, group in wrong.groupby("confusion_pair"):
            rows.append(
                {
                    "model": model,
                    "confusion_pair": pair,
                    "errors": len(group),
                    "target_review_pair": pair in {pair_key(*item) for item in TARGET_REVIEW_PAIRS},
                    "stage_adjacent_pair": pair in adjacent_pairs(STAGE_ORDER),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["model", "confusion_pair", "errors", "target_review_pair", "stage_adjacent_pair"])
    return pd.DataFrame(rows).sort_values(["errors", "model"], ascending=[False, True])


def write_markdown(
    out_path: Path,
    comparison: pd.DataFrame,
    source_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    low_confidence_threshold: float,
    models: list[str],
) -> None:
    lines = [
        "# Error Mechanism Diagnosis",
        "",
        f"- total test samples: {len(comparison)}",
        f"- models: {', '.join(models)}",
        f"- low confidence threshold: baseline confidence < {low_confidence_threshold}",
        f"- low confidence samples: {int(comparison['baseline_low_confidence'].sum())}",
        f"- model disagreement samples: {int(comparison['model_disagreement'].sum())}",
        f"- adjacent confusion samples: {int(comparison['any_adjacent_confusion'].sum())}",
        f"- relabel review candidates: {int(comparison['relabel_review_candidate'].sum())}",
        "",
        "## Model Accuracy",
        "",
    ]
    overall = source_summary[source_summary["group"] == "overall"].copy()
    if not overall.empty:
        overall["accuracy_percent"] = overall["accuracy"] * 100.0
        lines.append(dataframe_to_markdown(overall[["model", "samples", "correct", "errors", "accuracy_percent"]]))
        lines.append("")
    lines += ["## Source Accuracy", ""]
    source_rows = source_summary[source_summary["group"] == "source_type"].copy()
    if not source_rows.empty:
        source_rows["accuracy_percent"] = source_rows["accuracy"] * 100.0
        lines.append(dataframe_to_markdown(source_rows[["value", "model", "samples", "correct", "errors", "accuracy_percent"]]))
        lines.append("")
    lines += ["## Top Confusion Pairs", ""]
    if not pair_summary.empty:
        lines.append(dataframe_to_markdown(pair_summary.head(30)))
        lines.append("")
    lines += [
        "## Next Use",
        "",
        "Review `relabel_review_candidates.csv` first, then build pair-specialist datasets for the target adjacent pairs.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose error mechanisms for full wheat paper experiments.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--baseline-predictions", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--wheat-sage-cam-predictions", type=Path, default=DEFAULT_WHEAT_SAGE_CAM)
    parser.add_argument("--attention15-predictions", type=Path, default=DEFAULT_ATTENTION15)
    parser.add_argument("--skip-wheat-sage-cam", action="store_true")
    parser.add_argument("--skip-attention15", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.85)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    baseline = read_prediction(args.baseline_predictions, "baseline")
    optional_predictions: list[tuple[str, Path, bool]] = [
        ("wheat_sage_cam", args.wheat_sage_cam_predictions, args.skip_wheat_sage_cam),
        ("attention15", args.attention15_predictions, args.skip_attention15),
    ]
    models = ["baseline"]

    comparison = manifest.merge(baseline, on="image_id", how="left", validate="one_to_one")
    for model_name, prediction_path, skip in optional_predictions:
        if skip:
            continue
        if not prediction_path.exists():
            print(f"skip_missing_model={model_name}, path={prediction_path}")
            continue
        comparison = comparison.merge(read_prediction(prediction_path, model_name), on="image_id", how="left", validate="one_to_one")
        models.append(model_name)
    if comparison[[f"{model}_pred" for model in models]].isna().any().any():
        missing = comparison[comparison[[f"{model}_pred" for model in models]].isna().any(axis=1)]["image_id"].tolist()
        raise RuntimeError(f"Missing predictions for {len(missing)} images, first={missing[:5]}")

    review_pair_keys = {pair_key(*item) for item in TARGET_REVIEW_PAIRS}
    all_adjacent_pair_keys = adjacent_pairs(STAGE_ORDER)
    comparison["baseline_low_confidence"] = comparison["baseline_confidence"] < args.low_confidence_threshold
    comparison["model_disagreement"] = comparison[[f"{model}_pred" for model in models]].nunique(axis=1) > 1
    comparison["any_model_error"] = ~comparison[[f"{model}_correct" for model in models]].all(axis=1)

    adjacent_flags = []
    target_adjacent_flags = []
    for model in models:
        pred_col = f"{model}_pred"
        comparison[f"{model}_confusion_pair"] = comparison.apply(
            lambda row: pair_key(row["stage_id"], row[pred_col]), axis=1
        )
        comparison[f"{model}_stage_adjacent_error"] = (
            ~comparison[f"{model}_correct"] & comparison[f"{model}_confusion_pair"].isin(all_adjacent_pair_keys)
        )
        comparison[f"{model}_target_pair_error"] = (
            ~comparison[f"{model}_correct"] & comparison[f"{model}_confusion_pair"].isin(review_pair_keys)
        )
        adjacent_flags.append(f"{model}_stage_adjacent_error")
        target_adjacent_flags.append(f"{model}_target_pair_error")

    comparison["any_adjacent_confusion"] = comparison[adjacent_flags].any(axis=1)
    comparison["any_target_pair_confusion"] = comparison[target_adjacent_flags].any(axis=1)
    comparison["relabel_review_candidate"] = (
        comparison["any_target_pair_confusion"]
        | (comparison["baseline_low_confidence"] & comparison["model_disagreement"])
        | (comparison["model_disagreement"] & comparison["any_model_error"])
    )

    comparison.to_csv(args.out_dir / "all_model_comparison.csv", index=False, encoding="utf-8-sig")
    comparison[~comparison["baseline_correct"]].to_csv(args.out_dir / "baseline_errors.csv", index=False, encoding="utf-8-sig")
    for model in models:
        if model == "baseline":
            continue
        comparison[~comparison[f"{model}_correct"]].to_csv(
            args.out_dir / f"{model}_errors.csv", index=False, encoding="utf-8-sig"
        )
    comparison[comparison["baseline_low_confidence"]].sort_values("baseline_confidence").to_csv(
        args.out_dir / "low_confidence_samples.csv", index=False, encoding="utf-8-sig"
    )
    comparison[comparison["model_disagreement"]].to_csv(
        args.out_dir / "model_disagreement_samples.csv", index=False, encoding="utf-8-sig"
    )
    comparison[comparison["any_adjacent_confusion"]].to_csv(
        args.out_dir / "adjacent_confusions.csv", index=False, encoding="utf-8-sig"
    )
    comparison[comparison["any_target_pair_confusion"]].to_csv(
        args.out_dir / "target_pair_confusions.csv", index=False, encoding="utf-8-sig"
    )
    comparison[comparison["relabel_review_candidate"]].sort_values(
        ["any_target_pair_confusion", "model_disagreement", "baseline_confidence"],
        ascending=[False, False, True],
    ).to_csv(args.out_dir / "relabel_review_candidates.csv", index=False, encoding="utf-8-sig")

    source_summary = pd.DataFrame(accuracy_rows(comparison, models))
    source_summary.to_csv(args.out_dir / "source_accuracy_summary.csv", index=False, encoding="utf-8-sig")
    pair_summary = build_pair_summary(comparison, models)
    pair_summary.to_csv(args.out_dir / "confusion_pair_summary.csv", index=False, encoding="utf-8-sig")
    write_confusions(comparison, models, STAGE_ORDER, args.out_dir)
    write_markdown(args.out_dir / "diagnosis_summary.md", comparison, source_summary, pair_summary, args.low_confidence_threshold, models)

    print(f"out_dir={args.out_dir}")
    print(f"models={','.join(models)}")
    print(f"samples={len(comparison)}")
    print(f"low_confidence={int(comparison['baseline_low_confidence'].sum())}")
    print(f"model_disagreement={int(comparison['model_disagreement'].sum())}")
    print(f"adjacent_confusions={int(comparison['any_adjacent_confusion'].sum())}")
    print(f"target_pair_confusions={int(comparison['any_target_pair_confusion'].sum())}")
    print(f"relabel_review_candidates={int(comparison['relabel_review_candidate'].sum())}")


if __name__ == "__main__":
    main()
