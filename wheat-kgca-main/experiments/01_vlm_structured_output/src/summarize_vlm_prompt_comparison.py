from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "vlm_comparison" / "prompt_comparisons"

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
STAGE_INDEX = {stage: index for index, stage in enumerate(STAGE_ORDER)}


def parse_run_spec(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError(f"Run spec must be PROMPT_ID=RUN_DIR, got: {text}")
    prompt_id, run_dir = text.split("=", 1)
    prompt_id = prompt_id.strip()
    run_dir = Path(run_dir.strip())
    if not prompt_id:
        raise ValueError(f"Missing prompt id in run spec: {text}")
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    return prompt_id, run_dir


def load_results(prompt_id: str, run_dir: Path) -> pd.DataFrame:
    path = run_dir / "parsed_results.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["prompt_id"] = prompt_id
    df["run_dir"] = str(run_dir)
    df["correct_bool"] = df["correct"].astype(str).str.lower().isin(["true", "1", "yes"])
    return df


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def compute_summary(df: pd.DataFrame, prompt_id: str, run_dir: Path) -> dict[str, Any]:
    valid = df[df["label_valid"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    y_true = valid["stage_id"].astype(str).tolist()
    y_pred = valid["predicted_stage"].astype(str).tolist()

    summary: dict[str, Any] = {
        "prompt_id": prompt_id,
        "run_dir": str(run_dir),
        "total": len(df),
        "ok": int((df["status"].astype(str) == "ok").sum()) if "status" in df else len(df),
        "json_valid_rate": mean_bool(df, "json_valid"),
        "schema_valid_rate": mean_bool(df, "schema_valid"),
        "label_valid_rate": mean_bool(df, "label_valid"),
        "evidence_complete_rate": mean_bool(df, "evidence_complete"),
        "contradiction_rate": float((df.get("contradiction_count", 0).fillna(0).astype(float) > 0).mean())
        if "contradiction_count" in df
        else 0.0,
        "mean_latency_s": df["elapsed_sec"].map(safe_float).mean() if "elapsed_sec" in df else 0.0,
        "mean_confidence": df["confidence"].map(safe_float).mean() if "confidence" in df else 0.0,
    }

    if y_true:
        summary.update(
            {
                "accuracy": accuracy_score(y_true, y_pred),
                "macro_f1": f1_score(y_true, y_pred, labels=STAGE_ORDER, average="macro", zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            }
        )
    else:
        summary.update({"accuracy": 0.0, "macro_f1": 0.0, "balanced_accuracy": 0.0})

    errors = valid[valid["stage_id"].astype(str) != valid["predicted_stage"].astype(str)]
    distances = []
    for _, row in errors.iterrows():
        true_stage = str(row["stage_id"])
        pred_stage = str(row["predicted_stage"])
        if true_stage in STAGE_INDEX and pred_stage in STAGE_INDEX:
            distances.append(abs(STAGE_INDEX[true_stage] - STAGE_INDEX[pred_stage]))
    severe_count = sum(distance > 1 for distance in distances)
    adjacent_count = sum(distance == 1 for distance in distances)
    summary.update(
        {
            "errors": len(errors),
            "adjacent_error_count": adjacent_count,
            "adjacent_error_rate": adjacent_count / len(valid) if len(valid) else 0.0,
            "severe_error_count": severe_count,
            "severe_error_rate": severe_count / len(valid) if len(valid) else 0.0,
            "mean_error_distance": sum(distances) / len(distances) if distances else 0.0,
        }
    )
    return summary


def mean_bool(df: pd.DataFrame, column: str) -> float:
    if column not in df:
        return 0.0
    return float(df[column].astype(str).str.lower().isin(["true", "1", "yes"]).mean())


def per_stage_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for prompt_id, group in df.groupby("prompt_id", sort=False):
        for stage in STAGE_ORDER:
            subset = group[group["stage_id"].astype(str) == stage]
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "stage_id": stage,
                    "n": len(subset),
                    "accuracy": float(subset["correct_bool"].mean()) if len(subset) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def source_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    if "source_type" not in df:
        return pd.DataFrame()
    rows = []
    for (prompt_id, source_type), group in df.groupby(["prompt_id", "source_type"], sort=False):
        rows.append(
            {
                "prompt_id": prompt_id,
                "source_type": source_type,
                "n": len(group),
                "accuracy": float(group["correct_bool"].mean()) if len(group) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def confusion_rows(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    rows = []
    for prompt_id, group in df.groupby("prompt_id", sort=False):
        errors = group[group["stage_id"].astype(str) != group["predicted_stage"].astype(str)]
        counter = Counter(zip(errors["stage_id"].astype(str), errors["predicted_stage"].astype(str)))
        for (true_stage, pred_stage), count in counter.most_common(top_k):
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "true_stage": true_stage,
                    "predicted_stage": pred_stage,
                    "count": count,
                }
            )
    return pd.DataFrame(rows)


def flowering_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    subset = df[df["stage_id"].astype(str) == "flowering"]
    for prompt_id, group in subset.groupby("prompt_id", sort=False):
        pred_counter = Counter(group["predicted_stage"].astype(str))
        anther_counter = Counter(group.get("anther_visible", pd.Series(dtype=str)).astype(str))
        rows.append(
            {
                "prompt_id": prompt_id,
                "n": len(group),
                "accuracy": float(group["correct_bool"].mean()) if len(group) else 0.0,
                "pred_flowering": pred_counter.get("flowering", 0),
                "pred_heading": pred_counter.get("heading", 0),
                "pred_grain_filling": pred_counter.get("grain_filling", 0),
                "anther_yes": anther_counter.get("yes", 0),
                "anther_no": anther_counter.get("no", 0),
                "anther_uncertain": anther_counter.get("uncertain", 0),
            }
        )
    return pd.DataFrame(rows)


def grain_filling_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    subset = df[df["stage_id"].astype(str) == "grain_filling"]
    for prompt_id, group in subset.groupby("prompt_id", sort=False):
        pred_counter = Counter(group["predicted_stage"].astype(str))
        spike_counter = Counter(group.get("spike_status", pd.Series(dtype=str)).astype(str))
        grain_counter = Counter(group.get("grain_fullness", pd.Series(dtype=str)).astype(str))
        rows.append(
            {
                "prompt_id": prompt_id,
                "n": len(group),
                "accuracy": float(group["correct_bool"].mean()) if len(group) else 0.0,
                "pred_grain_filling": pred_counter.get("grain_filling", 0),
                "pred_heading": pred_counter.get("heading", 0),
                "pred_flowering": pred_counter.get("flowering", 0),
                "spike_full_grain": spike_counter.get("full_grain", 0),
                "spike_fully_emerged": spike_counter.get("fully_emerged", 0),
                "grain_none": grain_counter.get("none", 0),
                "grain_slight": grain_counter.get("slight", 0),
                "grain_moderate": grain_counter.get("moderate", 0),
                "grain_clear": grain_counter.get("clear", 0),
                "grain_full": grain_counter.get("full", 0),
                "grain_uncertain": grain_counter.get("uncertain", 0),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, columns: list[str], float_cols: set[str] | None = None) -> str:
    if df.empty:
        return "_No data._"
    float_cols = float_cols or set()
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row.get(column, "")
            if column in float_cols:
                values.append(f"{safe_float(value):.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_markdown(
    output_path: Path,
    summary: pd.DataFrame,
    per_stage: pd.DataFrame,
    by_source: pd.DataFrame,
    confusions: pd.DataFrame,
    flowering: pd.DataFrame,
    grain_filling: pd.DataFrame,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stage_pivot = per_stage.pivot(index="stage_id", columns="prompt_id", values="accuracy").reset_index()
    stage_columns = ["stage_id"] + [column for column in stage_pivot.columns if column != "stage_id"]

    text = f"""# VLM Prompt Comparison Summary

## Overall Metrics

{markdown_table(
    summary,
    [
        "prompt_id",
        "total",
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "schema_valid_rate",
        "evidence_complete_rate",
        "severe_error_rate",
        "mean_latency_s",
    ],
    {
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "schema_valid_rate",
        "evidence_complete_rate",
        "severe_error_rate",
        "mean_latency_s",
    },
)}

## Per-Stage Accuracy

{markdown_table(stage_pivot, stage_columns, set(stage_columns) - {"stage_id"})}

## Source Accuracy

{markdown_table(by_source, ["prompt_id", "source_type", "n", "accuracy"], {"accuracy"})}

## Top Confusions

{markdown_table(confusions, ["prompt_id", "true_stage", "predicted_stage", "count"])}

## Flowering Breakdown

{markdown_table(
    flowering,
    [
        "prompt_id",
        "n",
        "accuracy",
        "pred_flowering",
        "pred_heading",
        "pred_grain_filling",
        "anther_yes",
        "anther_no",
        "anther_uncertain",
    ],
    {"accuracy"},
)}

## Grain Filling Breakdown

{markdown_table(
    grain_filling,
    [
        "prompt_id",
        "n",
        "accuracy",
        "pred_grain_filling",
        "pred_heading",
        "pred_flowering",
        "spike_full_grain",
        "spike_fully_emerged",
        "grain_none",
        "grain_slight",
        "grain_moderate",
        "grain_clear",
        "grain_full",
        "grain_uncertain",
    ],
    {"accuracy"},
)}
"""
    output_path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize and compare VLM prompt run directories.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Prompt run spec in the form PROMPT_ID=RUN_DIR. Can be provided multiple times.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--name", type=str, default="prompt_comparison")
    parser.add_argument("--top-k-confusions", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_specs = [parse_run_spec(spec) for spec in args.run]
    frames = [load_results(prompt_id, run_dir) for prompt_id, run_dir in run_specs]
    all_results = pd.concat(frames, ignore_index=True)

    output_dir = args.output_dir / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results.to_csv(output_dir / "combined_results.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame([compute_summary(df, prompt_id, run_dir) for (prompt_id, run_dir), df in zip(run_specs, frames)])
    per_stage = per_stage_accuracy(all_results)
    by_source = source_accuracy(all_results)
    confusions = confusion_rows(all_results, args.top_k_confusions)
    flowering = flowering_breakdown(all_results)
    grain_filling = grain_filling_breakdown(all_results)

    summary.to_csv(output_dir / "overall_summary.csv", index=False, encoding="utf-8-sig")
    per_stage.to_csv(output_dir / "per_stage_accuracy.csv", index=False, encoding="utf-8-sig")
    if not by_source.empty:
        by_source.to_csv(output_dir / "source_accuracy.csv", index=False, encoding="utf-8-sig")
    confusions.to_csv(output_dir / "top_confusions.csv", index=False, encoding="utf-8-sig")
    flowering.to_csv(output_dir / "flowering_breakdown.csv", index=False, encoding="utf-8-sig")
    grain_filling.to_csv(output_dir / "grain_filling_breakdown.csv", index=False, encoding="utf-8-sig")
    write_markdown(
        output_dir / "prompt_comparison_summary.md",
        summary,
        per_stage,
        by_source,
        confusions,
        flowering,
        grain_filling,
    )

    print(output_dir / "prompt_comparison_summary.md")


if __name__ == "__main__":
    main()
