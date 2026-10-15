from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize one stage from a VLM parsed_results.csv file.")
    parser.add_argument("--parsed-results", type=Path, required=True)
    parser.add_argument("--stage-id", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", type=str, default="")
    return parser.parse_args()


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def md_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    if not rows:
        return "_No data._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def counter_rows(counter: Counter, key_name: str) -> list[dict[str, object]]:
    return [{key_name: key, "count": count} for key, count in counter.most_common()]


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.parsed_results)
    stage_id = args.stage_id.strip()
    focus = df[df["stage_id"].astype(str) == stage_id].copy()
    if focus.empty:
        raise ValueError(f"No rows for stage_id={stage_id}")

    focus["correct_bool"] = bool_series(focus["correct"]) if "correct" in focus else False
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    name = args.name.strip() or stage_id

    focus_csv = output_dir / f"{name}_rows.csv"
    focus.to_csv(focus_csv, index=False, encoding="utf-8-sig")

    total = len(focus)
    correct = int(focus["correct_bool"].sum())
    accuracy = correct / total if total else 0.0
    pred_counter = Counter(focus["predicted_stage"].astype(str))
    anther_counter = Counter(focus.get("anther_visible", pd.Series(dtype=str)).astype(str))
    spike_counter = Counter(focus.get("spike_status", pd.Series(dtype=str)).astype(str))
    source_counter = Counter(focus.get("source_type", pd.Series(dtype=str)).astype(str))

    example_columns = [
        "image_id",
        "stage_id",
        "predicted_stage",
        "second_candidate",
        "correct",
        "confidence",
        "spike_status",
        "anther_visible",
        "short_reason",
        "uncertainty_reason",
        "image_path",
    ]
    available_example_columns = [col for col in example_columns if col in focus.columns]
    errors = focus[~focus["correct_bool"]].copy()
    examples = errors[available_example_columns].head(20)
    examples_csv = output_dir / f"{name}_error_examples.csv"
    examples.to_csv(examples_csv, index=False, encoding="utf-8-sig")

    summary_rows = [
        {"metric": "total", "value": total},
        {"metric": "correct", "value": correct},
        {"metric": "accuracy", "value": f"{accuracy:.4f}"},
    ]
    markdown = f"""# VLM Stage Focus Summary: {stage_id}

## Input

- parsed_results: `{args.parsed_results}`
- stage_id: `{stage_id}`
- rows_csv: `{focus_csv}`
- error_examples_csv: `{examples_csv}`

## Overall

{md_table(summary_rows, ["metric", "value"])}

## Predicted Stage Distribution

{md_table(counter_rows(pred_counter, "predicted_stage"), ["predicted_stage", "count"])}

## Anther Visibility Distribution

{md_table(counter_rows(anther_counter, "anther_visible"), ["anther_visible", "count"])}

## Spike Status Distribution

{md_table(counter_rows(spike_counter, "spike_status"), ["spike_status", "count"])}

## Source Distribution

{md_table(counter_rows(source_counter, "source_type"), ["source_type", "count"])}

## First 20 Error Examples

{examples[available_example_columns].to_markdown(index=False) if not examples.empty else "_No errors._"}
"""
    summary_path = output_dir / f"{name}_summary.md"
    summary_path.write_text(markdown, encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
