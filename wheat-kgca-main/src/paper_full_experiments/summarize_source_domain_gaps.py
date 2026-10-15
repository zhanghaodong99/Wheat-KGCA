from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPARISON = PROJECT_ROOT / "outputs" / "paper_full_experiments" / "error_diagnosis" / "all_model_comparison.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "paper_full_experiments" / "source_domain_analysis"
MODELS = ["baseline", "wheat_sage_cam", "attention15"]


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


def summarize(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        correct_col = f"{model}_correct"
        if correct_col not in comparison.columns:
            continue
        for (stage, source), group in comparison.groupby(["stage_id", "source_type"], dropna=False):
            correct = int(group[correct_col].astype(str).str.lower().isin(["true", "1"]).sum())
            rows.append(
                {
                    "model": model,
                    "stage_id": stage,
                    "source_type": source,
                    "samples": len(group),
                    "correct": correct,
                    "errors": len(group) - correct,
                    "accuracy": correct / len(group),
                }
            )
    return pd.DataFrame(rows)


def build_gap_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in summary["model"].unique():
        subset = summary[summary["model"] == model]
        pivot = subset.pivot(index="stage_id", columns="source_type", values="accuracy").reset_index()
        for col in ["experiment_tong", "large_scale"]:
            if col not in pivot.columns:
                pivot[col] = pd.NA
        pivot["model"] = model
        pivot["experiment_tong_minus_large_scale"] = pivot["experiment_tong"] - pivot["large_scale"]
        rows.append(pivot[["model", "stage_id", "experiment_tong", "large_scale", "experiment_tong_minus_large_scale"]])
    return pd.concat(rows, ignore_index=True).sort_values(["experiment_tong_minus_large_scale", "model"])


def write_markdown(out_path: Path, summary: pd.DataFrame, gaps: pd.DataFrame) -> None:
    lines = [
        "# Source Domain Analysis",
        "",
        "Negative gap means `experiment_tong` is worse than `large_scale`.",
        "",
        "## Largest Source Gaps",
        "",
        dataframe_to_markdown(gaps.head(30)),
        "",
        "## Stage-Source Accuracy",
        "",
        dataframe_to_markdown(summary.sort_values(["accuracy", "model"]).head(40)),
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize source/domain performance gaps.")
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    comparison = pd.read_csv(args.comparison)
    summary = summarize(comparison)
    gaps = build_gap_table(summary)
    summary.to_csv(args.out_dir / "stage_source_accuracy.csv", index=False, encoding="utf-8-sig")
    gaps.to_csv(args.out_dir / "source_gap_by_stage.csv", index=False, encoding="utf-8-sig")
    write_markdown(args.out_dir / "source_domain_summary.md", summary, gaps)
    print(f"out_dir={args.out_dir}")
    print(gaps.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
