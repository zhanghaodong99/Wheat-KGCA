from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    columns = [str(col) for col in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        values = ["" if pd.isna(row[col]) else str(row[col]) for col in frame.columns]
        lines.append("| " + " | ".join(value.replace("|", "/") for value in values) + " |")
    return "\n".join(lines)


def collect_run(run_dir: Path) -> dict[str, Any] | None:
    metrics = read_json(run_dir / "metrics.json")
    if metrics is None:
        return None
    final = metrics.get("final_fusion_test", {})
    baseline = metrics.get("baseline_visual_test", {})
    return {
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "lambda_align": metrics.get("lambda_align"),
        "lambda_sem": metrics.get("lambda_sem", 0.0),
        "semantic_temperature": metrics.get("semantic_temperature"),
        "accuracy": final.get("accuracy"),
        "accuracy_percent": None if final.get("accuracy") is None else float(final["accuracy"]) * 100.0,
        "accuracy_change_vs_visual_pp": metrics.get("test_accuracy_change_vs_visual", 0.0) * 100.0,
        "macro_f1": final.get("macro_f1"),
        "macro_f1_change_vs_visual": metrics.get("test_macro_f1_change_vs_visual"),
        "balanced_accuracy": final.get("balanced_accuracy"),
        "baseline_accuracy": baseline.get("accuracy"),
        "test_gate_mean": metrics.get("test_gate_mean"),
        "best_val_macro_f1": metrics.get("best_val_macro_f1"),
    }


def write_markdown(out_md: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# Semantic Anchor Ablation Summary",
        "",
        "These runs add class semantic-anchor alignment loss `L_sem` to the discrepancy-gated CAM fusion code.",
        "",
    ]
    if summary.empty:
        lines += ["No completed semantic-anchor runs found.", ""]
    else:
        table = summary[
            [
                "run",
                "lambda_align",
                "lambda_sem",
                "accuracy_percent",
                "accuracy_change_vs_visual_pp",
                "macro_f1",
                "test_gate_mean",
                "best_val_macro_f1",
            ]
        ].copy()
        for col in [
            "lambda_align",
            "lambda_sem",
            "accuracy_percent",
            "accuracy_change_vs_visual_pp",
            "macro_f1",
            "test_gate_mean",
            "best_val_macro_f1",
        ]:
            table[col] = table[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.6f}")
        lines += [dataframe_to_markdown(table), ""]
        best = summary.dropna(subset=["accuracy"]).sort_values(
            ["accuracy", "macro_f1", "best_val_macro_f1"], ascending=False
        ).iloc[0]
        lines += [
            "## Current Best",
            "",
            f"- run: `{best['run']}`",
            f"- accuracy: `{float(best['accuracy']) * 100:.3f}%`",
            f"- change vs visual: `{float(best['accuracy_change_vs_visual_pp']):+.3f} pp`",
            "",
        ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize semantic-anchor full-data ablations.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = sorted(args.root.glob("expert_spatial_semantic_anchor_*"))
    rows = []
    for run_dir in run_dirs:
        row = collect_run(run_dir)
        if row is not None:
            rows.append(row)
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(["accuracy", "macro_f1", "best_val_macro_f1"], ascending=False)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    write_markdown(args.out_md, summary)
    print(f"completed_runs={len(summary)}")
    if not summary.empty:
        print(summary[["run", "lambda_align", "lambda_sem", "accuracy", "accuracy_change_vs_visual_pp"]].to_string(index=False))
    print(f"csv={args.out_csv}")
    print(f"md={args.out_md}")


if __name__ == "__main__":
    main()
