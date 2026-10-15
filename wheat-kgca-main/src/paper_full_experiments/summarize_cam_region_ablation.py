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
    lines = [
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    for _, row in frame.iterrows():
        values = ["" if pd.isna(row[col]) else str(row[col]) for col in frame.columns]
        lines.append("| " + " | ".join(value.replace("|", "/") for value in values) + " |")
    return "\n".join(lines)


def collect_run(run_dir: Path) -> dict[str, Any] | None:
    metrics = read_json(run_dir / "metrics.json")
    if metrics is None:
        return None
    baseline = metrics.get("baseline_visual_test", {})
    final = metrics.get("final_fusion_test", {})
    row = {
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "lambda_align": metrics.get("lambda_align"),
        "freeze_encoder": metrics.get("freeze_encoder"),
        "active_train_priors": metrics.get("active_train_priors"),
        "active_val_priors_for_audit": metrics.get("active_val_priors_for_audit"),
        "baseline_accuracy": baseline.get("accuracy"),
        "final_accuracy": final.get("accuracy"),
        "baseline_macro_f1": baseline.get("macro_f1"),
        "final_macro_f1": final.get("macro_f1"),
        "baseline_balanced_accuracy": baseline.get("balanced_accuracy"),
        "final_balanced_accuracy": final.get("balanced_accuracy"),
        "test_gate_mean": metrics.get("test_gate_mean"),
        "best_val_macro_f1": metrics.get("best_val_macro_f1"),
    }
    if row["final_accuracy"] is not None and row["baseline_accuracy"] is not None:
        row["accuracy_change_vs_visual"] = float(row["final_accuracy"]) - float(row["baseline_accuracy"])
        row["accuracy_change_vs_visual_pp"] = row["accuracy_change_vs_visual"] * 100.0
    else:
        row["accuracy_change_vs_visual"] = None
        row["accuracy_change_vs_visual_pp"] = None
    return row


def write_markdown(path: Path, summary: pd.DataFrame, title: str, note: str) -> None:
    lines = [
        f"# {title}",
        "",
        note,
        "",
    ]
    if summary.empty:
        lines.append("No completed CAM-region runs found.")
    else:
        display = summary[
            [
                "run",
                "lambda_align",
                "active_train_priors",
                "final_accuracy",
                "accuracy_change_vs_visual_pp",
                "final_macro_f1",
                "test_gate_mean",
            ]
        ].copy()
        for col in ["final_accuracy", "accuracy_change_vs_visual_pp", "final_macro_f1", "test_gate_mean"]:
            display[col] = display[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.6f}")
        lines.append(dataframe_to_markdown(display))
        lines.append("")
        best = summary.dropna(subset=["final_accuracy"]).sort_values(
            ["final_accuracy", "final_macro_f1"], ascending=False
        ).iloc[0]
        lines.extend(
            [
                "## Current Best",
                "",
                f"- run: `{best['run']}`",
                f"- final accuracy: `{float(best['final_accuracy']) * 100:.3f}%`",
                f"- change vs visual branch: `{float(best['accuracy_change_vs_visual_pp']):+.3f} pp`",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize CAM-region subset-evidence ablations.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pattern", type=str, default="cam_region_ablation_*")
    parser.add_argument("--title", type=str, default="CAM-Region Ablation Summary")
    parser.add_argument(
        "--note",
        type=str,
        default="These runs use full train/val/test image-level training, but CAM-region alignment is activated only on the available subset of VLM regional evidence.",
    )
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for run_dir in sorted(args.root.glob(args.pattern)):
        row = collect_run(run_dir)
        if row is not None:
            rows.append(row)
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(["final_accuracy", "final_macro_f1"], ascending=False, na_position="last")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    write_markdown(args.out_md, summary, args.title, args.note)
    print(f"completed_runs={len(summary)}")
    if not summary.empty:
        print(summary[["run", "lambda_align", "final_accuracy", "accuracy_change_vs_visual_pp"]].to_string(index=False))
    print(f"csv={args.out_csv}")
    print(f"md={args.out_md}")


if __name__ == "__main__":
    main()
