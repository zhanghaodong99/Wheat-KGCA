from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


BASELINE_REL = Path("convnext_features") / "full_test_2485" / "convnext_feature_metrics.json"


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


def collect_run(run_dir: Path, baseline_accuracy: float) -> dict[str, Any] | None:
    metrics_path = run_dir / "image_level_wheat_sage_metrics.json"
    metrics = read_json(metrics_path)
    if metrics is None:
        return None
    test = metrics.get("test", {})
    best_val = metrics.get("best_val", {})
    visual_accuracy = test.get("visual_accuracy")
    fused_accuracy = test.get("fused_accuracy")
    row = {
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "freeze_visual": metrics.get("freeze_visual"),
        "visual_accuracy": visual_accuracy,
        "fused_accuracy": fused_accuracy,
        "fused_macro_f1": test.get("fused_macro_f1"),
        "fused_balanced_accuracy": test.get("fused_balanced_accuracy"),
        "raw_gate_mean": test.get("raw_gate_mean"),
        "effective_gate_mean": test.get("effective_gate_mean"),
        "eligibility_mean": test.get("eligibility_mean"),
        "switch_count": test.get("switch_count"),
        "corrected_count": test.get("corrected_count"),
        "introduced_error_count": test.get("introduced_error_count"),
        "wrong_to_wrong_switch_count": test.get("wrong_to_wrong_switch_count"),
        "net_correct_change_vs_visual_branch": test.get("net_correct_change_vs_visual_branch"),
        "val_fused_accuracy": best_val.get("fused_accuracy"),
        "val_fused_macro_f1": best_val.get("fused_macro_f1"),
    }
    row["fused_accuracy_percent"] = None if fused_accuracy is None else float(fused_accuracy) * 100.0
    row["change_vs_original_convnext"] = None if fused_accuracy is None else float(fused_accuracy) - baseline_accuracy
    row["change_vs_original_convnext_pp"] = None if row["change_vs_original_convnext"] is None else row["change_vs_original_convnext"] * 100.0
    row["change_vs_visual_branch"] = (
        None if fused_accuracy is None or visual_accuracy is None else float(fused_accuracy) - float(visual_accuracy)
    )
    row["change_vs_visual_branch_pp"] = None if row["change_vs_visual_branch"] is None else row["change_vs_visual_branch"] * 100.0
    return row


def write_markdown(out_md: Path, summary: pd.DataFrame, baseline_accuracy: float) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Image-level Wheat-SAGE Ablation Summary",
        "",
        f"Original ConvNeXt full test baseline: `{baseline_accuracy:.9f}` ({baseline_accuracy * 100:.3f}%).",
        "",
    ]
    if summary.empty:
        lines += ["No completed image-level ablation runs were found.", ""]
    else:
        display_cols = [
            "run",
            "freeze_visual",
            "fused_accuracy_percent",
            "change_vs_original_convnext_pp",
            "change_vs_visual_branch_pp",
            "fused_macro_f1",
            "switch_count",
            "corrected_count",
            "introduced_error_count",
            "effective_gate_mean",
        ]
        table = summary[[col for col in display_cols if col in summary.columns]].copy()
        for col in [
            "fused_accuracy_percent",
            "change_vs_original_convnext_pp",
            "change_vs_visual_branch_pp",
            "fused_macro_f1",
            "effective_gate_mean",
        ]:
            if col in table.columns:
                table[col] = table[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.6f}")
        lines += ["## Completed Runs", "", dataframe_to_markdown(table), ""]
        best = summary.dropna(subset=["fused_accuracy"]).sort_values(
            ["fused_accuracy", "fused_macro_f1"], ascending=False
        ).iloc[0]
        lines += [
            "## Current Best",
            "",
            f"- run: `{best['run']}`",
            f"- fused accuracy: `{float(best['fused_accuracy']) * 100:.3f}%`",
            f"- change vs original ConvNeXt: `{float(best['change_vs_original_convnext_pp']):+.3f} pp`",
            "",
        ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize image-level Wheat-SAGE ablation outputs.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = read_json(args.root / BASELINE_REL)
    if baseline is None:
        raise FileNotFoundError(args.root / BASELINE_REL)
    baseline_accuracy = float(baseline["accuracy"])
    candidate_dirs = sorted(
        [
            *args.root.glob("image_ablation_*"),
            *args.root.glob("image_level_wheat_sage"),
            *args.root.glob("image_level_wheat_sage_smoke"),
        ]
    )
    candidate_dirs = [path for path in candidate_dirs if "micro" not in path.name and (path / "image_level_wheat_sage_metrics.json").exists()]
    rows = []
    for run_dir in candidate_dirs:
        row = collect_run(run_dir, baseline_accuracy)
        if row is not None:
            rows.append(row)
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(["fused_accuracy", "fused_macro_f1"], ascending=False, na_position="last")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    write_markdown(args.out_md, summary, baseline_accuracy)
    print(f"baseline_accuracy={baseline_accuracy:.9f}")
    print(f"completed_runs={len(summary)}")
    if not summary.empty:
        print(
            summary[
                [
                    "run",
                    "fused_accuracy",
                    "change_vs_original_convnext_pp",
                    "change_vs_visual_branch_pp",
                    "fused_macro_f1",
                ]
            ].to_string(index=False)
        )
    print(f"csv={args.out_csv}")
    print(f"md={args.out_md}")


if __name__ == "__main__":
    main()
