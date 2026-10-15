from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASELINE_REL = Path("convnext_features") / "full_test_2485" / "convnext_feature_metrics.json"
BASELINE_PRED_REL = Path("convnext_features") / "full_test_2485" / "convnext_predictions.csv"


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value) * 100.0


def read_baseline(root: Path) -> tuple[float, pd.DataFrame | None]:
    metrics = read_json(root / BASELINE_REL)
    if metrics is None:
        raise FileNotFoundError(root / BASELINE_REL)
    baseline_accuracy = float(metrics["accuracy"])
    pred_path = root / BASELINE_PRED_REL
    if pred_path.exists():
        base_pred = pd.read_csv(pred_path)
    else:
        base_pred = None
    return baseline_accuracy, base_pred


def movement_from_predictions(base_pred: pd.DataFrame | None, pred_path: Path) -> dict[str, int | None]:
    empty = {
        "switch_count": None,
        "corrected_count": None,
        "introduced_error_count": None,
        "wrong_to_wrong_switch_count": None,
        "net_correct_change": None,
    }
    if base_pred is None or not pred_path.exists():
        return empty
    pred = pd.read_csv(pred_path)
    if "pred_label" not in pred.columns:
        return empty
    base = base_pred[["image_key", "stage_id", "pred_label", "correct"]].rename(
        columns={
            "stage_id": "true_label_base",
            "pred_label": "convnext_pred",
            "correct": "convnext_correct",
        }
    )
    merged = base.merge(pred, on="image_key", how="inner", suffixes=("", "_fusion"))
    if len(merged) == 0:
        return empty
    if "true_label" in merged.columns:
        true_label = merged["true_label"]
    else:
        true_label = merged["true_label_base"]
    fusion_correct = merged["pred_label"] == true_label
    conv_correct = merged["convnext_correct"].astype(bool)
    switched = merged["convnext_pred"] != merged["pred_label"]
    return {
        "switch_count": int(switched.sum()),
        "corrected_count": int((switched & ~conv_correct & fusion_correct).sum()),
        "introduced_error_count": int((switched & conv_correct & ~fusion_correct).sum()),
        "wrong_to_wrong_switch_count": int((switched & ~conv_correct & ~fusion_correct).sum()),
        "net_correct_change": int(fusion_correct.sum() - conv_correct.sum()),
    }


def add_row(rows: list[dict[str, Any]], baseline_accuracy: float, **kwargs: Any) -> None:
    accuracy = kwargs.get("test_accuracy")
    kwargs["test_accuracy_percent"] = pct(accuracy)
    kwargs["accuracy_change_vs_convnext"] = None if accuracy is None else float(accuracy) - baseline_accuracy
    kwargs["accuracy_change_pp"] = pct(kwargs["accuracy_change_vs_convnext"])
    rows.append(kwargs)


def collect_stage1(root: Path, baseline_accuracy: float, rows: list[dict[str, Any]]) -> None:
    metrics_path = root / "stage1_fusion_head" / "fusion_head_metrics.csv"
    if not metrics_path.exists():
        return
    frame = pd.read_csv(metrics_path)
    for _, row in frame.iterrows():
        add_row(
            rows,
            baseline_accuracy,
            family="stage1_fusion_head",
            run_dir=str(metrics_path.parent),
            method=str(row["method"]),
            status="done",
            test_accuracy=float(row["test_accuracy"]),
            test_macro_f1=float(row["test_macro_f1"]),
            test_balanced_accuracy=float(row["test_balanced_accuracy"]),
            gate_mean=None,
            switch_count=None,
            corrected_count=None,
            introduced_error_count=None,
            wrong_to_wrong_switch_count=None,
            net_correct_change=None,
            notes="Full Stage1 shallow ablation.",
        )


def collect_stage6(root: Path, baseline_accuracy: float, rows: list[dict[str, Any]]) -> None:
    for run_dir in sorted(root.glob("stage6_conservative_gate*")):
        metrics = read_json(run_dir / "metrics.json")
        if metrics is None:
            continue
        gated = metrics.get("gated_stage1_fusion_test", {})
        raw = metrics.get("raw_stage1_fusion_test", {})
        visual = metrics.get("visual_test", {})
        add_row(
            rows,
            baseline_accuracy,
            family="stage6_gate",
            run_dir=str(run_dir),
            method=f"{run_dir.name}: raw_stage1_fusion",
            status="done",
            test_accuracy=raw.get("accuracy"),
            test_macro_f1=raw.get("macro_f1"),
            test_balanced_accuracy=raw.get("balanced_accuracy"),
            gate_mean=None,
            switch_count=raw.get("switch_count"),
            corrected_count=raw.get("corrected_count"),
            introduced_error_count=raw.get("introduced_error_count"),
            wrong_to_wrong_switch_count=raw.get("wrong_to_wrong_switch_count"),
            net_correct_change=raw.get("net_correct_change"),
            notes=f"Raw Stage1 fusion before conservative gate; visual_acc={visual.get('accuracy')}.",
        )
        add_row(
            rows,
            baseline_accuracy,
            family="stage6_gate",
            run_dir=str(run_dir),
            method=f"{run_dir.name}: gated_stage1_fusion",
            status="done",
            test_accuracy=gated.get("accuracy"),
            test_macro_f1=gated.get("macro_f1"),
            test_balanced_accuracy=gated.get("balanced_accuracy"),
            gate_mean=None,
            switch_count=gated.get("switch_count"),
            corrected_count=gated.get("corrected_count"),
            introduced_error_count=gated.get("introduced_error_count"),
            wrong_to_wrong_switch_count=gated.get("wrong_to_wrong_switch_count"),
            net_correct_change=gated.get("net_correct_change"),
            notes=f"Gate={metrics.get('best_gate')}, C={metrics.get('c')}.",
        )


def collect_semantic_modules(root: Path, baseline_accuracy: float, base_pred: pd.DataFrame | None, rows: list[dict[str, Any]]) -> None:
    original = read_json(root / "semantic_visual_module" / "semantic_visual_module_metrics.json")
    if original is not None:
        move = movement_from_predictions(base_pred, root / "semantic_visual_module" / "test_predictions.csv")
        add_row(
            rows,
            baseline_accuracy,
            family="semantic_visual_module",
            run_dir=str(root / "semantic_visual_module"),
            method="06_original_film_gated_residual",
            status="done",
            test_accuracy=original.get("test_accuracy"),
            test_macro_f1=original.get("test_macro_f1"),
            test_balanced_accuracy=original.get("test_balanced_accuracy"),
            gate_mean=original.get("test_gate_mean"),
            **move,
            notes="Original 06. Known issue: gate almost fully open if gate_mean is near 1.",
        )

    for run_dir in sorted(root.glob("semantic_visual_module_v2*")):
        metrics = read_json(run_dir / "semantic_visual_module_v2_metrics.json")
        if metrics is None:
            continue
        test = metrics.get("wheat_sage_v2_test", {})
        add_row(
            rows,
            baseline_accuracy,
            family="wheat_sage_v2",
            run_dir=str(run_dir),
            method=run_dir.name,
            status="done",
            test_accuracy=test.get("accuracy"),
            test_macro_f1=test.get("macro_f1"),
            test_balanced_accuracy=test.get("balanced_accuracy"),
            gate_mean=metrics.get("test_effective_gate_mean"),
            raw_gate_mean=metrics.get("test_raw_gate_mean"),
            eligibility_mean=metrics.get("test_eligibility_mean"),
            switch_count=test.get("switch_count"),
            corrected_count=test.get("corrected_count"),
            introduced_error_count=test.get("introduced_error_count"),
            wrong_to_wrong_switch_count=test.get("wrong_to_wrong_switch_count"),
            net_correct_change=test.get("net_correct_change"),
            notes="Conservative semantic residual module.",
        )


def write_markdown(out_md: Path, summary: pd.DataFrame, baseline_accuracy: float) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Full Fusion Search Summary 2026-07-24",
        "",
        f"ConvNeXt full test baseline accuracy: `{baseline_accuracy:.9f}` ({baseline_accuracy * 100:.3f}%).",
        "",
    ]
    if summary.empty:
        lines += [
            "No completed fusion experiments were found.",
            "",
            "Run scripts under `scripts\\paper_full_experiments\\full_search` first.",
        ]
    else:
        display_cols = [
            "method",
            "test_accuracy_percent",
            "accuracy_change_pp",
            "test_macro_f1",
            "switch_count",
            "corrected_count",
            "introduced_error_count",
            "net_correct_change",
            "gate_mean",
        ]
        available = [col for col in display_cols if col in summary.columns]
        table = summary[available].copy()
        for col in ["test_accuracy_percent", "accuracy_change_pp", "test_macro_f1", "gate_mean"]:
            if col in table.columns:
                table[col] = table[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.6f}")
        lines += ["## Completed Results", "", dataframe_to_markdown(table), ""]
        best = summary.dropna(subset=["test_accuracy"]).sort_values(
            ["test_accuracy", "test_macro_f1"], ascending=False
        ).iloc[0]
        lines += [
            "## Current Best",
            "",
            f"- method: `{best['method']}`",
            f"- test accuracy: `{float(best['test_accuracy']) * 100:.3f}%`",
            f"- change vs ConvNeXt: `{float(best['accuracy_change_pp']):+.3f} pp`",
            "",
        ]
        improved = summary[summary["accuracy_change_vs_convnext"].fillna(-1.0) > 0]
        if len(improved):
            lines += [
                "## Improved Methods",
                "",
                "At least one completed method improved over the ConvNeXt full baseline.",
                "",
            ]
        else:
            lines += [
                "## Improved Methods",
                "",
                "No completed method has improved over the ConvNeXt full baseline yet.",
                "",
            ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    columns = [str(col) for col in frame.columns]
    rows = []
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in frame.iterrows():
        values = ["" if pd.isna(row[col]) else str(row[col]) for col in frame.columns]
        rows.append("| " + " | ".join(value.replace("|", "/") for value in values) + " |")
    return "\n".join(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize full-data fusion search outputs.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_accuracy, base_pred = read_baseline(args.root)
    rows: list[dict[str, Any]] = []
    collect_stage1(args.root, baseline_accuracy, rows)
    collect_stage6(args.root, baseline_accuracy, rows)
    collect_semantic_modules(args.root, baseline_accuracy, base_pred, rows)
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(["test_accuracy", "test_macro_f1"], ascending=False, na_position="last")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    write_markdown(args.out_md, summary, baseline_accuracy)
    print(f"baseline_accuracy={baseline_accuracy:.9f}")
    print(f"completed_methods={len(summary)}")
    if not summary.empty:
        print(summary[["method", "test_accuracy", "accuracy_change_pp", "test_macro_f1"]].to_string(index=False))
    print(f"csv={args.out_csv}")
    print(f"md={args.out_md}")


if __name__ == "__main__":
    main()
