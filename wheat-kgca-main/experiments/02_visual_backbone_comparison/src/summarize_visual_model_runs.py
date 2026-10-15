from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "src_model_training"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "data_split_811_full" / "wheat_split_811_manifest.csv"
DEFAULT_ANALYSIS_ROOT = PROJECT_ROOT / "_analysis_outputs" / "visual_model_supplementary_analysis"

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics")


@dataclass(frozen=True)
class ModelRun:
    model: str
    paper_name: str
    run_dir: Path
    model_dir: Path
    metrics_path: Path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_path_text(value: str) -> str:
    return str(Path(value)).replace("/", "\\").lower()


def discover_runs(paths: list[Path], output_root: Path) -> list[ModelRun]:
    if paths:
        candidates = [path.resolve() for path in paths]
    else:
        candidates = sorted(
            [path for path in output_root.iterdir() if path.is_dir()],
            key=lambda path: path.stat().st_mtime,
        )

    runs: list[ModelRun] = []
    seen_model_dirs: set[Path] = set()
    for candidate in candidates:
        for metrics_path in candidate.glob("*/metrics_full.json"):
            model_dir = metrics_path.parent.resolve()
            if model_dir in seen_model_dirs:
                continue
            seen_model_dirs.add(model_dir)
            metrics = read_json(metrics_path)
            runs.append(
                ModelRun(
                    model=str(metrics.get("model", model_dir.name)),
                    paper_name=str(metrics.get("paper_name", metrics.get("model", model_dir.name))),
                    run_dir=candidate.resolve(),
                    model_dir=model_dir,
                    metrics_path=metrics_path.resolve(),
                )
            )
    return runs


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    manifest["target_norm"] = manifest["target_path"].map(normalize_path_text)
    manifest["target_name"] = manifest["target_path"].map(lambda value: Path(str(value)).name.lower())
    manifest["source_norm"] = manifest["source_path"].map(normalize_path_text)
    return manifest


def attach_manifest(predictions: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    predictions = predictions.copy()
    predictions["image_norm"] = predictions["image_path"].map(normalize_path_text)
    predictions["target_name"] = predictions["image_path"].map(lambda value: Path(str(value)).name.lower())

    merged = predictions.merge(
        manifest[
            [
                "split",
                "stage_id",
                "source_type",
                "source_dir",
                "source_path",
                "target_path",
                "target_norm",
                "target_name",
                "original_filename",
            ]
        ],
        left_on="image_norm",
        right_on="target_norm",
        how="left",
        suffixes=("", "_manifest"),
    )
    missing = merged["source_type"].isna()
    if missing.any():
        fallback = predictions.loc[missing].merge(
            manifest[
                [
                    "split",
                    "stage_id",
                    "source_type",
                    "source_dir",
                    "source_path",
                    "target_path",
                    "target_name",
                    "original_filename",
                ]
            ],
            on="target_name",
            how="left",
            suffixes=("", "_manifest"),
        )
        for column in fallback.columns:
            if column in merged.columns:
                merged.loc[missing, column] = fallback[column].to_numpy()
    return merged


def metric_row(frame: pd.DataFrame, prefix: str = "") -> dict[str, Any]:
    y_true = frame["true_label"].astype(str).tolist()
    y_pred = frame["pred_label"].astype(str).tolist()
    labels = sorted(set(y_true) | set(y_pred))
    return {
        f"{prefix}samples": int(len(frame)),
        f"{prefix}accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        f"{prefix}accuracy_percent": round(float(accuracy_score(y_true, y_pred) * 100), 2),
        f"{prefix}macro_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 6),
        f"{prefix}weighted_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)), 6),
        f"{prefix}balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 6),
    }


def summarize_metrics(runs: list[ModelRun]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        metrics = read_json(run.metrics_path)
        rows.append(
            {
                "model": run.model,
                "paper_name": run.paper_name,
                "run_dir": str(run.run_dir),
                "model_dir": str(run.model_dir),
                "params_M": metrics.get("params_M"),
                "best_epoch": metrics.get("best_epoch"),
                "best_val_macro_f1": metrics.get("best_selection_value"),
                "train_time_s": metrics.get("train_time_s"),
                "peak_gpu_memory_mb": metrics.get("peak_gpu_memory_mb"),
                "clean_accuracy_percent": metrics.get("clean", {}).get("accuracy_percent"),
                "clean_macro_f1": metrics.get("clean", {}).get("macro_f1"),
                "clean_weighted_f1": metrics.get("clean", {}).get("weighted_f1"),
                "clean_balanced_accuracy": metrics.get("clean", {}).get("balanced_accuracy"),
                "noise_accuracy_percent": metrics.get("noise", {}).get("accuracy_percent"),
                "noise_macro_f1": metrics.get("noise", {}).get("macro_f1"),
                "noise_weighted_f1": metrics.get("noise", {}).get("weighted_f1"),
                "noise_balanced_accuracy": metrics.get("noise", {}).get("balanced_accuracy"),
                "accuracy_drop_percent": metrics.get("accuracy_drop_percent"),
            }
        )
    rows.sort(key=lambda row: (row.get("clean_accuracy_percent") or 0), reverse=True)
    return rows


def load_predictions(run: ModelRun, split_name: str, manifest: pd.DataFrame) -> pd.DataFrame:
    path = run.model_dir / f"test_predictions_{split_name}.csv"
    data = pd.read_csv(path)
    data["model"] = run.model
    data["paper_name"] = run.paper_name
    data["eval_set"] = split_name
    data = attach_manifest(data, manifest)
    return data


def summarize_by_source(runs: list[ModelRun], manifest: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        for eval_set in ["clean", "noise"]:
            predictions = load_predictions(run, eval_set, manifest)
            for source_type, group in predictions.groupby("source_type", dropna=False):
                row = {
                    "model": run.model,
                    "paper_name": run.paper_name,
                    "eval_set": eval_set,
                    "source_type": source_type if pd.notna(source_type) else "UNKNOWN",
                }
                row.update(metric_row(group))
                rows.append(row)
            for (source_type, true_label), group in predictions.groupby(["source_type", "true_label"], dropna=False):
                row = {
                    "model": run.model,
                    "paper_name": run.paper_name,
                    "eval_set": eval_set,
                    "source_type": source_type if pd.notna(source_type) else "UNKNOWN",
                    "true_label": true_label,
                }
                row.update(metric_row(group))
                rows.append(row)
    return rows


def per_class_rows(runs: list[ModelRun]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        metrics = read_json(run.metrics_path)
        for eval_set in ["clean", "noise"]:
            report_path = run.model_dir / f"classification_report_{eval_set}.json"
            if report_path.exists():
                report = read_json(report_path)
            else:
                predictions = pd.read_csv(run.model_dir / f"test_predictions_{eval_set}.csv")
                report = {}
                for label in sorted(predictions["true_label"].astype(str).unique()):
                    subset_true = predictions["true_label"].astype(str).tolist()
                    subset_pred = predictions["pred_label"].astype(str).tolist()
                    report[label] = {"f1-score": f1_score(subset_true, subset_pred, labels=[label], average="macro")}
            for label, values in report.items():
                if label in {"accuracy", "macro avg", "weighted avg"}:
                    continue
                rows.append(
                    {
                        "model": run.model,
                        "paper_name": run.paper_name,
                        "eval_set": eval_set,
                        "class": label,
                        "precision": round(float(values.get("precision", 0)), 6),
                        "recall": round(float(values.get("recall", 0)), 6),
                        "f1": round(float(values.get("f1-score", 0)), 6),
                        "support": int(values.get("support", 0)),
                        "params_M": metrics.get("params_M"),
                    }
                )
    return rows


def mcnemar_p_value(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    # Continuity-corrected chi-square with df=1.
    statistic = (abs(b - c) - 1) ** 2 / n
    return float(math.erfc(math.sqrt(statistic / 2)))


def significance_rows(
    runs: list[ModelRun],
    manifest: pd.DataFrame,
    bootstrap_iterations: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prediction_cache: dict[tuple[str, str], pd.DataFrame] = {}
    for run in runs:
        for eval_set in ["clean", "noise"]:
            prediction_cache[(run.model, eval_set)] = load_predictions(run, eval_set, manifest)

    mcnemar_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for eval_set in ["clean", "noise"]:
        for i, run_a in enumerate(runs):
            for run_b in runs[i + 1 :]:
                a = prediction_cache[(run_a.model, eval_set)]
                b = prediction_cache[(run_b.model, eval_set)]
                a = a[["target_name", "true_label", "pred_label", "correct"]].rename(
                    columns={"pred_label": "pred_a", "correct": "correct_a"}
                )
                b = b[["target_name", "pred_label", "correct"]].rename(
                    columns={"pred_label": "pred_b", "correct": "correct_b"}
                )
                paired = a.merge(b, on="target_name", how="inner")
                correct_a = paired["correct_a"].astype(int).to_numpy()
                correct_b = paired["correct_b"].astype(int).to_numpy()
                b_count = int(((correct_a == 1) & (correct_b == 0)).sum())
                c_count = int(((correct_a == 0) & (correct_b == 1)).sum())
                mcnemar_rows.append(
                    {
                        "eval_set": eval_set,
                        "model_a": run_a.model,
                        "model_b": run_b.model,
                        "paired_samples": int(len(paired)),
                        "a_correct_b_wrong": b_count,
                        "a_wrong_b_correct": c_count,
                        "accuracy_a": round(float(correct_a.mean()), 6),
                        "accuracy_b": round(float(correct_b.mean()), 6),
                        "accuracy_diff_a_minus_b": round(float(correct_a.mean() - correct_b.mean()), 6),
                        "mcnemar_p_value_approx": round(mcnemar_p_value(b_count, c_count), 8),
                    }
                )
                if bootstrap_iterations > 0 and len(paired) > 0:
                    diffs: list[float] = []
                    indices = np.arange(len(paired))
                    for _ in range(bootstrap_iterations):
                        sample = rng.choice(indices, size=len(indices), replace=True)
                        diffs.append(float(correct_a[sample].mean() - correct_b[sample].mean()))
                    bootstrap_rows.append(
                        {
                            "eval_set": eval_set,
                            "model_a": run_a.model,
                            "model_b": run_b.model,
                            "iterations": bootstrap_iterations,
                            "accuracy_diff_a_minus_b": round(float(correct_a.mean() - correct_b.mean()), 6),
                            "ci95_low": round(float(np.percentile(diffs, 2.5)), 6),
                            "ci95_high": round(float(np.percentile(diffs, 97.5)), 6),
                        }
                    )
    return mcnemar_rows, bootstrap_rows


def exact_duplicate_audit(manifest: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in ["source_path", "original_filename", "target_path"]:
        duplicated = manifest[manifest.duplicated(column, keep=False)].copy()
        rows.append(
            {
                "audit": f"exact_duplicate_{column}",
                "duplicate_rows": int(len(duplicated)),
                "duplicate_groups": int(duplicated[column].nunique()) if len(duplicated) else 0,
                "note": "Exact duplicate rows across manifest.",
            }
        )
    return rows


def average_hash(path: Path, hash_size: int) -> int:
    image = Image.open(path).convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(image, dtype=np.float32)
    avg = float(pixels.mean())
    bits = pixels > avg
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming_distance(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def near_duplicate_audit(
    manifest: pd.DataFrame,
    project_root: Path,
    max_hamming: int,
    hash_size: int,
    max_pairs_per_group: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    test_rows = manifest[manifest["split"].isin(["train", "val", "test"])].copy()
    hashes: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for _, row in test_rows.iterrows():
        target_path = project_root / str(row["target_path"])
        if not target_path.exists():
            continue
        try:
            value = average_hash(target_path, hash_size)
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "audit": "near_duplicate_read_error",
                    "image_id": row.get("image_id", ""),
                    "target_path": row.get("target_path", ""),
                    "error": str(exc),
                }
            )
            continue
        hashes[value].append(row.to_dict())

    found = 0
    values = list(hashes)
    for i, value_a in enumerate(values):
        possible_values = [value_a]
        # Exact hash buckets catch the highest-risk near duplicates quickly.
        for value_b in possible_values:
            if value_b not in hashes:
                continue
            candidates = hashes[value_b]
            for left_index in range(len(candidates)):
                for right_index in range(left_index + 1, len(candidates)):
                    left = candidates[left_index]
                    right = candidates[right_index]
                    if left["split"] == right["split"]:
                        continue
                    distance = hamming_distance(value_a, value_b)
                    if distance <= max_hamming:
                        rows.append(
                            {
                                "audit": "near_duplicate_average_hash",
                                "hamming_distance": distance,
                                "left_split": left["split"],
                                "right_split": right["split"],
                                "left_stage": left["stage_id"],
                                "right_stage": right["stage_id"],
                                "left_target": left["target_path"],
                                "right_target": right["target_path"],
                            }
                        )
                        found += 1
                        if found >= max_pairs_per_group:
                            return rows
        if i % 1000 == 0:
            continue
    return rows


def plot_outputs(summary_rows: list[dict[str, Any]], per_class: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"Skipping plots because matplotlib is unavailable: {exc}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        order = summary.sort_values("clean_accuracy_percent", ascending=False)["model"].tolist()
        x = np.arange(len(order))
        width = 0.35
        ordered = summary.set_index("model").loc[order]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width / 2, ordered["clean_accuracy_percent"], width, label="Clean Acc")
        ax.bar(x + width / 2, ordered["noise_accuracy_percent"], width, label="Noise Acc")
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=25, ha="right")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Visual model clean/noise accuracy")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "visual_model_accuracy_clean_noise.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(order, ordered["accuracy_drop_percent"])
        ax.set_xticklabels(order, rotation=25, ha="right")
        ax.set_ylabel("Accuracy drop (%)")
        ax.set_title("Clean-to-noise accuracy drop")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "visual_model_accuracy_drop.png", dpi=200)
        plt.close(fig)

    per_class_df = pd.DataFrame(per_class)
    if not per_class_df.empty:
        for eval_set in ["clean", "noise"]:
            subset = per_class_df[per_class_df["eval_set"] == eval_set]
            pivot = subset.pivot_table(index="class", columns="model", values="f1")
            fig, ax = plt.subplots(figsize=(12, 6))
            im = ax.imshow(pivot.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="viridis")
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_title(f"Per-class F1 ({eval_set})")
            fig.colorbar(im, ax=ax, label="F1")
            fig.tight_layout()
            fig.savefig(output_dir / f"per_class_f1_{eval_set}.png", dpi=200)
            plt.close(fig)


def write_markdown_report(
    path: Path,
    summary_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
) -> None:
    lines: list[str] = []
    lines.append("# 视觉模型补充实验汇总")
    lines.append("")
    lines.append("## 总体结果")
    lines.append("")
    lines.append("| Model | Params(M) | Clean Acc | Clean Macro-F1 | Noise Acc | Noise Macro-F1 | ΔAcc | Best epoch |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        lines.append(
            "| {model} | {params_M} | {clean_accuracy_percent} | {clean_macro_f1} | "
            "{noise_accuracy_percent} | {noise_macro_f1} | {accuracy_drop_percent} | {best_epoch} |".format(**row)
        )
    if source_rows:
        lines.append("")
        lines.append("## 分来源结果")
        lines.append("")
        lines.append("| Model | Eval | Source | N | Acc | Macro-F1 |")
        lines.append("|---|---|---|---:|---:|---:|")
        for row in source_rows:
            if "true_label" in row:
                continue
            lines.append(
                "| {model} | {eval_set} | {source_type} | {samples} | {accuracy_percent} | {macro_f1} |".format(**row)
            )
    if audit_rows:
        lines.append("")
        lines.append("## 数据重复审计")
        lines.append("")
        lines.append("| Audit | Duplicate rows | Duplicate groups | Note |")
        lines.append("|---|---:|---:|---|")
        for row in audit_rows:
            if row.get("audit", "").startswith("exact_duplicate"):
                lines.append(
                    f"| {row.get('audit')} | {row.get('duplicate_rows')} | "
                    f"{row.get('duplicate_groups')} | {row.get('note', '')} |"
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("utf-8", errors="replace").decode("ascii", errors="replace"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", type=Path, default=[], help="Specific run directory. Can repeat.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--analysis-output-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--name", type=str, default="visual_model_supplementary_analysis")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--near-duplicate-audit", action="store_true")
    parser.add_argument("--hash-size", type=int, default=8)
    parser.add_argument("--max-hamming", type=int, default=0)
    parser.add_argument("--max-near-duplicate-pairs", type=int, default=5000)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    out_dir = args.analysis_output_root.resolve() / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.manifest.resolve())
    runs = discover_runs(args.run_dir, output_root)
    if not runs:
        raise SystemExit("No completed model runs with metrics_full.json were found.")

    summary_rows = summarize_metrics(runs)
    source_rows = summarize_by_source(runs, manifest)
    class_rows = per_class_rows(runs)
    mcnemar_rows, bootstrap_rows = significance_rows(
        runs=runs,
        manifest=manifest,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    audit_rows = exact_duplicate_audit(manifest)
    if args.near_duplicate_audit:
        audit_rows.extend(
            near_duplicate_audit(
                manifest=manifest,
                project_root=PROJECT_ROOT,
                max_hamming=args.max_hamming,
                hash_size=args.hash_size,
                max_pairs_per_group=args.max_near_duplicate_pairs,
            )
        )

    write_csv(out_dir / "visual_model_overall_summary.csv", summary_rows)
    write_csv(out_dir / "visual_model_source_metrics.csv", source_rows)
    write_csv(out_dir / "visual_model_per_class_metrics.csv", class_rows)
    write_csv(out_dir / "visual_model_mcnemar_tests.csv", mcnemar_rows)
    write_csv(out_dir / "visual_model_bootstrap_accuracy_diff_ci.csv", bootstrap_rows)
    write_csv(out_dir / "visual_model_duplicate_audit.csv", audit_rows)
    write_markdown_report(out_dir / "visual_model_supplementary_summary.md", summary_rows, source_rows, audit_rows)
    if not args.no_plots:
        plot_outputs(summary_rows, class_rows, out_dir / "figures")

    safe_print(f"Completed visual model supplementary analysis: {out_dir}")
    safe_print(f"Models: {', '.join(row['model'] for row in summary_rows)}")


if __name__ == "__main__":
    main()
