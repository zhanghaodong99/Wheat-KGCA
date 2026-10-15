from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "src" / "run_vlm_connectivity_test.py"
DEFAULT_MANIFEST = PROJECT_ROOT / "outputs" / "vlm_comparison" / "manifests" / "vlm_prompt_dev_400.csv"
DEFAULT_PROMPT_DIR = PROJECT_ROOT / "outputs" / "vlm_comparison" / "prompts"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "vlm_comparison" / "prompt_screening"

PROMPT_FILES = {
    "P1": "P1_label_only.txt",
    "P2": "P2_bbch_knowledge.txt",
    "P4": "P4_organ_scan_rule_arbitration.txt",
    "P6": "P6_calibrated_organ_scan.txt",
    "P7": "P7_dataset_aligned_flowering.txt",
    "P8": "P8_fine_grained_stage_boundaries.txt",
    "P9": "P9_grain_filling_boundary_fix.txt",
    "P10": "P10_unified_post_heading_evidence.txt",
    "P11": "P11_flag_leaf_boundary_strict.txt",
}


def atomic_json_dump(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def parse_prompt_ids(text: str) -> list[str]:
    prompt_ids = [item.strip().upper() for item in text.split(",") if item.strip()]
    unknown = [item for item in prompt_ids if item not in PROMPT_FILES]
    if unknown:
        raise ValueError(f"Unknown prompt ids: {unknown}. Available: {sorted(PROMPT_FILES)}")
    return prompt_ids


def load_metrics(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "connectivity_metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def latest_child_dir(root: Path, before: set[Path]) -> Path | None:
    current = {p for p in root.iterdir() if p.is_dir()} if root.exists() else set()
    created = sorted(current - before, key=lambda p: p.stat().st_mtime, reverse=True)
    return created[0] if created else None


def score_metrics(metrics: dict[str, Any]) -> float:
    macro_f1 = float(metrics.get("macro_f1_valid_label_rows", 0.0) or 0.0)
    balanced = float(metrics.get("balanced_accuracy_valid_label_rows", 0.0) or 0.0)
    json_valid = float(metrics.get("json_valid_rate", 0.0) or 0.0)
    evidence = float(metrics.get("evidence_complete_rate", 0.0) or 0.0)
    contradiction = float(metrics.get("contradiction_rate", 1.0) or 0.0)
    return (
        0.35 * macro_f1
        + 0.20 * balanced
        + 0.15 * json_valid
        + 0.15 * evidence
        + 0.10 * (1.0 - contradiction)
        + 0.05 * 1.0
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2: VLM prompt screening.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--prompt-dir", type=Path, default=DEFAULT_PROMPT_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--prompt-ids", type=str, default="P1,P2,P4,P6")
    parser.add_argument("--model", type=str, default="qwen3.7-plus")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--vlm-max-side", type=int, default=1024)
    parser.add_argument("--vlm-jpeg-quality", type=int, default=85)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt_ids = parse_prompt_ids(args.prompt_ids)
    args.output_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stage2_root = args.output_root.resolve() / f"{args.model}_stage2_{timestamp}"
    stage2_root.mkdir(parents=True, exist_ok=False)

    run_config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "manifest": str(args.manifest.resolve()),
        "prompt_ids": prompt_ids,
        "model": args.model,
        "temperature": args.temperature,
        "limit": args.limit,
        "runner": str(RUNNER.resolve()),
    }
    atomic_json_dump(stage2_root / "stage2_run_config.json", run_config)
    shutil.copy2(args.manifest, stage2_root / "manifest_used.csv")

    before_dirs = {p for p in stage2_root.iterdir() if p.is_dir()}
    rows: list[dict[str, Any]] = []

    for prompt_id in prompt_ids:
        prompt_file = args.prompt_dir / PROMPT_FILES[prompt_id]
        if not prompt_file.exists():
            raise FileNotFoundError(prompt_file)
        prompt_out_root = stage2_root / prompt_id
        prompt_out_root.mkdir(parents=True, exist_ok=True)

        run_name = f"{args.model}_{prompt_id}_prompt_screening"
        cmd = [
            sys.executable,
            str(RUNNER),
            "--manifest",
            str(args.manifest),
            "--prompt-file",
            str(prompt_file),
            "--output-root",
            str(prompt_out_root),
            "--run-name",
            run_name,
            "--model",
            args.model,
            "--temperature",
            str(args.temperature),
            "--sleep",
            str(args.sleep),
            "--timeout",
            str(args.timeout),
            "--vlm-max-side",
            str(args.vlm_max_side),
            "--vlm-jpeg-quality",
            str(args.vlm_jpeg_quality),
        ]
        if args.limit > 0:
            cmd.extend(["--limit", str(args.limit)])

        command_text = " ".join(cmd)
        print(f"\n[Stage2] {prompt_id}: {command_text}")
        if args.dry_run:
            rows.append({"prompt_id": prompt_id, "status": "dry_run", "command": command_text})
            continue

        start_dirs = {p for p in prompt_out_root.iterdir() if p.is_dir()}
        completed = subprocess.run(cmd, cwd=PROJECT_ROOT.parent, check=False)
        if completed.returncode != 0:
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "status": "error",
                    "returncode": completed.returncode,
                    "command": command_text,
                }
            )
            atomic_json_dump(stage2_root / "stage2_partial_summary.json", {"rows": rows})
            raise RuntimeError(f"Prompt {prompt_id} failed with return code {completed.returncode}")

        run_dir = prompt_out_root / run_name
        if not run_dir.exists():
            run_dir = latest_child_dir(prompt_out_root, start_dirs)
        metrics = load_metrics(run_dir) if run_dir else {}
        row = {
            "prompt_id": prompt_id,
            "prompt_file": str(prompt_file.resolve()),
            "status": "ok",
            "run_dir": str(run_dir.resolve()) if run_dir else "",
            "score": score_metrics(metrics),
            **metrics,
        }
        rows.append(row)
        atomic_json_dump(stage2_root / "stage2_partial_summary.json", {"rows": rows})

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(stage2_root / "stage2_prompt_summary.csv", index=False, encoding="utf-8-sig")
    atomic_json_dump(
        stage2_root / "stage2_summary.json",
        {
            "run_config": run_config,
            "rows": rows,
            "best_by_score": max(rows, key=lambda row: float(row.get("score", 0.0) or 0.0)) if rows else None,
        },
    )
    print(f"\nStage 2 summary: {stage2_root / 'stage2_prompt_summary.csv'}")


if __name__ == "__main__":
    main()
