from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "src" / "run_vlm_connectivity_test.py"
DEFAULT_MANIFEST = PROJECT_ROOT / "outputs" / "vlm_comparison" / "manifests" / "vlm_model_test_balanced_800.csv"
DEFAULT_PROMPT_DIR = PROJECT_ROOT / "outputs" / "vlm_comparison" / "prompts"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "vlm_comparison" / "model_comparison"

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


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "item"


def parse_csv_items(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_prompt_ids(text: str) -> list[str]:
    prompt_ids = [item.upper() for item in parse_csv_items(text)]
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
    success = float(metrics.get("success_rate", 0.0) or 0.0)
    return (
        0.35 * macro_f1
        + 0.20 * balanced
        + 0.15 * json_valid
        + 0.15 * evidence
        + 0.10 * (1.0 - contradiction)
        + 0.05 * success
    )


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data._"
    columns = [
        "model",
        "prompt_id",
        "total",
        "ok",
        "score",
        "accuracy_valid_label_rows",
        "macro_f1_valid_label_rows",
        "balanced_accuracy_valid_label_rows",
        "json_valid_rate",
        "schema_valid_rate",
        "evidence_complete_rate",
        "contradiction_rate",
        "mean_latency_s",
    ]
    present = [column for column in columns if column in df.columns]
    lines = []
    lines.append("| " + " | ".join(present) + " |")
    lines.append("| " + " | ".join("---" for _ in present) + " |")
    for _, row in df.iterrows():
        values = []
        for column in present:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_markdown_summary(path: Path, run_config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(rows)
    text = f"""# Stage 3 VLM Model Comparison Summary

## Run Config

```json
{json.dumps(run_config, ensure_ascii=False, indent=2)}
```

## Results

{markdown_table(df)}

## Notes

- Stage 3 uses a fixed test manifest and frozen prompt(s).
- Do not tune prompts based on these test results.
- If a run has per-image errors but the runner exits normally, inspect its `parsed_results.csv` and raw error files.
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3: fixed-prompt VLM model comparison on test subsets.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--prompt-dir", type=Path, default=DEFAULT_PROMPT_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--prompt-ids", type=str, default="P7")
    parser.add_argument("--models", type=str, default="qwen3.7-plus")
    parser.add_argument(
        "--provider",
        type=str,
        default="dashscope",
        choices=[
            "dashscope",
            "kimi",
            "moonshot",
            "bigmodel",
            "zhipu",
            "zhipuai",
            "openai",
        ],
    )
    parser.add_argument("--api-key-env", type=str, default="DASHSCOPE_API_KEY")
    parser.add_argument("--api-base-url", type=str, default="https://api.moonshot.cn/v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--include-temperature",
        action="store_true",
        help="Forward temperature to providers that accept it. For Kimi K2.6/K2.7, keep this off unless required.",
    )
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--vlm-max-side", type=int, default=1024)
    parser.add_argument("--vlm-jpeg-quality", type=int, default=85)
    parser.add_argument("--retry-attempts", type=int, default=1)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    parser.add_argument("--stop-after-consecutive-errors", type=int, default=0)
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="",
        choices=["", "none", "minimal", "low", "medium", "high", "xhigh"],
        help="OpenAI Responses API reasoning.effort. Leave empty for provider default.",
    )
    parser.add_argument(
        "--openai-image-detail",
        type=str,
        default="high",
        choices=["auto", "low", "high", "original"],
        help="OpenAI image detail level used by Responses API image input.",
    )
    parser.add_argument(
        "--openai-api-mode",
        type=str,
        default="responses",
        choices=["responses", "chat_completions"],
        help="Use Responses API or OpenAI-compatible Chat Completions for provider=openai.",
    )
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.provider in {"bigmodel", "zhipu", "zhipuai"} and args.api_base_url == "https://api.moonshot.cn/v1":
        args.api_base_url = "https://open.bigmodel.cn/api/paas/v4"
    if args.provider == "openai" and args.api_base_url == "https://api.moonshot.cn/v1":
        args.api_base_url = "https://api.openai.com/v1"
    prompt_ids = parse_prompt_ids(args.prompt_ids)
    models = parse_csv_items(args.models)
    if not models:
        raise ValueError("At least one model must be provided via --models.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stage3_root = args.output_root.resolve() / f"stage3_{timestamp}"
    stage3_root.mkdir(parents=True, exist_ok=False)

    run_config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "manifest": str(args.manifest.resolve()),
        "prompt_ids": prompt_ids,
        "models": models,
        "provider": args.provider,
        "api_key_env": args.api_key_env,
        "api_base_url": args.api_base_url
        if args.provider
        in {"kimi", "moonshot", "bigmodel", "zhipu", "zhipuai", "openai"}
        else "",
        "temperature": args.temperature,
        "include_temperature": args.include_temperature,
        "reasoning_effort": args.reasoning_effort,
        "openai_image_detail": args.openai_image_detail,
        "openai_api_mode": args.openai_api_mode,
        "limit": args.limit,
        "enable_thinking": args.enable_thinking,
        "retry_attempts": args.retry_attempts,
        "retry_sleep": args.retry_sleep,
        "stop_after_consecutive_errors": args.stop_after_consecutive_errors,
        "runner": str(RUNNER.resolve()),
    }
    atomic_json_dump(stage3_root / "stage3_run_config.json", run_config)
    shutil.copy2(args.manifest, stage3_root / "manifest_used.csv")

    rows: list[dict[str, Any]] = []
    for model in models:
        model_root = stage3_root / safe_name(model)
        model_root.mkdir(parents=True, exist_ok=True)
        for prompt_id in prompt_ids:
            prompt_file = args.prompt_dir / PROMPT_FILES[prompt_id]
            if not prompt_file.exists():
                raise FileNotFoundError(prompt_file)

            prompt_out_root = model_root / prompt_id
            prompt_out_root.mkdir(parents=True, exist_ok=True)
            manifest_tag = safe_name(args.manifest.stem)
            run_name = f"{safe_name(model)}_{prompt_id}_{manifest_tag}"
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
                "--provider",
                args.provider,
                "--api-key-env",
                args.api_key_env,
                "--api-base-url",
                args.api_base_url,
                "--model",
                model,
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
                "--retry-attempts",
                str(args.retry_attempts),
                "--retry-sleep",
                str(args.retry_sleep),
                "--stop-after-consecutive-errors",
                str(args.stop_after_consecutive_errors),
                "--reasoning-effort",
                args.reasoning_effort,
                "--openai-image-detail",
                args.openai_image_detail,
                "--openai-api-mode",
                args.openai_api_mode,
            ]
            if args.enable_thinking:
                cmd.append("--enable-thinking")
            if args.include_temperature:
                cmd.append("--include-temperature")
            if args.limit > 0:
                cmd.extend(["--limit", str(args.limit)])

            command_text = " ".join(cmd)
            print(f"\n[Stage3] model={model} prompt={prompt_id}: {command_text}")
            if args.dry_run:
                rows.append(
                    {
                        "model": model,
                        "prompt_id": prompt_id,
                        "prompt_file": str(prompt_file.resolve()),
                        "status": "dry_run",
                        "command": command_text,
                    }
                )
                continue

            start_dirs = {p for p in prompt_out_root.iterdir() if p.is_dir()}
            completed = subprocess.run(cmd, cwd=PROJECT_ROOT.parent, check=False)
            if completed.returncode != 0:
                row = {
                    "model": model,
                    "prompt_id": prompt_id,
                    "prompt_file": str(prompt_file.resolve()),
                    "status": "error",
                    "returncode": completed.returncode,
                    "command": command_text,
                }
                rows.append(row)
                atomic_json_dump(stage3_root / "stage3_partial_summary.json", {"rows": rows})
                if not args.continue_on_error:
                    raise RuntimeError(f"Stage 3 run failed for model={model}, prompt={prompt_id}")
                continue

            run_dir = prompt_out_root / run_name
            if not run_dir.exists():
                run_dir = latest_child_dir(prompt_out_root, start_dirs)
            metrics = load_metrics(run_dir) if run_dir else {}
            row = {
                "model": model,
                "prompt_id": prompt_id,
                "prompt_file": str(prompt_file.resolve()),
                "status": "ok",
                "run_dir": str(run_dir.resolve()) if run_dir else "",
                "score": score_metrics(metrics),
                **metrics,
            }
            rows.append(row)
            atomic_json_dump(stage3_root / "stage3_partial_summary.json", {"rows": rows})

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(stage3_root / "stage3_model_summary.csv", index=False, encoding="utf-8-sig")
    best = None
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if ok_rows:
        best = max(ok_rows, key=lambda row: float(row.get("score", 0.0) or 0.0))
    atomic_json_dump(stage3_root / "stage3_summary.json", {"run_config": run_config, "rows": rows, "best_by_score": best})
    write_markdown_summary(stage3_root / "stage3_model_comparison_summary.md", run_config, rows)
    print(f"\nStage 3 summary: {stage3_root / 'stage3_model_comparison_summary.md'}")


if __name__ == "__main__":
    main()
