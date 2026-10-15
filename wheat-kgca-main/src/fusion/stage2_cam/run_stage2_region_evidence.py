from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.run_vlm_connectivity_test import call_vlm, extract_json_object  # noqa: E402


DEFAULT_SAMPLES = PROJECT_ROOT / "outputs" / "fusion" / "stage2_cam" / "stage2_cam_selected_samples.csv"
DEFAULT_PROMPT = PROJECT_ROOT / "outputs" / "fusion" / "stage2_cam" / "P11_region_evidence_3x3.txt"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "outputs" / "fusion" / "stage2_cam" / "region_evidence"

GRID_LABELS = {
    "upper_left",
    "upper_center",
    "upper_right",
    "center_left",
    "center",
    "center_right",
    "lower_left",
    "lower_center",
    "lower_right",
}
ORGANS = {
    "spike",
    "anther",
    "grain_or_plump_spikelet",
    "flag_leaf_or_boot",
    "tillering_base",
    "elongated_stem",
    "yellowing_canopy",
    "background_noise",
}
VISIBILITY = {"yes", "uncertain"}
QUALITY = {"good", "medium", "poor"}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def atomic_json_dump(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def validate_region_schema(parsed: dict[str, Any]) -> tuple[bool, list[dict[str, Any]], str]:
    evidence = parsed.get("region_evidence")
    if not isinstance(evidence, list):
        return False, [], "region_evidence is not a list"
    if len(evidence) > 4:
        return False, [], "region_evidence has more than 4 items"

    normalized = []
    for item in evidence:
        if not isinstance(item, dict):
            return False, [], "region_evidence item is not an object"
        grid = str(item.get("grid", "")).strip()
        organ = str(item.get("organ", "")).strip()
        visibility = str(item.get("visibility", "")).strip()
        if grid not in GRID_LABELS:
            return False, [], f"invalid grid: {grid}"
        if organ not in ORGANS:
            return False, [], f"invalid organ: {organ}"
        if visibility not in VISIBILITY:
            return False, [], f"invalid visibility: {visibility}"
        normalized.append(
            {
                "grid": grid,
                "organ": organ,
                "visibility": visibility,
                "confidence": clamp01(item.get("confidence", 0.0)),
                "reason": str(item.get("reason", "")).strip(),
            }
        )

    quality = str(parsed.get("overall_localization_quality", "")).strip()
    if quality not in QUALITY:
        return False, normalized, f"invalid overall_localization_quality: {quality}"
    return True, normalized, ""


def load_existing(run_dir: Path) -> dict[str, dict[str, Any]]:
    results_path = run_dir / "region_evidence_results.csv"
    if not results_path.exists():
        return {}
    existing = pd.read_csv(results_path)
    rows = existing.to_dict(orient="records")
    return {str(row["image_id"]): row for row in rows if str(row.get("status")) == "ok"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2.2: request VLM 3x3 regional evidence for selected CAM samples.")
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-name", type=str, default="stage2_region_evidence_82")
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "dashscope", "kimi", "moonshot", "bigmodel", "zhipu", "zhipuai"])
    parser.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--api-base-url", type=str, default="")
    parser.add_argument("--model", type=str, default="gpt-5.5")
    parser.add_argument("--reasoning-effort", type=str, default="low", choices=["", "none", "minimal", "low", "medium", "high", "xhigh"])
    parser.add_argument("--openai-image-detail", type=str, default="high", choices=["auto", "low", "high", "original"])
    parser.add_argument("--openai-api-mode", type=str, default="chat_completions", choices=["responses", "chat_completions"])
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--sleep", type=float, default=2.0)
    parser.add_argument("--retry-attempts", type=int, default=6)
    parser.add_argument("--retry-sleep", type=float, default=20.0)
    parser.add_argument("--vlm-max-side", type=int, default=1024)
    parser.add_argument("--vlm-jpeg-quality", type=int, default=85)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Environment variable {args.api_key_env} is empty.")

    if not args.api_base_url:
        args.api_base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if args.provider == "openai" and not args.api_base_url:
        args.api_base_url = "https://api.openai.com/v1"

    samples = pd.read_csv(args.samples)
    if args.limit > 0:
        samples = samples.head(args.limit).copy()
    prompt = args.prompt_file.read_text(encoding="utf-8")

    run_dir = args.output_root / args.run_name
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"{run_dir} exists. Use --resume to continue this run or change --run-name.")
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = run_dir / "raw"
    parsed_dir = run_dir / "parsed_json"
    raw_dir.mkdir(exist_ok=True)
    parsed_dir.mkdir(exist_ok=True)

    existing = load_existing(run_dir) if args.resume else {}
    rows = list(existing.values())
    pending = samples[~samples["image_id"].astype(str).isin(existing)].copy()

    atomic_json_dump(
        run_dir / "run_config.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "samples": str(args.samples.resolve()),
            "prompt_file": str(args.prompt_file.resolve()),
            "provider": args.provider,
            "api_base_url": args.api_base_url,
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "openai_api_mode": args.openai_api_mode,
            "timeout": args.timeout,
            "retry_attempts": args.retry_attempts,
            "retry_sleep": args.retry_sleep,
            "total_samples": int(len(samples)),
        },
    )

    for _, item in tqdm(pending.iterrows(), total=len(pending), desc="Stage2 region evidence"):
        image_id = str(item["image_id"])
        image_path = Path(str(item["image_path"]))
        start = time.time()
        raw_text = ""
        parsed: dict[str, Any] = {}
        usage: dict[str, Any] = {}
        error = ""
        status = "error"
        schema_valid = False
        normalized_evidence: list[dict[str, Any]] = []
        attempts_used = 0

        for attempt in range(1, max(1, args.retry_attempts) + 1):
            attempts_used = attempt
            try:
                raw_text, preprocess, usage = call_vlm(
                    provider=args.provider,
                    api_key=api_key,
                    model=args.model,
                    image_path=image_path,
                    prompt=prompt,
                    timeout=args.timeout,
                    temperature=0.0,
                    max_side=args.vlm_max_side,
                    jpeg_quality=args.vlm_jpeg_quality,
                    enable_thinking=False,
                    api_base_url=args.api_base_url,
                    include_temperature=False,
                    reasoning_effort=args.reasoning_effort,
                    openai_image_detail=args.openai_image_detail,
                    openai_api_mode=args.openai_api_mode,
                )
                parsed = extract_json_object(raw_text)
                schema_valid, normalized_evidence, validation_error = validate_region_schema(parsed)
                if not schema_valid:
                    raise RuntimeError(f"Region schema invalid: {validation_error}")
                atomic_json_dump(raw_dir / f"{image_id}.json", {"raw_text": raw_text, "usage": usage, "preprocess": preprocess})
                atomic_json_dump(parsed_dir / f"{image_id}.json", parsed)
                status = "ok"
                break
            except Exception as exc:  # noqa: BLE001
                error = repr(exc)
                if attempt < args.retry_attempts:
                    time.sleep(args.retry_sleep * attempt)

        row = {
            "image_id": image_id,
            "stage2_index": item.get("stage2_index", ""),
            "selection_reason": item.get("selection_reason", ""),
            "true_label": item.get("true_label", ""),
            "conv_pred": item.get("conv_pred", ""),
            "fusion_pred": item.get("fusion_pred", ""),
            "image_path": str(image_path),
            "status": status,
            "error": error,
            "elapsed_sec": round(time.time() - start, 3),
            "attempts_used": attempts_used,
            "schema_valid": schema_valid,
            "overall_localization_quality": parsed.get("overall_localization_quality", "") if parsed else "",
            "global_observation": parsed.get("global_observation", "") if parsed else "",
            "localization_uncertainty": parsed.get("localization_uncertainty", "") if parsed else "",
            "region_evidence_json": json.dumps(normalized_evidence, ensure_ascii=False),
            "region_evidence_count": len(normalized_evidence),
            "prompt_tokens": usage.get("prompt_tokens", ""),
            "completion_tokens": usage.get("completion_tokens", ""),
            "total_tokens": usage.get("total_tokens", ""),
        }
        rows.append(row)
        write_csv(run_dir / "region_evidence_results.csv", rows)
        if args.sleep > 0:
            time.sleep(args.sleep)

    result = pd.DataFrame(rows)
    ok = result[result["status"] == "ok"]
    summary = {
        "total": int(len(result)),
        "ok": int(len(ok)),
        "error": int((result["status"] != "ok").sum()),
        "success_rate": float(len(ok) / len(result)) if len(result) else 0.0,
        "schema_valid_rate": float(ok["schema_valid"].mean()) if len(ok) else 0.0,
        "mean_latency_s": float(ok["elapsed_sec"].mean()) if len(ok) else 0.0,
        "mean_region_evidence_count": float(ok["region_evidence_count"].mean()) if len(ok) else 0.0,
    }
    atomic_json_dump(run_dir / "region_evidence_metrics.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Output: {run_dir}")


if __name__ == "__main__":
    main()
