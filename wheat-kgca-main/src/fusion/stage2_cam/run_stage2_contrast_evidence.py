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
DEFAULT_PROMPT = PROJECT_ROOT / "outputs" / "fusion" / "stage2_cam" / "P12_contrastive_candidate_evidence.txt"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "outputs" / "fusion" / "stage2_cam" / "contrast_evidence"

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
LIST_FIELDS = ("support_a", "support_b", "contradiction_a")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def clamp01(value: Any) -> float:
    try:
        return min(max(float(value), 0.0), 1.0)
    except (TypeError, ValueError):
        return 0.0


def validate_items(value: Any, field_name: str) -> tuple[bool, list[dict[str, Any]], str]:
    if not isinstance(value, list):
        return False, [], f"{field_name} is not a list"
    if len(value) > 3:
        return False, [], f"{field_name} has more than three items"
    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            return False, [], f"{field_name} contains a non-object item"
        grid = str(item.get("grid", "")).strip()
        organ = str(item.get("organ", "")).strip()
        visibility = str(item.get("visibility", "")).strip()
        if grid not in GRID_LABELS:
            return False, [], f"{field_name} invalid grid: {grid}"
        if organ not in ORGANS:
            return False, [], f"{field_name} invalid organ: {organ}"
        if visibility not in VISIBILITY:
            return False, [], f"{field_name} invalid visibility: {visibility}"
        normalized.append(
            {
                "grid": grid,
                "organ": organ,
                "visibility": visibility,
                "confidence": clamp01(item.get("confidence")),
                "reason": str(item.get("reason", "")).strip(),
            }
        )
    return True, normalized, ""


def validate_schema(parsed: dict[str, Any]) -> tuple[bool, dict[str, list[dict[str, Any]]], str]:
    normalized: dict[str, list[dict[str, Any]]] = {}
    for field_name in LIST_FIELDS:
        valid, items, error = validate_items(parsed.get(field_name), field_name)
        if not valid:
            return False, normalized, error
        normalized[field_name] = items
    quality = str(parsed.get("overall_localization_quality", "")).strip()
    if quality not in QUALITY:
        return False, normalized, f"invalid overall_localization_quality: {quality}"
    return True, normalized, ""


def load_existing(run_dir: Path) -> dict[str, dict[str, Any]]:
    path = run_dir / "contrast_evidence_results.csv"
    if not path.exists():
        return {}
    rows = pd.read_csv(path).to_dict(orient="records")
    return {str(row["image_id"]): row for row in rows if str(row.get("status")) == "ok"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2.4: VLM localization of evidence that contrasts ConvNeXt and VLM candidates."
    )
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional prepared manifest containing image_path, candidate_a, and candidate_b. "
        "When supplied, no true-label-based diagnostic filtering is performed.",
    )
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-name", type=str, default="stage2_contrast_evidence_18")
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

    if args.manifest is not None:
        candidates = pd.read_csv(args.manifest)
        required_columns = {"image_id", "image_path", "candidate_a", "candidate_b"}
        missing_columns = sorted(required_columns - set(candidates.columns))
        if missing_columns:
            raise ValueError(f"Manifest is missing required columns: {missing_columns}")
        candidates = candidates[candidates["candidate_a"] != candidates["candidate_b"]].copy()
        manifest_source = args.manifest
    else:
        samples = pd.read_csv(args.samples)
        # true_label is used only to identify an already observed diagnostic error subset.
        # It is never inserted into the VLM prompt.
        candidates = samples[
            (samples["conv_pred"] != samples["true_label"])
            & (samples["conv_pred"] != samples["vlm_pred"])
        ].copy()
        candidates = candidates.rename(columns={"conv_pred": "candidate_a", "vlm_pred": "candidate_b"})
        manifest_source = args.samples
    if args.limit > 0:
        candidates = candidates.head(args.limit).copy()
    if len(candidates) == 0:
        raise RuntimeError("No samples have both a ConvNeXt error and a disagreeing VLM candidate.")

    prompt_template = args.prompt_file.read_text(encoding="utf-8")
    run_dir = args.output_root / args.run_name
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"{run_dir} exists. Use --resume or choose another --run-name.")
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = run_dir / "raw"
    parsed_dir = run_dir / "parsed_json"
    raw_dir.mkdir(exist_ok=True)
    parsed_dir.mkdir(exist_ok=True)
    candidates.to_csv(run_dir / "contrast_manifest.csv", index=False, encoding="utf-8-sig")

    existing = load_existing(run_dir) if args.resume else {}
    rows = list(existing.values())
    pending = candidates[~candidates["image_id"].astype(str).isin(existing)].copy()
    dump_json(
        run_dir / "run_config.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "manifest": str(manifest_source.resolve()),
            "prompt_file": str(args.prompt_file.resolve()),
            "provider": args.provider,
            "api_base_url": args.api_base_url,
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "openai_api_mode": args.openai_api_mode,
            "total_candidates": int(len(candidates)),
            "note": "true_label is never included in VLM prompts.",
        },
    )

    for _, item in tqdm(pending.iterrows(), total=len(pending), desc="Stage2 contrast evidence"):
        image_id = str(item["image_id"])
        candidate_a = str(item["candidate_a"])
        candidate_b = str(item["candidate_b"])
        prompt = (
            prompt_template.replace("{{CANDIDATE_A}}", candidate_a)
            .replace("{{CANDIDATE_B}}", candidate_b)
        )
        start = time.time()
        parsed: dict[str, Any] = {}
        normalized: dict[str, list[dict[str, Any]]] = {field_name: [] for field_name in LIST_FIELDS}
        usage: dict[str, Any] = {}
        error = ""
        status = "error"
        schema_valid = False
        attempts_used = 0

        for attempt in range(1, max(1, args.retry_attempts) + 1):
            attempts_used = attempt
            try:
                raw_text, preprocess, usage = call_vlm(
                    provider=args.provider,
                    api_key=api_key,
                    model=args.model,
                    image_path=Path(str(item["image_path"])),
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
                schema_valid, normalized, validation_error = validate_schema(parsed)
                if not schema_valid:
                    raise RuntimeError(f"Contrast schema invalid: {validation_error}")
                dump_json(raw_dir / f"{image_id}.json", {"raw_text": raw_text, "usage": usage, "preprocess": preprocess})
                dump_json(parsed_dir / f"{image_id}.json", parsed)
                status = "ok"
                break
            except Exception as exc:  # noqa: BLE001
                error = repr(exc)
                if attempt < args.retry_attempts:
                    time.sleep(args.retry_sleep * attempt)

        row = {
            "image_id": image_id,
            "selection_reason": item.get("selection_reason", ""),
            "true_label": item.get("true_label", item.get("stage_id", "")),
            "stage_id": item.get("stage_id", ""),
            "candidate_a": candidate_a,
            "candidate_b": candidate_b,
            "image_path": item.get("image_path", ""),
            "status": status,
            "error": error,
            "elapsed_sec": round(time.time() - start, 3),
            "attempts_used": attempts_used,
            "schema_valid": schema_valid,
            "overall_localization_quality": parsed.get("overall_localization_quality", "") if parsed else "",
            "localization_uncertainty": parsed.get("localization_uncertainty", "") if parsed else "",
            **{
                f"{field_name}_json": json.dumps(normalized[field_name], ensure_ascii=False)
                for field_name in LIST_FIELDS
            },
            **{f"{field_name}_count": len(normalized[field_name]) for field_name in LIST_FIELDS},
            "prompt_tokens": usage.get("prompt_tokens", ""),
            "completion_tokens": usage.get("completion_tokens", ""),
            "total_tokens": usage.get("total_tokens", ""),
        }
        rows.append(row)
        write_csv(run_dir / "contrast_evidence_results.csv", rows)
        if args.sleep > 0:
            time.sleep(args.sleep)

    result = pd.DataFrame(rows)
    ok = result[result["status"] == "ok"]
    metrics = {
        "total": int(len(result)),
        "ok": int(len(ok)),
        "error": int((result["status"] != "ok").sum()),
        "success_rate": float(len(ok) / len(result)) if len(result) else 0.0,
        "schema_valid_rate": float(ok["schema_valid"].mean()) if len(ok) else 0.0,
        "mean_latency_s": float(ok["elapsed_sec"].mean()) if len(ok) else 0.0,
        **{f"mean_{field_name}_count": float(ok[f"{field_name}_count"].mean()) if len(ok) else 0.0 for field_name in LIST_FIELDS},
    }
    dump_json(run_dir / "contrast_evidence_metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Output: {run_dir}")


if __name__ == "__main__":
    main()
