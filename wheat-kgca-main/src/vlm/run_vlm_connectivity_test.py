from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VLM_EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "01_vlm_structured_output"
DEFAULT_MANIFEST = VLM_EXPERIMENT_ROOT / "manifests" / "vlm_connectivity_40.csv"
DEFAULT_PROMPT = VLM_EXPERIMENT_ROOT / "prompts" / "P4_organ_scan_rule_arbitration.txt"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "outputs" / "vlm_comparison" / "connectivity"

STAGE_ORDER = [
    "seedling",
    "tillering",
    "stem_elongation",
    "flag_leaf_booting",
    "heading",
    "flowering",
    "grain_filling",
    "maturity",
]
SPIKE_STATUS = {"not_visible", "booting", "partly_emerged", "fully_emerged", "full_grain", "mature_dry", "uncertain"}
TRINARY = {"yes", "no", "uncertain"}
GRAIN_FULLNESS = {"none", "slight", "moderate", "clear", "full", "uncertain"}
PLANT_COLOR = {"green", "mixed_yellow", "yellow_dry", "uncertain"}
IMAGE_QUALITY = {"good", "medium", "poor"}


def atomic_json_dump(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "item"


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def clamp01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(number):
        return 0.0
    return max(0.0, min(1.0, number))


def image_to_data_url(path: Path, max_side: int, jpeg_quality: int) -> tuple[str, dict[str, Any]]:
    with Image.open(path) as img:
        original_width, original_height = img.size
        img = img.convert("RGB")
        scale = 1.0
        if max_side > 0:
            longest = max(original_width, original_height)
            if longest > max_side:
                scale = max_side / longest
                new_size = (
                    max(1, int(round(original_width * scale))),
                    max(1, int(round(original_height * scale))),
                )
                img = img.resize(new_size, Image.Resampling.LANCZOS)

        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
        payload = buffer.getvalue()

    preprocess = {
        "original_width": original_width,
        "original_height": original_height,
        "sent_width": img.size[0],
        "sent_height": img.size[1],
        "max_side": max_side,
        "jpeg_quality": jpeg_quality,
        "keep_aspect_ratio": True,
        "crop": False,
        "scale": scale,
        "encoded_bytes": len(payload),
        "format": "JPEG",
    }
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}", preprocess


def call_dashscope_vlm(
    api_key: str,
    model: str,
    image_path: Path,
    prompt: str,
    timeout: int,
    temperature: float,
    max_side: int,
    jpeg_quality: int,
    enable_thinking: bool,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    endpoint = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    image_url, preprocess = image_to_data_url(image_path, max_side, jpeg_quality)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a strict JSON-only wheat growth-stage visual expert."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "enable_thinking": enable_thinking,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DashScope HTTP {exc.code}: {error_body}") from exc

    data = json.loads(body)
    return data["choices"][0]["message"]["content"], preprocess, data.get("usage", {})


def chat_completions_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def responses_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/responses"):
        return base
    return f"{base}/responses"


def extract_openai_response_text(data: dict[str, Any]) -> str:
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    text_parts: list[str] = []
    for item in data.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            if content.get("type") in {"output_text", "text"}:
                text = content.get("text", "")
                if isinstance(text, str):
                    text_parts.append(text)
    if text_parts:
        return "\n".join(text_parts)

    raise RuntimeError(f"OpenAI response did not contain output text. Keys: {sorted(data.keys())}")


def call_openai_responses_vlm(
    api_key: str,
    model: str,
    image_path: Path,
    prompt: str,
    timeout: int,
    max_side: int,
    jpeg_quality: int,
    api_base_url: str,
    reasoning_effort: str,
    openai_image_detail: str,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    endpoint = responses_endpoint(api_base_url)
    image_url, preprocess = image_to_data_url(image_path, max_side, jpeg_quality)
    image_content: dict[str, Any] = {"type": "input_image", "image_url": image_url}
    if openai_image_detail:
        image_content["detail"] = openai_image_detail

    payload: dict[str, Any] = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": "You are a strict JSON-only wheat growth-stage visual expert.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    image_content,
                ],
            },
        ],
        "text": {"format": {"type": "json_object"}},
    }
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI HTTP {exc.code}: {error_body}") from exc

    data = json.loads(body)
    return extract_openai_response_text(data), preprocess, data.get("usage", {})


def call_openai_chat_completions_vlm(
    api_key: str,
    model: str,
    image_path: Path,
    prompt: str,
    timeout: int,
    max_side: int,
    jpeg_quality: int,
    api_base_url: str,
    reasoning_effort: str,
    openai_image_detail: str,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    endpoint = chat_completions_endpoint(api_base_url)
    image_url, preprocess = image_to_data_url(image_path, max_side, jpeg_quality)
    image_url_content: dict[str, Any] = {"url": image_url}
    if openai_image_detail:
        image_url_content["detail"] = openai_image_detail

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a strict JSON-only wheat growth-stage visual expert."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": image_url_content},
                ],
            },
        ],
        "response_format": {"type": "json_object"},
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI Chat Completions HTTP {exc.code}: {error_body}") from exc

    data = json.loads(body)
    return data["choices"][0]["message"]["content"], preprocess, data.get("usage", {})


def call_kimi_vlm(
    api_key: str,
    model: str,
    image_path: Path,
    prompt: str,
    timeout: int,
    max_side: int,
    jpeg_quality: int,
    api_base_url: str,
    enable_thinking: bool,
    include_temperature: bool,
    temperature: float,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    endpoint = chat_completions_endpoint(api_base_url)
    image_url, preprocess = image_to_data_url(image_path, max_side, jpeg_quality)
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a strict JSON-only wheat growth-stage visual expert."},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        "response_format": {"type": "json_object"},
    }
    if enable_thinking:
        payload["thinking"] = {"type": "enabled"}
    if include_temperature:
        payload["temperature"] = temperature

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Kimi HTTP {exc.code}: {error_body}") from exc

    data = json.loads(body)
    return data["choices"][0]["message"]["content"], preprocess, data.get("usage", {})


def call_bigmodel_vlm(
    api_key: str,
    model: str,
    image_path: Path,
    prompt: str,
    timeout: int,
    max_side: int,
    jpeg_quality: int,
    api_base_url: str,
    enable_thinking: bool,
    include_temperature: bool,
    temperature: float,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    endpoint = chat_completions_endpoint(api_base_url)
    image_url, preprocess = image_to_data_url(image_path, max_side, jpeg_quality)
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    if enable_thinking:
        payload["thinking"] = {"type": "enabled"}
    if include_temperature:
        payload["temperature"] = temperature

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"BigModel HTTP {exc.code}: {error_body}") from exc

    data = json.loads(body)
    return data["choices"][0]["message"]["content"], preprocess, data.get("usage", {})


def call_vlm(
    provider: str,
    api_key: str,
    model: str,
    image_path: Path,
    prompt: str,
    timeout: int,
    temperature: float,
    max_side: int,
    jpeg_quality: int,
    enable_thinking: bool,
    api_base_url: str,
    include_temperature: bool,
    reasoning_effort: str,
    openai_image_detail: str,
    openai_api_mode: str,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    provider = provider.lower().strip()
    if provider == "dashscope":
        return call_dashscope_vlm(
            api_key=api_key,
            model=model,
            image_path=image_path,
            prompt=prompt,
            timeout=timeout,
            temperature=temperature,
            max_side=max_side,
            jpeg_quality=jpeg_quality,
            enable_thinking=enable_thinking,
        )
    if provider in {"kimi", "moonshot"}:
        return call_kimi_vlm(
            api_key=api_key,
            model=model,
            image_path=image_path,
            prompt=prompt,
            timeout=timeout,
            max_side=max_side,
            jpeg_quality=jpeg_quality,
            api_base_url=api_base_url,
            enable_thinking=enable_thinking,
            include_temperature=include_temperature,
            temperature=temperature,
        )
    if provider == "openai":
        if openai_api_mode == "chat_completions":
            return call_openai_chat_completions_vlm(
                api_key=api_key,
                model=model,
                image_path=image_path,
                prompt=prompt,
                timeout=timeout,
                max_side=max_side,
                jpeg_quality=jpeg_quality,
                api_base_url=api_base_url,
                reasoning_effort=reasoning_effort,
                openai_image_detail=openai_image_detail,
            )
        return call_openai_responses_vlm(
            api_key=api_key,
            model=model,
            image_path=image_path,
            prompt=prompt,
            timeout=timeout,
            max_side=max_side,
            jpeg_quality=jpeg_quality,
            api_base_url=api_base_url,
            reasoning_effort=reasoning_effort,
            openai_image_detail=openai_image_detail,
        )
    if provider in {"bigmodel", "zhipu", "zhipuai"}:
        return call_bigmodel_vlm(
            api_key=api_key,
            model=model,
            image_path=image_path,
            prompt=prompt,
            timeout=timeout,
            max_side=max_side,
            jpeg_quality=jpeg_quality,
            api_base_url=api_base_url,
            enable_thinking=enable_thinking,
            include_temperature=include_temperature,
            temperature=temperature,
        )
    raise ValueError(f"Unsupported provider: {provider}")


def validate_schema(parsed: dict[str, Any]) -> dict[str, Any]:
    organ = parsed.get("organ_scan", {})
    evidence = parsed.get("evidence_scores", {})
    predicted_stage = str(parsed.get("predicted_stage", "")).strip()
    second_candidate = str(parsed.get("second_candidate", "")).strip()
    field_noise = organ.get("field_noise", [])
    if isinstance(field_noise, str):
        field_noise = [field_noise]

    required_top = [
        "free_observation",
        "organ_scan",
        "evidence_scores",
        "image_quality",
        "rule_check",
        "predicted_stage",
        "second_candidate",
        "confidence",
        "uncertainty_reason",
        "short_reason",
    ]
    required_organ = [
        "spike_status",
        "anther_visible",
        "flag_leaf_booting_clue",
        "tillering_clue",
        "stem_elongation_clue",
        "plant_color",
        "field_noise",
    ]
    required_scores = [
        "spike_visibility",
        "anther_visibility",
        "flag_leaf_booting_strength",
        "tillering_strength",
        "stem_elongation_strength",
        "yellowing_degree",
        "noise_degree",
    ]

    missing_top = [key for key in required_top if key not in parsed]
    missing_organ = [key for key in required_organ if key not in organ]
    missing_scores = [key for key in required_scores if key not in evidence]

    valid_values = (
        predicted_stage in STAGE_ORDER
        and (second_candidate in STAGE_ORDER or second_candidate == "")
        and organ.get("spike_status") in SPIKE_STATUS
        and organ.get("anther_visible") in TRINARY
        and organ.get("flag_leaf_booting_clue") in TRINARY
        and organ.get("tillering_clue") in TRINARY
        and organ.get("stem_elongation_clue") in TRINARY
        and ("grain_fullness" not in organ or organ.get("grain_fullness") in GRAIN_FULLNESS)
        and organ.get("plant_color") in PLANT_COLOR
        and parsed.get("image_quality") in IMAGE_QUALITY
    )
    schema_valid = not missing_top and not missing_organ and not missing_scores and valid_values
    evidence_complete = not missing_organ and not missing_scores
    return {
        "schema_valid": schema_valid,
        "label_valid": predicted_stage in STAGE_ORDER,
        "evidence_complete": evidence_complete,
        "missing_top": missing_top,
        "missing_organ": missing_organ,
        "missing_scores": missing_scores,
        "confidence_clamped": clamp01(parsed.get("confidence", 0.0)),
        "field_noise": field_noise,
    }


def contradiction_flags(parsed: dict[str, Any]) -> list[str]:
    flags = []
    organ = parsed.get("organ_scan", {})
    stage = str(parsed.get("predicted_stage", "")).strip()
    spike = organ.get("spike_status", "uncertain")
    anther = organ.get("anther_visible", "uncertain")
    color = organ.get("plant_color", "uncertain")
    if stage == "flowering" and anther == "no":
        flags.append("flowering_without_anther")
    if stage in {"heading", "flowering", "grain_filling"} and spike == "not_visible":
        flags.append("spike_stage_without_visible_spike")
    if stage == "maturity" and color == "green" and spike != "mature_dry":
        flags.append("maturity_but_green_without_mature_spike")
    if stage in {"seedling", "tillering"} and spike in {"fully_emerged", "full_grain", "mature_dry"}:
        flags.append("early_stage_with_emerged_or_mature_spike")
    return flags


def metrics_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row["status"] == "ok"]
    valid_label_rows = [row for row in ok_rows if row["label_valid"]]
    result = {
        "total": len(rows),
        "ok": len(ok_rows),
        "errors": len(rows) - len(ok_rows),
        "success_rate": len(ok_rows) / len(rows) if rows else 0.0,
        "json_valid_rate": sum(row["json_valid"] for row in rows) / len(rows) if rows else 0.0,
        "schema_valid_rate": sum(row["schema_valid"] for row in rows) / len(rows) if rows else 0.0,
        "label_valid_rate": sum(row["label_valid"] for row in rows) / len(rows) if rows else 0.0,
        "evidence_complete_rate": sum(row["evidence_complete"] for row in rows) / len(rows) if rows else 0.0,
        "contradiction_rate": sum(row["contradiction_count"] > 0 for row in rows) / len(rows) if rows else 0.0,
        "mean_latency_s": sum(float(row["elapsed_sec"]) for row in rows) / len(rows) if rows else 0.0,
    }
    if valid_label_rows:
        y_true = [row["stage_id"] for row in valid_label_rows]
        y_pred = [row["predicted_stage"] for row in valid_label_rows]
        result.update(
            {
                "accuracy_valid_label_rows": accuracy_score(y_true, y_pred),
                "macro_f1_valid_label_rows": f1_score(y_true, y_pred, average="macro", labels=STAGE_ORDER, zero_division=0),
                "balanced_accuracy_valid_label_rows": balanced_accuracy_score(y_true, y_pred),
            }
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VLM connectivity test on a fixed manifest.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-name", type=str, default="")
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
    parser.add_argument("--model", type=str, default="qwen3.7-plus")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--include-temperature",
        action="store_true",
        help="Send temperature to non-DashScope providers. Kimi K2.6/K2.7 usually requires provider defaults.",
    )
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--vlm-max-side", type=int, default=1024)
    parser.add_argument("--vlm-jpeg-quality", type=int, default=85)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retry-attempts", type=int, default=1)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    parser.add_argument("--stop-after-consecutive-errors", type=int, default=0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing raw JSON records in the run directory and skip completed images.",
    )
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Environment variable {args.api_key_env} is empty.")
    if args.provider == "openai" and args.api_base_url == "https://api.moonshot.cn/v1":
        args.api_base_url = "https://api.openai.com/v1"

    manifest = pd.read_csv(args.manifest)
    if args.limit > 0:
        manifest = manifest.head(args.limit).copy()
    prompt = args.prompt_file.read_text(encoding="utf-8")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name.strip() or f"{safe_name(args.model)}_P4_connectivity_{timestamp}"
    run_dir = args.output_root.resolve() / run_name
    raw_dir = run_dir / "raw"
    parsed_dir = run_dir / "parsed_json"
    run_dir.mkdir(parents=True, exist_ok=False)
    raw_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    atomic_json_dump(
        run_dir / "run_config.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "manifest": str(args.manifest.resolve()),
            "prompt_file": str(args.prompt_file.resolve()),
            "model": args.model,
            "provider": args.provider,
            "api_key_env": args.api_key_env,
            "api_base_url": args.api_base_url
            if args.provider
            in {"kimi", "moonshot", "bigmodel", "zhipu", "zhipuai", "openai"}
            else "",
            "temperature": args.temperature,
            "include_temperature": args.include_temperature,
            "enable_thinking": args.enable_thinking,
            "reasoning_effort": args.reasoning_effort,
            "openai_image_detail": args.openai_image_detail,
            "openai_api_mode": args.openai_api_mode,
            "vlm_max_side": args.vlm_max_side,
            "vlm_jpeg_quality": args.vlm_jpeg_quality,
            "retry_attempts": args.retry_attempts,
            "retry_sleep": args.retry_sleep,
            "stop_after_consecutive_errors": args.stop_after_consecutive_errors,
            "num_images": int(len(manifest)),
        },
    )
    (run_dir / "prompt_used.txt").write_text(prompt, encoding="utf-8")

    rows: list[dict[str, Any]] = []
    consecutive_errors = 0
    progress = tqdm(manifest.to_dict(orient="records"), desc="VLM connectivity", dynamic_ncols=True)
    for item in progress:
        image_path = Path(item["image_path"])
        image_id = str(item["image_id"])
        raw_path = raw_dir / f"{safe_name(image_id)}.json"
        parsed_path = parsed_dir / f"{safe_name(image_id)}.json"
        start = time.time()
        status = "ok"
        error = ""
        raw_text = ""
        parsed: dict[str, Any] = {}
        usage: dict[str, Any] = {}
        preprocess: dict[str, Any] = {}
        json_valid = False
        validation = {
            "schema_valid": False,
            "label_valid": False,
            "evidence_complete": False,
            "confidence_clamped": 0.0,
            "field_noise": [],
            "missing_top": [],
            "missing_organ": [],
            "missing_scores": [],
        }
        flags: list[str] = []
        attempts_used = 0
        loaded_from_resume = False

        if args.resume and raw_path.exists():
            try:
                cached = json.loads(raw_path.read_text(encoding="utf-8"))
                status = str(cached.get("status", "ok"))
                error = str(cached.get("error", ""))
                raw_text = str(cached.get("raw_text", ""))
                parsed = cached.get("parsed", {}) if isinstance(cached.get("parsed", {}), dict) else {}
                cached_validation = cached.get("validation", {})
                if isinstance(cached_validation, dict):
                    validation = {**validation, **cached_validation}
                flags = cached.get("contradiction_flags", []) if isinstance(cached.get("contradiction_flags", []), list) else []
                usage = cached.get("usage", {}) if isinstance(cached.get("usage", {}), dict) else {}
                preprocess = cached.get("vlm_image_preprocess", {}) if isinstance(cached.get("vlm_image_preprocess", {}), dict) else {}
                elapsed = float(cached.get("elapsed_sec", 0.0) or 0.0)
                attempts_used = int(cached.get("attempts_used", 0) or 0)
                json_valid = bool(parsed)
                loaded_from_resume = True
            except Exception as exc:  # noqa: BLE001
                progress.write(f"Could not reuse cached record for {image_id}: {repr(exc)[:240]}")

        if not loaded_from_resume:
            max_attempts = max(1, args.retry_attempts)
            for attempt in range(1, max_attempts + 1):
                attempts_used = attempt
                try:
                    raw_text, preprocess, usage = call_vlm(
                        provider=args.provider,
                        api_key=api_key,
                        model=args.model,
                        image_path=image_path,
                        prompt=prompt,
                        timeout=args.timeout,
                        temperature=args.temperature,
                        max_side=args.vlm_max_side,
                        jpeg_quality=args.vlm_jpeg_quality,
                        enable_thinking=args.enable_thinking,
                        api_base_url=args.api_base_url,
                        include_temperature=args.include_temperature,
                        reasoning_effort=args.reasoning_effort,
                        openai_image_detail=args.openai_image_detail,
                        openai_api_mode=args.openai_api_mode,
                    )
                    parsed = extract_json_object(raw_text)
                    json_valid = True
                    validation = validate_schema(parsed)
                    flags = contradiction_flags(parsed)
                    status = "ok"
                    error = ""
                    break
                except Exception as exc:  # noqa: BLE001
                    status = "error"
                    error = repr(exc)
                    if attempt < max_attempts:
                        wait_s = max(0.0, args.retry_sleep * attempt)
                        progress.write(
                            f"Retry {attempt}/{max_attempts - 1} for {image_id} after error: {error[:240]}"
                        )
                        if wait_s > 0:
                            time.sleep(wait_s)

            elapsed = time.time() - start
            raw_record = {
                "manifest_item": item,
                "status": status,
                "error": error,
                "elapsed_sec": elapsed,
                "raw_text": raw_text,
                "parsed": parsed,
                "validation": validation,
                "contradiction_flags": flags,
                "vlm_image_preprocess": preprocess,
                "usage": usage,
                "attempts_used": attempts_used,
            }
            atomic_json_dump(raw_path, raw_record)
            if parsed:
                atomic_json_dump(parsed_path, parsed)

        organ = parsed.get("organ_scan", {}) if isinstance(parsed, dict) else {}
        scores = parsed.get("evidence_scores", {}) if isinstance(parsed, dict) else {}
        row = {
            "image_id": image_id,
            "split": item.get("split", ""),
            "stage_cn": item.get("stage_cn", ""),
            "stage_id": item.get("stage_id", ""),
            "source_type": item.get("source_type", ""),
            "source_dir": item.get("source_dir", ""),
            "image_path": str(image_path),
            "status": status,
            "error": error,
            "elapsed_sec": round(elapsed, 3),
            "attempts_used": attempts_used,
            "json_valid": json_valid,
            "schema_valid": bool(validation["schema_valid"]),
            "label_valid": bool(validation["label_valid"]),
            "evidence_complete": bool(validation["evidence_complete"]),
            "predicted_stage": parsed.get("predicted_stage", "") if isinstance(parsed, dict) else "",
            "second_candidate": parsed.get("second_candidate", "") if isinstance(parsed, dict) else "",
            "correct": (parsed.get("predicted_stage", "") == item.get("stage_id", "")) if isinstance(parsed, dict) else False,
            "confidence": validation["confidence_clamped"],
            "image_quality": parsed.get("image_quality", "") if isinstance(parsed, dict) else "",
            "spike_status": organ.get("spike_status", ""),
            "anther_visible": organ.get("anther_visible", ""),
            "grain_fullness": organ.get("grain_fullness", ""),
            "flag_leaf_booting_clue": organ.get("flag_leaf_booting_clue", ""),
            "tillering_clue": organ.get("tillering_clue", ""),
            "stem_elongation_clue": organ.get("stem_elongation_clue", ""),
            "plant_color": organ.get("plant_color", ""),
            "field_noise": "|".join(validation["field_noise"]),
            "spike_visibility": scores.get("spike_visibility", ""),
            "anther_visibility": scores.get("anther_visibility", ""),
            "grain_fullness_score": scores.get("grain_fullness_score", ""),
            "flag_leaf_booting_strength": scores.get("flag_leaf_booting_strength", ""),
            "tillering_strength": scores.get("tillering_strength", ""),
            "stem_elongation_strength": scores.get("stem_elongation_strength", ""),
            "yellowing_degree": scores.get("yellowing_degree", ""),
            "noise_degree": scores.get("noise_degree", ""),
            "contradiction_count": len(flags),
            "contradiction_flags": "|".join(flags),
            "short_reason": parsed.get("short_reason", "") if isinstance(parsed, dict) else "",
            "uncertainty_reason": parsed.get("uncertainty_reason", "") if isinstance(parsed, dict) else "",
            "prompt_tokens": usage.get("prompt_tokens", ""),
            "completion_tokens": usage.get("completion_tokens", ""),
            "total_tokens": usage.get("total_tokens", ""),
        }
        rows.append(row)
        if status == "ok":
            consecutive_errors = 0
        else:
            consecutive_errors += 1
        progress.set_postfix(ok=sum(r["status"] == "ok" for r in rows), errors=sum(r["status"] != "ok" for r in rows))

        if status == "error" and ("AllocationQuota" in error or "FreeTierOnly" in error):
            progress.write("Quota error detected. Stop remaining calls.")
            break
        if (
            args.stop_after_consecutive_errors > 0
            and consecutive_errors >= args.stop_after_consecutive_errors
        ):
            progress.write(
                f"Stop after {consecutive_errors} consecutive errors. Last error: {error[:300]}"
            )
            break
        if args.sleep > 0:
            time.sleep(args.sleep)

    write_csv(run_dir / "parsed_results.csv", rows)
    metrics = metrics_from_rows(rows)
    atomic_json_dump(run_dir / "connectivity_metrics.json", metrics)
    print(json.dumps({"run_dir": str(run_dir), **metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
