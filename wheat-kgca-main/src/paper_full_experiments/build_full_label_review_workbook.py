from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MANIFEST_DIR = PROJECT_ROOT / "outputs" / "paper_full_experiments" / "manifests"
DEFAULT_CONVNEXT_DIR = PROJECT_ROOT / "outputs" / "paper_full_experiments" / "convnext_features"
DEFAULT_VLM_OLD_DIR = PROJECT_ROOT / "outputs" / "paper_full_experiments" / "vlm_semantics"
DEFAULT_VLM_REVISED_DIR = PROJECT_ROOT / "outputs" / "paper_full_experiments" / "vlm_semantics_boundary_v3"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "paper_full_experiments" / "full_label_review"

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

STAGE_CN = {
    "seedling": "苗期",
    "tillering": "分蘖期",
    "stem_elongation": "拔节期",
    "flag_leaf_booting": "旗叶孕穗期",
    "heading": "抽穗期",
    "flowering": "开花期",
    "grain_filling": "灌浆期",
    "maturity": "成熟期",
}

ADJACENT_PAIRS = [
    ("seedling", "tillering"),
    ("tillering", "stem_elongation"),
    ("stem_elongation", "flag_leaf_booting"),
    ("flag_leaf_booting", "heading"),
    ("heading", "flowering"),
    ("flowering", "grain_filling"),
    ("grain_filling", "maturity"),
]

TARGET_BOUNDARY_PAIRS = {
    "seedling__vs__tillering",
    "heading__vs__flowering",
    "flowering__vs__grain_filling",
    "grain_filling__vs__maturity",
}


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def pair_name(a: str, b: str) -> str:
    if not a or not b or a == b:
        return ""
    if a not in STAGE_ORDER or b not in STAGE_ORDER:
        return ""
    ia = STAGE_ORDER.index(a)
    ib = STAGE_ORDER.index(b)
    if abs(ia - ib) != 1:
        return ""
    left, right = sorted([a, b], key=STAGE_ORDER.index)
    return f"{left}__vs__{right}"


def infer_original_pair(stage: str) -> str:
    if stage == "seedling":
        return "seedling__vs__tillering"
    if stage == "tillering":
        return "seedling__vs__tillering"
    if stage == "heading":
        return "heading__vs__flowering"
    if stage == "flowering":
        return "heading__vs__flowering;flowering__vs__grain_filling"
    if stage == "grain_filling":
        return "flowering__vs__grain_filling;grain_filling__vs__maturity"
    if stage == "maturity":
        return "grain_filling__vs__maturity"
    if stage == "flag_leaf_booting":
        return "flag_leaf_booting__vs__heading"
    return ""


def load_manifest(split: str, manifest_dir: Path) -> pd.DataFrame:
    path = manifest_dir / f"full_{split}_{'19877' if split == 'train' else '2485'}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    df = pd.read_csv(path)
    if "image_path" not in df.columns:
        df["image_path"] = df["target_path"].map(lambda p: str((PROJECT_ROOT / str(p)).resolve()))
    return df


def load_predictions(split: str, convnext_dir: Path) -> pd.DataFrame:
    path = convnext_dir / f"full_{split}_{'19877' if split == 'train' else '2485'}" / "convnext_predictions.csv"
    if not path.exists():
        return pd.DataFrame(columns=["image_id"])
    df = pd.read_csv(path)
    keep = [c for c in ["image_id", "pred_label", "confidence", "correct"] if c in df.columns]
    df = df[keep].copy()
    return df.rename(
        columns={
            "pred_label": "convnext_pred",
            "confidence": "convnext_confidence",
            "correct": "convnext_correct",
        }
    )


def candidate_vlm_paths(split: str, old_dir: Path, revised_dir: Path) -> Iterable[Path]:
    size = "19877" if split == "train" else "2485"
    yield revised_dir / f"revised_{split}_{size}" / "parsed_results.csv"
    yield old_dir / f"full_{split}_{size}" / "parsed_results.csv"


def load_vlm(split: str, old_dir: Path, revised_dir: Path) -> tuple[pd.DataFrame, str]:
    for path in candidate_vlm_paths(split, old_dir, revised_dir):
        if path.exists():
            df = pd.read_csv(path)
            keep = [
                "image_id",
                "predicted_stage",
                "second_candidate",
                "confidence",
                "image_quality",
                "spike_status",
                "anther_visible",
                "grain_fullness",
                "flag_leaf_booting_clue",
                "tillering_clue",
                "stem_elongation_clue",
                "plant_color",
                "field_noise",
                "spike_visibility",
                "anther_visibility",
                "grain_fullness_score",
                "flag_leaf_booting_strength",
                "tillering_strength",
                "stem_elongation_strength",
                "yellowing_degree",
                "noise_degree",
                "contradiction_count",
                "contradiction_flags",
                "short_reason",
                "uncertainty_reason",
            ]
            keep = [c for c in keep if c in df.columns]
            out = df[keep].copy()
            rename = {c: f"vlm_{c}" for c in keep if c != "image_id"}
            return out.rename(columns=rename), str(path)
    return pd.DataFrame(columns=["image_id"]), ""


def add_review_columns(df: pd.DataFrame) -> pd.DataFrame:
    original = df["stage_id"].map(normalize_text)
    convnext = df.get("convnext_pred", pd.Series([""] * len(df))).map(normalize_text)
    vlm_pred = df.get("vlm_predicted_stage", pd.Series([""] * len(df))).map(normalize_text)
    vlm_second = df.get("vlm_second_candidate", pd.Series([""] * len(df))).map(normalize_text)

    pred_pair = [pair_name(a, b) for a, b in zip(original, convnext)]
    vlm_pair = [pair_name(a, b) for a, b in zip(vlm_pred, vlm_second)]
    original_pair = original.map(infer_original_pair)

    df["original_boundary_pair_hint"] = original_pair
    df["convnext_boundary_pair_hint"] = pred_pair
    df["vlm_boundary_pair_hint"] = vlm_pair

    confidence = pd.to_numeric(df.get("convnext_confidence", 1.0), errors="coerce").fillna(1.0)
    convnext_correct = df.get("convnext_correct", True)
    if not isinstance(convnext_correct, pd.Series):
        convnext_correct = pd.Series([True] * len(df))
    convnext_correct = convnext_correct.astype(str).str.lower().isin(["true", "1", "yes"])

    priority = []
    reason = []
    for i, row in df.iterrows():
        hints = {
            normalize_text(row.get("original_boundary_pair_hint")),
            normalize_text(row.get("convnext_boundary_pair_hint")),
            normalize_text(row.get("vlm_boundary_pair_hint")),
        }
        hint_text = ";".join(sorted(h for h in hints if h))
        row_reasons = []
        score = 5

        if "grain_filling__vs__maturity" in hint_text:
            score = min(score, 1)
            row_reasons.append("grain_filling/maturity rule check")
        if any(pair in hint_text for pair in TARGET_BOUNDARY_PAIRS):
            score = min(score, 2)
            row_reasons.append("target adjacent boundary")
        if not bool(convnext_correct.iloc[i]):
            score = min(score, 2)
            row_reasons.append("convnext error")
        if float(confidence.iloc[i]) < 0.80:
            score = min(score, 3)
            row_reasons.append("low visual confidence")
        if normalize_text(row.get("vlm_contradiction_count")) not in {"", "0", "0.0"}:
            score = min(score, 3)
            row_reasons.append("vlm contradiction")
        if score == 5:
            row_reasons.append("routine full review")

        priority.append(score)
        reason.append("; ".join(row_reasons))

    df["review_priority"] = priority
    df["review_reason"] = reason
    df["review_status"] = ""
    df["review_label"] = ""
    df["secondary_label"] = ""
    df["primary_weight"] = ""
    df["secondary_weight"] = ""
    df["rule_id"] = ""
    df["wheat_subject_color"] = ""
    df["spike_review"] = ""
    df["anther_review"] = ""
    df["grain_review"] = ""
    df["key_region_visible"] = ""
    df["reviewer_note"] = ""
    return df


def build_split(split: str, args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    manifest = load_manifest(split, args.manifest_dir)
    predictions = load_predictions(split, args.convnext_dir)
    vlm, vlm_path = load_vlm(split, args.vlm_old_dir, args.vlm_revised_dir)

    df = manifest.merge(predictions, on="image_id", how="left")
    df = df.merge(vlm, on="image_id", how="left")
    df = add_review_columns(df)

    front_cols = [
        "review_priority",
        "review_reason",
        "review_status",
        "review_label",
        "secondary_label",
        "primary_weight",
        "secondary_weight",
        "rule_id",
        "wheat_subject_color",
        "spike_review",
        "anther_review",
        "grain_review",
        "key_region_visible",
        "reviewer_note",
        "image_id",
        "split",
        "stage_id",
        "stage_cn",
        "source_type",
        "source_dir",
        "image_path",
        "source_path",
        "original_filename",
        "convnext_pred",
        "convnext_confidence",
        "convnext_correct",
        "vlm_predicted_stage",
        "vlm_second_candidate",
        "vlm_confidence",
        "vlm_image_quality",
        "vlm_plant_color",
        "vlm_spike_status",
        "vlm_anther_visible",
        "vlm_grain_fullness",
        "vlm_field_noise",
        "vlm_yellowing_degree",
        "vlm_short_reason",
        "vlm_uncertainty_reason",
        "original_boundary_pair_hint",
        "convnext_boundary_pair_hint",
        "vlm_boundary_pair_hint",
    ]
    cols = [c for c in front_cols if c in df.columns]
    rest = [c for c in df.columns if c not in cols]
    df = df[cols + rest].sort_values(["review_priority", "split", "stage_id", "image_id"])
    return df, vlm_path


def write_outputs(split_frames: dict[str, pd.DataFrame], out_dir: Path, write_xlsx: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_df = pd.concat(split_frames.values(), ignore_index=True)

    for split, df in split_frames.items():
        df.to_csv(out_dir / f"full_label_review_{split}.csv", index=False, encoding="utf-8-sig")
    all_df.to_csv(out_dir / "full_label_review_all.csv", index=False, encoding="utf-8-sig")

    summary = (
        all_df.groupby(["split", "stage_id", "review_priority"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["split", "review_priority", "stage_id"])
    )
    summary.to_csv(out_dir / "full_label_review_summary.csv", index=False, encoding="utf-8-sig")

    if not write_xlsx:
        return

    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("openpyxl is not installed; CSV files were written, XLSX was skipped.")
        return

    xlsx_path = out_dir / "full_label_review_all.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        for split, df in split_frames.items():
            df.to_excel(writer, sheet_name=split, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build full manual label-review tables.")
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--convnext-dir", type=Path, default=DEFAULT_CONVNEXT_DIR)
    parser.add_argument("--vlm-old-dir", type=Path, default=DEFAULT_VLM_OLD_DIR)
    parser.add_argument("--vlm-revised-dir", type=Path, default=DEFAULT_VLM_REVISED_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--no-xlsx", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_frames = {}
    vlm_paths = {}
    for split in ["train", "val", "test"]:
        frame, vlm_path = build_split(split, args)
        split_frames[split] = frame
        vlm_paths[split] = vlm_path

    write_outputs(split_frames, args.out_dir, write_xlsx=not args.no_xlsx)

    total = sum(len(df) for df in split_frames.values())
    print(f"Wrote full review tables to: {args.out_dir}")
    print(f"Total rows: {total}")
    for split, frame in split_frames.items():
        print(f"{split}: {len(frame)} rows; VLM: {vlm_paths[split] or 'not found'}")


if __name__ == "__main__":
    main()
