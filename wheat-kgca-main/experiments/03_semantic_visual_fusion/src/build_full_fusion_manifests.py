from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "data_split_811_full" / "wheat_split_811_manifest.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "paper_full_experiments" / "manifests"
REQUIRED_COLUMNS = ["image_id", "split", "stage_id", "stage_cn", "source_type", "target_path"]
SPLIT_ORDER = ["train", "val", "test"]


def resolve_image_path(value: str) -> str:
    path = Path(str(value))
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def build_manifests(source: Path, out_dir: Path, strict: bool) -> None:
    df = pd.read_csv(source)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {source}: {missing}")

    df = df.copy()
    df["image_path"] = df["target_path"].map(resolve_image_path)
    if strict:
        missing_files = [path for path in df["image_path"] if not Path(path).exists()]
        if missing_files:
            examples = missing_files[:10]
            raise FileNotFoundError(f"Missing {len(missing_files)} image files. Examples: {examples}")

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str | int] = {
        "source_manifest": str(source.resolve()),
        "out_dir": str(out_dir.resolve()),
        "total_rows": int(len(df)),
    }

    keep_cols = [
        "image_id",
        "split",
        "stage_id",
        "stage_cn",
        "source_type",
        "source_dir",
        "source_path",
        "target_path",
        "image_path",
        "original_filename",
        "link_mode",
    ]
    keep_cols = [col for col in keep_cols if col in df.columns]

    for split in SPLIT_ORDER:
        split_df = df[df["split"].astype(str) == split].copy()
        split_df = split_df.sort_values(["stage_id", "image_id"]).reset_index(drop=True)
        out_path = out_dir / f"full_{split}_{len(split_df)}.csv"
        split_df[keep_cols].to_csv(out_path, index=False, encoding="utf-8-sig")
        outputs[f"{split}_rows"] = int(len(split_df))
        outputs[f"{split}_manifest"] = str(out_path.resolve())

    stage_summary = (
        df.groupby(["split", "stage_id", "stage_cn"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["split", "stage_id"])
    )
    source_summary = (
        df.groupby(["split", "stage_id", "source_type"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["split", "stage_id", "source_type"])
    )
    stage_summary.to_csv(out_dir / "full_stage_summary.csv", index=False, encoding="utf-8-sig")
    source_summary.to_csv(out_dir / "full_stage_source_summary.csv", index=False, encoding="utf-8-sig")
    (out_dir / "full_manifest_summary.json").write_text(
        json.dumps(outputs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build full train/val/test manifests for paper-scale fusion experiments.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--strict", action="store_true", help="Check that every image_path exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_manifests(args.source, args.out_dir, args.strict)


if __name__ == "__main__":
    main()
