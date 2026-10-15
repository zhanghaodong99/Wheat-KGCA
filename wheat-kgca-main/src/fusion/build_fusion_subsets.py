from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_MANIFEST = PROJECT_ROOT / "data" / "data_split_811_full" / "wheat_split_811_manifest.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "fusion" / "manifests"
DEFAULT_SEED = 20260715

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


def normalize_path(path_value: str) -> str:
    return str((PROJECT_ROOT / str(path_value)).resolve())


def prepare_manifest(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["image_path"] = df["target_path"].map(normalize_path)
    df["source_abs_path"] = df["source_path"].map(normalize_path)
    return df


def balanced_sample(
    df: pd.DataFrame,
    split: str,
    per_stage: int,
    seed: int,
    subset_name: str,
    sample_role: str,
) -> pd.DataFrame:
    rows = []
    split_df = df[df["split"] == split].copy()
    for stage_index, stage_id in enumerate(STAGE_ORDER):
        stage_df = split_df[split_df["stage_id"] == stage_id].copy()
        if stage_df.empty:
            raise ValueError(f"No rows for split={split}, stage_id={stage_id}")
        take_n = min(per_stage, len(stage_df))

        parts = []
        for source_type, group in stage_df.groupby("source_type", sort=False):
            source_n = max(1, round(take_n * len(group) / len(stage_df)))
            source_n = min(source_n, len(group))
            part = group.sample(n=source_n, random_state=seed + stage_index)
            parts.append(part)

        sampled = pd.concat(parts, ignore_index=False)
        if len(sampled) > take_n:
            sampled = sampled.sample(n=take_n, random_state=seed + 1000 + stage_index)
        elif len(sampled) < take_n:
            remaining = stage_df.drop(index=sampled.index, errors="ignore")
            need = take_n - len(sampled)
            if need > 0 and not remaining.empty:
                extra = remaining.sample(n=min(need, len(remaining)), random_state=seed + 2000 + stage_index)
                sampled = pd.concat([sampled, extra], ignore_index=False)

        sampled = sampled.sample(frac=1.0, random_state=seed + 3000 + stage_index)
        rows.append(sampled)

    subset = pd.concat(rows, ignore_index=True)
    subset["subset_name"] = subset_name
    subset["sample_role"] = sample_role
    subset["sample_seed"] = seed
    subset["subset_index"] = range(1, len(subset) + 1)
    return subset


def write_subset(subset: pd.DataFrame, out_dir: Path, filename: str) -> Path:
    columns = [
        "subset_index",
        "subset_name",
        "sample_role",
        "sample_seed",
        "image_id",
        "split",
        "stage_cn",
        "stage_id",
        "source_type",
        "source_dir",
        "image_path",
        "target_path",
        "source_abs_path",
        "source_path",
        "original_filename",
        "link_mode",
    ]
    path = out_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    subset[columns].to_csv(path, index=False, encoding="utf-8-sig")
    return path


def summarize(subsets: dict[str, pd.DataFrame], out_dir: Path, source_manifest: Path, seed: int) -> None:
    summary = {
        "source_manifest": str(source_manifest.resolve()),
        "output_dir": str(out_dir.resolve()),
        "seed": seed,
        "subsets": {},
    }
    rows = []
    for filename, subset in subsets.items():
        by_stage = subset.groupby(["split", "stage_id", "source_type"]).size().reset_index(name="count")
        summary["subsets"][filename] = {
            "total": int(len(subset)),
            "by_split": {str(k): int(v) for k, v in subset.groupby("split").size().to_dict().items()},
            "by_stage_source": by_stage.to_dict(orient="records"),
        }
        for row in by_stage.to_dict(orient="records"):
            rows.append({"subset_file": filename, **row})

    (out_dir / "fusion_subsets_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_csv(out_dir / "fusion_subsets_stage_source_summary.csv", index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed train/val/test manifests for ConvNeXt-GPT fusion.")
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-per-stage", type=int, default=100)
    parser.add_argument("--val-per-stage", type=int, default=50)
    parser.add_argument("--test-per-stage", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = prepare_manifest(pd.read_csv(args.source_manifest))

    subsets = {
        "fusion_train_800.csv": balanced_sample(
            df,
            split="train",
            per_stage=args.train_per_stage,
            seed=args.seed + 10,
            subset_name="fusion_train_800",
            sample_role="fusion_train",
        ),
        "fusion_val_400.csv": balanced_sample(
            df,
            split="val",
            per_stage=args.val_per_stage,
            seed=args.seed + 20,
            subset_name="fusion_val_400",
            sample_role="fusion_val",
        ),
        "fusion_test_800.csv": balanced_sample(
            df,
            split="test",
            per_stage=args.test_per_stage,
            seed=args.seed + 30,
            subset_name="fusion_test_800",
            sample_role="fusion_test",
        ),
    }

    for filename, subset in subsets.items():
        path = write_subset(subset, args.out_dir, filename)
        print(f"Wrote {len(subset):4d} rows: {path}")
    summarize(subsets, args.out_dir, args.source_manifest, args.seed)
    print(f"Summary: {args.out_dir / 'fusion_subsets_summary.json'}")


if __name__ == "__main__":
    main()
