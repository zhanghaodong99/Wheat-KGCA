from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "data_split_811_full" / "wheat_split_811_manifest.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "vlm_comparison" / "manifests"
DEFAULT_OUTPUT_NAME = "vlm_prompt_dev_80_stage10.csv"
DEFAULT_SEED = 20260707

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
    if "source_path" in df.columns:
        df["source_abs_path"] = df["source_path"].map(normalize_path)
    else:
        df["source_abs_path"] = ""
    return df


def allocate_by_source(stage_df: pd.DataFrame, per_stage: int) -> dict[str, int]:
    counts = stage_df["source_type"].value_counts().to_dict()
    sources = sorted(counts)
    if not sources:
        return {}

    quotas = {}
    remainders = []
    for source in sources:
        exact = per_stage * counts[source] / len(stage_df)
        base = int(exact)
        quotas[source] = base
        remainders.append((exact - base, source))

    if per_stage >= len(sources):
        for source in sources:
            if counts[source] > 0 and quotas[source] == 0:
                quotas[source] = 1

    while sum(quotas.values()) > per_stage:
        removable = [
            (quotas[source], counts[source], source)
            for source in sources
            if quotas[source] > 0
        ]
        _, _, source = sorted(removable, key=lambda item: (item[0], -item[1], item[2]))[0]
        quotas[source] -= 1

    for _, source in sorted(remainders, reverse=True):
        if sum(quotas.values()) >= per_stage:
            break
        if quotas[source] < counts[source]:
            quotas[source] += 1

    return quotas


def sample_stage10(
    df: pd.DataFrame,
    split: str,
    per_stage: int,
    seed: int,
    subset_name: str,
) -> pd.DataFrame:
    split_df = df[df["split"] == split].copy()
    rows = []

    for stage_idx, stage_id in enumerate(STAGE_ORDER):
        stage_df = split_df[split_df["stage_id"] == stage_id].copy()
        if len(stage_df) < per_stage:
            raise ValueError(
                f"Not enough rows for split={split}, stage_id={stage_id}: "
                f"need {per_stage}, found {len(stage_df)}"
            )

        quotas = allocate_by_source(stage_df, per_stage)
        parts = []
        used_idx = set()
        for source_idx, (source_type, take_n) in enumerate(sorted(quotas.items())):
            if take_n <= 0:
                continue
            group = stage_df[stage_df["source_type"] == source_type]
            take_n = min(take_n, len(group))
            part = group.sample(
                n=take_n,
                random_state=seed + stage_idx * 100 + source_idx,
            )
            parts.append(part)
            used_idx.update(part.index.tolist())

        sampled = pd.concat(parts, ignore_index=False) if parts else pd.DataFrame()
        if len(sampled) < per_stage:
            need = per_stage - len(sampled)
            remaining = stage_df.drop(index=list(used_idx), errors="ignore")
            extra = remaining.sample(
                n=need,
                random_state=seed + stage_idx * 100 + 90,
            )
            sampled = pd.concat([sampled, extra], ignore_index=False)

        sampled = sampled.sample(frac=1, random_state=seed + stage_idx * 100 + 99)
        rows.append(sampled)

    subset = pd.concat(rows, ignore_index=True)
    subset["subset_name"] = subset_name
    subset["sample_role"] = "prompt_development_small"
    subset["sample_seed"] = seed
    subset["subset_index"] = range(1, len(subset) + 1)
    return subset


def write_subset(subset: pd.DataFrame, out_path: Path) -> None:
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
    available_columns = [column for column in columns if column in subset.columns]
    subset[available_columns].to_csv(out_path, index=False, encoding="utf-8-sig")


def summarize(subset: pd.DataFrame, source_manifest: Path, out_path: Path) -> dict:
    by_stage = subset.groupby("stage_id").size().reindex(STAGE_ORDER, fill_value=0)
    by_stage_source = (
        subset.groupby(["stage_id", "source_type"])
        .size()
        .reset_index(name="count")
        .sort_values(["stage_id", "source_type"])
    )
    return {
        "source_manifest": str(source_manifest.resolve()),
        "output_manifest": str(out_path.resolve()),
        "total": int(len(subset)),
        "split": sorted(subset["split"].unique().tolist()),
        "by_stage": {stage: int(count) for stage, count in by_stage.items()},
        "by_stage_source": by_stage_source.to_dict(orient="records"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a small VLM prompt-development subset with N images per stage."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--output-name", type=str, default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--per-stage", type=int, default=10)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.manifest)
    df = prepare_manifest(df)

    subset_name = Path(args.output_name).stem
    subset = sample_stage10(
        df=df,
        split=args.split,
        per_stage=args.per_stage,
        seed=args.seed,
        subset_name=subset_name,
    )

    out_path = args.out_dir / args.output_name
    write_subset(subset, out_path)
    summary = summarize(subset, args.manifest, out_path)
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Generated: {out_path.resolve()}")
    print(f"Summary: {summary_path.resolve()}")
    print(f"Total: {len(subset)}")
    print(subset.groupby(['stage_id', 'source_type']).size().to_string())


if __name__ == "__main__":
    main()
