import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "data_split_811_full" / "wheat_split_811_manifest.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "vlm_comparison" / "manifests"
DEFAULT_SEED = 20260703

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


def normalize_path(path_value):
    return str((PROJECT_ROOT / str(path_value)).resolve())


def prepare_manifest(df):
    df = df.copy()
    df["image_path"] = df["target_path"].map(normalize_path)
    df["source_abs_path"] = df["source_path"].map(normalize_path)
    return df


def sample_stage_source(
    df,
    split,
    per_stage,
    seed,
    subset_name,
    sample_role,
    source_targets=None,
):
    rows = []
    split_df = df[df["split"] == split].copy()

    for stage_idx, stage_id in enumerate(STAGE_ORDER):
        stage_df = split_df[split_df["stage_id"] == stage_id].copy()
        if stage_df.empty:
            raise ValueError(f"No rows for split={split}, stage_id={stage_id}")

        if source_targets is None:
            take_total = min(per_stage, len(stage_df))
            parts = []
            for _, group in stage_df.groupby("source_type", sort=False):
                take_n = min(
                    len(group),
                    max(1, round(take_total * len(group) / len(stage_df))),
                )
                parts.append(group.sample(n=take_n, random_state=seed + stage_idx))
            sampled = pd.concat(parts, ignore_index=False)

            if len(sampled) > take_total:
                sampled = sampled.sample(n=take_total, random_state=seed + 1000 + stage_idx)
            elif len(sampled) < take_total:
                remaining = stage_df.drop(index=sampled.index, errors="ignore")
                need = take_total - len(sampled)
                if need > 0 and not remaining.empty:
                    extra = remaining.sample(
                        n=min(need, len(remaining)),
                        random_state=seed + 2000 + stage_idx,
                    )
                    sampled = pd.concat([sampled, extra], ignore_index=False)
        else:
            parts = []
            used_idx = set()
            for source_type, target_n in source_targets.items():
                group = stage_df[stage_df["source_type"] == source_type]
                take_n = min(target_n, len(group))
                if take_n > 0:
                    part = group.sample(n=take_n, random_state=seed + stage_idx)
                    parts.append(part)
                    used_idx.update(part.index.tolist())

            sampled = pd.concat(parts, ignore_index=False) if parts else pd.DataFrame()
            if len(sampled) < per_stage:
                need = per_stage - len(sampled)
                remaining = stage_df.drop(index=list(used_idx), errors="ignore")
                if need > 0 and not remaining.empty:
                    extra = remaining.sample(
                        n=min(need, len(remaining)),
                        random_state=seed + 2000 + stage_idx,
                    )
                    sampled = pd.concat([sampled, extra], ignore_index=False)

        sampled = sampled.sample(frac=1, random_state=seed + 3000 + stage_idx)
        rows.append(sampled)

    subset = pd.concat(rows, ignore_index=True)
    subset["subset_name"] = subset_name
    subset["sample_role"] = sample_role
    subset["sample_seed"] = seed
    subset["subset_index"] = range(1, len(subset) + 1)
    return subset


def full_split_subset(df, split, seed, subset_name, sample_role):
    subset = df[df["split"] == split].copy()
    subset = subset.sort_values(["stage_id", "source_type", "image_id"]).reset_index(drop=True)
    subset["subset_name"] = subset_name
    subset["sample_role"] = sample_role
    subset["sample_seed"] = seed
    subset["subset_index"] = range(1, len(subset) + 1)
    return subset


def write_subset(subset, out_dir, filename):
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
    subset[columns].to_csv(path, index=False, encoding="utf-8-sig")
    return path


def summarize_subset(subset):
    by_stage = (
        subset.groupby(["stage_id", "source_type"])
        .size()
        .reset_index(name="count")
        .sort_values(["stage_id", "source_type"])
    )
    by_split = subset.groupby(["split"]).size().to_dict()
    return {
        "total": int(len(subset)),
        "by_split": {str(k): int(v) for k, v in by_split.items()},
        "by_stage_source": by_stage.to_dict(orient="records"),
    }


def build_subsets(manifest_path, out_dir, seed):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(manifest_path)
    df = prepare_manifest(df)

    prompt_dev_800 = sample_stage_source(
        df,
        split="val",
        per_stage=100,
        seed=seed + 20,
        subset_name="vlm_prompt_dev_800",
        sample_role="prompt_development",
    )
    prompt_dev_400 = sample_stage_source(
        prompt_dev_800,
        split="val",
        per_stage=50,
        seed=seed + 10,
        subset_name="vlm_prompt_dev_400",
        sample_role="prompt_development",
    )

    subsets = {
        "vlm_connectivity_40.csv": sample_stage_source(
            df,
            split="val",
            per_stage=5,
            seed=seed,
            subset_name="vlm_connectivity_40",
            sample_role="connectivity_test",
            source_targets={"large_scale": 3, "experiment_tong": 2},
        ),
        "vlm_prompt_dev_400.csv": prompt_dev_400,
        "vlm_prompt_dev_800.csv": prompt_dev_800,
        "vlm_model_test_balanced_800.csv": sample_stage_source(
            df,
            split="test",
            per_stage=100,
            seed=seed + 30,
            subset_name="vlm_model_test_balanced_800",
            sample_role="model_comparison_test",
        ),
        "vlm_model_test_full_2485.csv": full_split_subset(
            df,
            split="test",
            seed=seed,
            subset_name="vlm_model_test_full_2485",
            sample_role="final_full_test",
        ),
    }

    summary = {
        "source_manifest": str(manifest_path.resolve()),
        "output_dir": str(out_dir.resolve()),
        "base_seed": seed,
        "subsets": {},
    }

    for filename, subset in subsets.items():
        path = write_subset(subset, out_dir, filename)
        summary["subsets"][filename] = {
            "path": str(path.resolve()),
            **summarize_subset(subset),
        }

    summary_path = out_dir / "vlm_fixed_subsets_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    summary_csv_rows = []
    for filename, item in summary["subsets"].items():
        for row in item["by_stage_source"]:
            summary_csv_rows.append(
                {
                    "subset_file": filename,
                    "stage_id": row["stage_id"],
                    "source_type": row["source_type"],
                    "count": row["count"],
                    "total": item["total"],
                }
            )
    summary_csv = out_dir / "vlm_fixed_subsets_stage_source_summary.csv"
    pd.DataFrame(summary_csv_rows).to_csv(summary_csv, index=False, encoding="utf-8-sig")

    return summary_path, summary_csv


def parse_args():
    parser = argparse.ArgumentParser(description="Build fixed VLM evaluation subsets.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    summary_path, summary_csv = build_subsets(
        args.manifest.resolve(),
        args.out_dir.resolve(),
        args.seed,
    )
    print(f"Generated VLM fixed subsets in: {args.out_dir.resolve()}")
    print(f"Summary JSON: {summary_path}")
    print(f"Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
