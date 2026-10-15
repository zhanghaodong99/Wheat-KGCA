from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "outputs" / "vlm_comparison" / "manifests" / "vlm_model_test_balanced_800.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "vlm_comparison" / "manifests"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a stage-specific VLM manifest from an existing manifest.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--stage-id", type=str, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--subset-name", type=str, default="")
    parser.add_argument("--sample-role", type=str, default="stage_focus_test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    df = pd.read_csv(source)
    if "stage_id" not in df.columns:
        raise ValueError(f"Missing stage_id column in {source}")

    stage_id = args.stage_id.strip()
    stage_df = df[df["stage_id"].astype(str) == stage_id].copy()
    if stage_df.empty:
        available = sorted(df["stage_id"].astype(str).unique())
        raise ValueError(f"No rows for stage_id={stage_id}. Available: {available}")

    subset_name = args.subset_name.strip() or f"{source.stem}_{stage_id}_{len(stage_df)}"
    stage_df["subset_name"] = subset_name
    stage_df["sample_role"] = args.sample_role
    stage_df["subset_index"] = range(1, len(stage_df) + 1)

    output = args.output
    if output is None:
        output = DEFAULT_OUT_DIR / f"{subset_name}.csv"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    stage_df.to_csv(output, index=False, encoding="utf-8-sig")

    summary = {
        "source": str(source),
        "output": str(output),
        "stage_id": stage_id,
        "subset_name": subset_name,
        "sample_role": args.sample_role,
        "rows": int(len(stage_df)),
        "source_type_counts": stage_df["source_type"].value_counts().to_dict() if "source_type" in stage_df else {},
        "split_counts": stage_df["split"].value_counts().to_dict() if "split" in stage_df else {},
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
