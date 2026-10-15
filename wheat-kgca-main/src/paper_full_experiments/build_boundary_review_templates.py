from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAIR_ROOT = PROJECT_ROOT / "outputs" / "paper_full_experiments" / "pair_specialists" / "manifests"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "paper_full_experiments" / "boundary_review_templates"


REVIEW_COLUMNS = [
    "review_status",
    "review_label",
    "boundary_weight",
    "review_note",
]


def read_review_files(pair_root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(pair_root.glob("*/*_review_candidates.csv")):
        frame = pd.read_csv(path)
        split = path.name.replace("_review_candidates.csv", "")
        frame["review_file"] = str(path)
        frame["review_split"] = split
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No *_review_candidates.csv files found under {pair_root}")
    return pd.concat(frames, ignore_index=True)


def compact_template(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "image_id",
        "review_split",
        "pair_id",
        "pair_left",
        "pair_right",
        "stage_id",
        "stage_cn",
        "binary_label",
        "baseline_pred",
        "baseline_confidence",
        "baseline_confusion_pair",
        "source_type",
        "source_dir",
        "original_filename",
        "image_path",
        "source_path",
        "relabel_review_candidate",
        "baseline_low_confidence",
        "model_disagreement",
        "any_target_pair_confusion",
    ]
    columns = [col for col in columns if col in frame.columns]
    out = frame[columns].copy()
    out.insert(0, "review_priority", 3)
    if "baseline_low_confidence" in out.columns:
        out.loc[out["baseline_low_confidence"].astype(bool), "review_priority"] = 2
    if "any_target_pair_confusion" in out.columns:
        out.loc[out["any_target_pair_confusion"].astype(bool), "review_priority"] = 1
    for col in REVIEW_COLUMNS:
        out[col] = ""
    out["review_status_options"] = "clear_original|clear_corrected|boundary|uncertain"
    out["boundary_weight_rule"] = "clear=1.0; boundary=0.5; uncertain=0.0"
    return out.sort_values(["review_priority", "pair_id", "review_split", "baseline_confidence"], ascending=[True, True, True, True])


def write_guide(out_dir: Path) -> None:
    guide = """# Boundary Review Guide

Fill these columns:

```text
review_status:
  clear_original  - original 8-class label is clearly correct
  clear_corrected - original label is clearly wrong
  boundary        - image is a transition/boundary case between adjacent stages
  uncertain       - low quality or cannot judge

review_label:
  clear_original  -> use original stage_id
  clear_corrected -> use corrected 8-class stage_id
  boundary        -> use pair format such as seedling|tillering
  uncertain       -> leave empty or write uncertain

boundary_weight:
  clear_original/clear_corrected -> 1.0
  boundary -> 0.5
  uncertain -> 0.0

review_note:
  short agronomic reason, for example visible tillers, exposed anthers, grain filling, poor image quality.
```

Review priority:

```text
1 = target adjacent confusion
2 = low confidence
3 = other review candidate
```

Do not change `stage_id` in this template. Use `review_label` to record corrections.
"""
    (out_dir / "boundary_review_guide.md").write_text(guide, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact manual boundary review templates.")
    parser.add_argument("--pair-root", type=Path, default=DEFAULT_PAIR_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_reviews = read_review_files(args.pair_root)
    template = compact_template(all_reviews)
    template.to_csv(args.out_dir / "boundary_review_all.csv", index=False, encoding="utf-8-sig")
    for pair_id, group in template.groupby("pair_id"):
        pair_dir = args.out_dir / str(pair_id)
        pair_dir.mkdir(parents=True, exist_ok=True)
        group.to_csv(pair_dir / "boundary_review.csv", index=False, encoding="utf-8-sig")
        for split, split_group in group.groupby("review_split"):
            split_group.to_csv(pair_dir / f"boundary_review_{split}.csv", index=False, encoding="utf-8-sig")
    summary = (
        template.groupby(["pair_id", "review_split"], dropna=False)
        .size()
        .reset_index(name="review_candidates")
        .sort_values(["pair_id", "review_split"])
    )
    summary.to_csv(args.out_dir / "boundary_review_summary.csv", index=False, encoding="utf-8-sig")
    write_guide(args.out_dir)
    print(f"out_dir={args.out_dir}")
    print(f"rows={len(template)}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
