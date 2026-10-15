from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


VALID_ACTIONS = {"", "keep", "relabel", "drop", "uncertain"}
STAGE_CN = {
    "seedling": "苗期",
    "tillering": "分蘖期",
    "stem_elongation": "拔节期",
    "flag_leaf_booting": "孕穗期",
    "heading": "抽穗期",
    "flowering": "开花期",
    "grain_filling": "灌浆期",
    "maturity": "成熟期",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply reviewed label-revision actions to one split manifest.")
    parser.add_argument("--split-name", choices=["train", "val", "test"], required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-drop",
        action="store_true",
        help="Allow drop/uncertain actions to remove samples. For val/test, prefer relabel/keep and avoid dropping.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    review = pd.read_csv(args.review_csv)
    if "review_action" not in review.columns:
        raise RuntimeError("review_action column is missing.")
    review["review_action"] = review["review_action"].fillna("").astype(str).str.strip().str.lower()
    invalid = sorted(set(review["review_action"]) - VALID_ACTIONS)
    if invalid:
        raise RuntimeError(f"Invalid review_action values: {invalid}")

    action_by_id = {}
    for _, row in review.iterrows():
        action = str(row["review_action"]).strip().lower()
        if not action:
            continue
        image_id = str(row["image_id"])
        action_by_id[image_id] = {
            "action": action,
            "corrected_label": str(row.get("corrected_label", "")).strip(),
            "review_note": str(row.get("review_note", "")).strip(),
        }

    revised = manifest.copy()
    dropped_ids: set[str] = set()
    relabeled = []
    ignored_drop = []
    for index, row in revised.iterrows():
        image_id = str(row["image_id"])
        item = action_by_id.get(image_id)
        if item is None or item["action"] == "keep":
            continue
        if item["action"] in {"drop", "uncertain"}:
            if args.allow_drop:
                dropped_ids.add(image_id)
            else:
                ignored_drop.append({"image_id": image_id, "action": item["action"]})
            continue
        if item["action"] == "relabel":
            corrected_label = item["corrected_label"]
            if not corrected_label:
                raise RuntimeError(f"Missing corrected_label for relabel action: {image_id}")
            if corrected_label not in STAGE_CN:
                raise RuntimeError(f"Invalid corrected_label for {image_id}: {corrected_label}")
            old_label = str(revised.at[index, "stage_id"])
            revised.at[index, "stage_id"] = corrected_label
            if "stage_cn" in revised.columns:
                revised.at[index, "stage_cn"] = STAGE_CN[corrected_label]
            relabeled.append({"image_id": image_id, "old_label": old_label, "new_label": corrected_label})

    revised = revised[~revised["image_id"].astype(str).isin(dropped_ids)].copy()
    out_manifest = args.out_dir / f"full_{args.split_name}_revised.csv"
    revised.to_csv(out_manifest, index=False, encoding="utf-8-sig")
    pd.DataFrame(relabeled).to_csv(
        args.out_dir / f"{args.split_name}_relabel_log.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame({"image_id": sorted(dropped_ids)}).to_csv(
        args.out_dir / f"{args.split_name}_drop_log.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(ignored_drop).to_csv(
        args.out_dir / f"{args.split_name}_ignored_drop_log.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = {
        "split_name": args.split_name,
        "original_count": int(len(manifest)),
        "revised_count": int(len(revised)),
        "drop_count": int(len(dropped_ids)),
        "ignored_drop_count": int(len(ignored_drop)),
        "relabel_count": int(len(relabeled)),
        "out_manifest": str(out_manifest),
    }
    (args.out_dir / f"{args.split_name}_revised_manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
