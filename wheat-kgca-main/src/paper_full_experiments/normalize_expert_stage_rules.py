from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_SOURCE = PROJECT_ROOT / "相关生成文档" / "expert_rules" / "小麦生育期标注规范_完整版.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "paper_full_experiments" / "expert_rules"


COLUMN_MAP = {
    "ID": "rule_id",
    "tag_type": "rule_type",
    "label_name": "stage_or_pair",
    "中文名称": "stage_or_pair_cn",
    "正向判据": "positive_criteria",
    "反向跳转条件": "negative_or_jump_criteria",
    "关键器官": "key_regions",
    "主标签": "primary_label",
    "次标签": "secondary_label",
    "score1": "primary_weight",
    "score2": "secondary_weight",
    "status": "review_status",
}


def normalize_rules(source: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(source, encoding="utf-8-sig")
    missing = [column for column in COLUMN_MAP if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Missing columns in expert rule csv: {missing}")

    normalized = frame.rename(columns=COLUMN_MAP)[list(COLUMN_MAP.values())].copy()
    for column in [
        "rule_id",
        "rule_type",
        "stage_or_pair",
        "stage_or_pair_cn",
        "positive_criteria",
        "negative_or_jump_criteria",
        "key_regions",
        "primary_label",
        "secondary_label",
        "review_status",
    ]:
        normalized[column] = normalized[column].fillna("").astype(str).str.strip()
    normalized["primary_weight"] = pd.to_numeric(normalized["primary_weight"], errors="coerce").fillna(0.0)
    normalized["secondary_weight"] = pd.to_numeric(normalized["secondary_weight"], errors="coerce").fillna(0.0)
    normalized["pair_id"] = normalized["stage_or_pair"].where(
        normalized["stage_or_pair"].str.contains("__vs__", regex=False),
        "",
    )
    normalized["is_hard_rule"] = normalized["rule_type"].isin(["stage", "boundary"])
    normalized["is_soft_rule"] = normalized["rule_type"].eq("soft_label")
    normalized["is_uncertain_rule"] = normalized["rule_type"].eq("uncertain")

    normalized_path = out_dir / "expert_stage_rules_normalized.csv"
    soft_path = out_dir / "expert_soft_label_rules.csv"
    region_path = out_dir / "expert_region_prior_rules.csv"
    summary_path = out_dir / "expert_stage_rules_summary.json"

    normalized.to_csv(normalized_path, index=False, encoding="utf-8-sig")
    normalized[normalized["is_soft_rule"]].to_csv(soft_path, index=False, encoding="utf-8-sig")
    normalized[["rule_id", "rule_type", "stage_or_pair", "primary_label", "key_regions"]].to_csv(
        region_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "source": str(source),
        "normalized": str(normalized_path),
        "soft_label_rules": str(soft_path),
        "region_prior_rules": str(region_path),
        "total_rules": int(len(normalized)),
        "rule_type_counts": {str(key): int(value) for key, value in normalized["rule_type"].value_counts().items()},
        "soft_label_pairs": sorted(
            str(value) for value in normalized.loc[normalized["is_soft_rule"], "pair_id"].dropna().unique() if value
        ),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize reviewed wheat stage expert rules.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    normalize_rules(args.source, args.out_dir)


if __name__ == "__main__":
    main()
