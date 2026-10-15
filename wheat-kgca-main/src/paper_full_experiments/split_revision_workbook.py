from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split reviewed revision workbook into per-split CSV files.")
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    workbook = pd.ExcelFile(args.workbook)
    for split in ["train", "val", "test"]:
        if split not in workbook.sheet_names:
            continue
        frame = pd.read_excel(args.workbook, sheet_name=split)
        out_csv = args.out_dir / f"{split}_label_revision_reviewed.csv"
        frame.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"{split}: {len(frame)} -> {out_csv}")


if __name__ == "__main__":
    main()
