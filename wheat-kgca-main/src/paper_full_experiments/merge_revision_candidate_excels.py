from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge split candidate CSVs into one Excel workbook.")
    parser.add_argument("--revision-dir", type=Path, required=True)
    parser.add_argument("--out-xlsx", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.out_xlsx, engine="openpyxl") as writer:
        for split in ["train", "val", "test"]:
            path = args.revision_dir / f"{split}_label_revision_candidates.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            frame.to_excel(writer, sheet_name=split, index=False)
    print(str(args.out_xlsx))


if __name__ == "__main__":
    main()
