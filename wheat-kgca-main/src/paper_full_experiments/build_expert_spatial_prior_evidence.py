from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


EXPERT_PRIORS: dict[str, list[dict[str, Any]]] = {
    "seedling": [
        {
            "grid": "lower_center",
            "organ": "young_seedling_leaves",
            "visibility": "yes",
            "confidence": 0.90,
            "reason": "Seedling-stage evidence is mainly expected in short young leaves near the lower plant region.",
        },
        {
            "grid": "center",
            "organ": "young_seedling_leaves",
            "visibility": "yes",
            "confidence": 0.70,
            "reason": "Young leaves can extend from the basal region toward the image center.",
        },
    ],
    "tillering": [
        {
            "grid": "lower_center",
            "organ": "basal_tillers",
            "visibility": "yes",
            "confidence": 0.95,
            "reason": "Tillering is identified by multiple shoots arising from the basal region.",
        },
        {
            "grid": "lower_left",
            "organ": "basal_tillers",
            "visibility": "yes",
            "confidence": 0.75,
            "reason": "Lateral tillers often spread around the lower canopy.",
        },
        {
            "grid": "lower_right",
            "organ": "basal_tillers",
            "visibility": "yes",
            "confidence": 0.75,
            "reason": "Lateral tillers often spread around the lower canopy.",
        },
    ],
    "stem_elongation": [
        {
            "grid": "center",
            "organ": "elongated_stem",
            "visibility": "yes",
            "confidence": 0.95,
            "reason": "Stem elongation is mainly indicated by elongated culms in the central vertical region.",
        },
        {
            "grid": "center_left",
            "organ": "elongated_stem",
            "visibility": "yes",
            "confidence": 0.70,
            "reason": "Elongated culms can occupy the central lateral canopy.",
        },
        {
            "grid": "center_right",
            "organ": "elongated_stem",
            "visibility": "yes",
            "confidence": 0.70,
            "reason": "Elongated culms can occupy the central lateral canopy.",
        },
    ],
    "flag_leaf_booting": [
        {
            "grid": "center",
            "organ": "flag_leaf_or_boot",
            "visibility": "yes",
            "confidence": 0.90,
            "reason": "Flag leaf and booting evidence is expected around upper stem sheaths before spike emergence.",
        },
        {
            "grid": "upper_center",
            "organ": "flag_leaf_or_boot",
            "visibility": "yes",
            "confidence": 0.80,
            "reason": "Booting structures are usually near the upper central stem.",
        },
        {
            "grid": "center_right",
            "organ": "flag_leaf_or_boot",
            "visibility": "yes",
            "confidence": 0.65,
            "reason": "Leaf sheath evidence may appear around the side of the upper stem.",
        },
    ],
    "heading": [
        {
            "grid": "upper_center",
            "organ": "emerged_spike",
            "visibility": "yes",
            "confidence": 0.95,
            "reason": "Heading is identified by emerged spikes, which are usually located in the upper image region.",
        },
        {
            "grid": "upper_left",
            "organ": "emerged_spike",
            "visibility": "yes",
            "confidence": 0.70,
            "reason": "Spike tips can appear in the upper lateral canopy.",
        },
        {
            "grid": "upper_right",
            "organ": "emerged_spike",
            "visibility": "yes",
            "confidence": 0.70,
            "reason": "Spike tips can appear in the upper lateral canopy.",
        },
    ],
    "flowering": [
        {
            "grid": "upper_center",
            "organ": "anthers_on_spike",
            "visibility": "yes",
            "confidence": 0.95,
            "reason": "Flowering evidence is expected on exposed spikes, especially visible anthers in the upper canopy.",
        },
        {
            "grid": "upper_left",
            "organ": "anthers_on_spike",
            "visibility": "yes",
            "confidence": 0.75,
            "reason": "Anthers can be distributed along upper lateral spikes.",
        },
        {
            "grid": "upper_right",
            "organ": "anthers_on_spike",
            "visibility": "yes",
            "confidence": 0.75,
            "reason": "Anthers can be distributed along upper lateral spikes.",
        },
    ],
    "grain_filling": [
        {
            "grid": "upper_center",
            "organ": "grain_filled_spike",
            "visibility": "yes",
            "confidence": 0.90,
            "reason": "Grain filling is mainly indicated by plump developing grains on spikes in the upper canopy.",
        },
        {
            "grid": "center",
            "organ": "grain_filled_spike",
            "visibility": "yes",
            "confidence": 0.75,
            "reason": "Lodging or image framing can place grain-filled spikes near the image center.",
        },
        {
            "grid": "upper_right",
            "organ": "grain_filled_spike",
            "visibility": "yes",
            "confidence": 0.65,
            "reason": "Spikelets can be distributed across upper lateral regions.",
        },
    ],
    "maturity": [
        {
            "grid": "upper_center",
            "organ": "yellowing_mature_spike",
            "visibility": "yes",
            "confidence": 0.85,
            "reason": "Maturity is commonly indicated by senescent spikes and yellowing upper canopy.",
        },
        {
            "grid": "center",
            "organ": "yellowing_canopy",
            "visibility": "yes",
            "confidence": 0.75,
            "reason": "Canopy yellowing often extends through the central image region.",
        },
        {
            "grid": "lower_center",
            "organ": "yellowing_canopy",
            "visibility": "yes",
            "confidence": 0.60,
            "reason": "Late senescence can affect lower canopy regions as well.",
        },
    ],
}


def build_evidence(manifest_path: Path, out_path: Path) -> None:
    manifest = pd.read_csv(manifest_path)
    rows = []
    for _, row in manifest.iterrows():
        stage = str(row["stage_id"])
        support = EXPERT_PRIORS.get(stage)
        if not support:
            raise ValueError(f"No expert spatial prior for stage_id={stage}")
        rows.append(
            {
                "image_id": row["image_id"],
                "image_key": Path(str(row["image_path"])).name.lower(),
                "split": row["split"],
                "stage_id": stage,
                "image_path": row["image_path"],
                "status": "ok",
                "schema_valid": True,
                "candidate_b": stage,
                "support_b_json": json.dumps(support, ensure_ascii=False),
                "support_b_count": len(support),
                "prior_source": "expert_spatial_prior",
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"rows={len(rows)}")
    print(f"out={out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full expert 3x3 spatial prior evidence CSVs.")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    build_evidence(args.train_manifest, args.out_dir / "expert_spatial_prior_train.csv")
    build_evidence(args.val_manifest, args.out_dir / "expert_spatial_prior_val.csv")

    (args.out_dir / "expert_spatial_prior_config.json").write_text(
        json.dumps(EXPERT_PRIORS, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )


if __name__ == "__main__":
    main()
