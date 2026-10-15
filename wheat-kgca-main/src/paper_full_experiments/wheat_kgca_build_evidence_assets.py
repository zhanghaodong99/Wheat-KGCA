from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


STAGE_RULES = [
    {
        "stage_id": "seedling",
        "stage_cn": "苗期",
        "required_evidence": "Low young plants; single main shoots or no clearly formed basal tillers; no internode elongation; no reproductive organs.",
        "supporting_evidence": "Sparse low canopy, narrow leaves, no visible spike, no booting structure.",
        "exclusion_evidence": "Multiple clearly visible basal tillers; obvious stem elongation; booting or emerged spike.",
        "adjacent_confusers": "tillering",
        "key_region": "basal plant region",
    },
    {
        "stage_id": "tillering",
        "stage_cn": "分蘖期",
        "required_evidence": "Multiple formed basal tillers are visible; no obvious internode elongation; no reproductive organs.",
        "supporting_evidence": "Clustered low shoots from the basal region; vegetative green canopy.",
        "exclusion_evidence": "Only single young shoots with no formed tillers; clear stem elongation; booting or spike evidence.",
        "adjacent_confusers": "seedling, stem_elongation",
        "key_region": "basal plant region",
    },
    {
        "stage_id": "stem_elongation",
        "stage_cn": "拔节期",
        "required_evidence": "Elongated stems or internode elongation; no booting sheath swelling; no emerged spike.",
        "supporting_evidence": "Upright taller vegetative canopy, visible culm elongation.",
        "exclusion_evidence": "Flag leaf sheath swollen with enclosed spike; emerged spike; fresh anthers.",
        "adjacent_confusers": "tillering, flag_leaf_booting",
        "key_region": "stem and upper canopy region",
    },
    {
        "stage_id": "flag_leaf_booting",
        "stage_cn": "孕穗期",
        "required_evidence": "Flag leaf sheath swelling or booting structure; spike still enclosed in the sheath.",
        "supporting_evidence": "Upper stem swelling, flag leaf evidence, no visible emerged spike.",
        "exclusion_evidence": "Spike already emerged from sheath; visible fresh anthers.",
        "adjacent_confusers": "stem_elongation, heading",
        "key_region": "upper stem and flag leaf sheath region",
    },
    {
        "stage_id": "heading",
        "stage_cn": "抽穗期",
        "required_evidence": "Spike emerged from sheath; no clearly visible fresh white or yellow anthers.",
        "supporting_evidence": "Green fully or partly emerged spikes, no anthesis evidence.",
        "exclusion_evidence": "Clear exposed white/yellow anthers; enlarged grains with no fresh anthers.",
        "adjacent_confusers": "flag_leaf_booting, flowering",
        "key_region": "spike region",
    },
    {
        "stage_id": "flowering",
        "stage_cn": "开花期",
        "required_evidence": "Clear exposed fresh white or yellow anthers; flowering evidence has priority over heading.",
        "supporting_evidence": "Fully emerged green spikes, visible anther-like protrusions from spikelets.",
        "exclusion_evidence": "No fresh anthers and closed spikelets; only glume color or sunlight reflection; enlarged grains.",
        "adjacent_confusers": "heading, grain_filling",
        "key_region": "spikelet and anther region",
    },
    {
        "stage_id": "grain_filling",
        "stage_cn": "灌浆期",
        "required_evidence": "No fresh exposed anthers; spikelets closed; grains enlarged or filling; spike still mainly green.",
        "supporting_evidence": "Plump green spikes, early yellowing may appear, dry or absent anthers.",
        "exclusion_evidence": "Fresh white/yellow exposed anthers; mostly dry mature plant.",
        "adjacent_confusers": "flowering, maturity",
        "key_region": "spike and grain region",
    },
    {
        "stage_id": "maturity",
        "stage_cn": "成熟期",
        "required_evidence": "Plants and spikes mostly yellow or dry; grains mature.",
        "supporting_evidence": "Dry canopy, straw color, mature spike appearance.",
        "exclusion_evidence": "Mostly green spikes with developing grains; fresh anthers.",
        "adjacent_confusers": "grain_filling",
        "key_region": "whole plant and spike region",
    },
]


ADJACENT_EDGES = [
    ("seedling", "tillering"),
    ("tillering", "stem_elongation"),
    ("stem_elongation", "flag_leaf_booting"),
    ("flag_leaf_booting", "heading"),
    ("heading", "flowering"),
    ("flowering", "grain_filling"),
    ("grain_filling", "maturity"),
]


PROMPT = """You are an agronomic visual evidence extractor for wheat growth-stage recognition.

Your task is NOT to guess from the folder name or acquisition time. Inspect the visible image evidence and output structured JSON only.

Use these boundary rules:
1. seedling vs tillering:
   - seedling: young low plants, single main shoots or no clearly formed basal tillers.
   - tillering: multiple clearly visible basal tillers; no stem elongation or reproductive organs.
   - if the basal region is occluded, do not infer tillering unless multiple low basal shoots are clearly visible.
2. heading vs flowering:
   - heading: spike emerged but no clear fresh white/yellow anthers.
   - flowering: clear exposed fresh white/yellow anthers; visible anthers have priority over heading.
3. flowering vs grain_filling:
   - flowering: fresh exposed anthers are visible.
   - grain_filling: no fresh anthers, spikelets closed, grains enlarged/filling, spike still mainly green.
4. flag_leaf_booting vs heading:
   - flag_leaf_booting: spike still enclosed in swollen flag leaf sheath.
   - heading: spike emerged from sheath.
5. grain_filling vs maturity:
   - grain_filling: green spike with filling grains.
   - maturity: plant and spike mostly yellow/dry with mature grains.

Return JSON only with this schema. Do not add Markdown fences.
{
  "free_observation": "one concise visual observation sentence",
  "organ_scan": {
    "spike_status": "not_visible" | "booting" | "partly_emerged" | "fully_emerged" | "full_grain" | "mature_dry" | "uncertain",
    "anther_visible": "yes" | "no" | "uncertain",
    "grain_fullness": "none" | "slight" | "moderate" | "clear" | "full" | "uncertain",
    "flag_leaf_booting_clue": "yes" | "no" | "uncertain",
    "tillering_clue": "yes" | "no" | "uncertain",
    "stem_elongation_clue": "yes" | "no" | "uncertain",
    "plant_color": "green" | "mixed_yellow" | "yellow_dry" | "uncertain",
    "field_noise": ["occlusion", "blur", "dense_canopy", "illumination", "background", "none"]
  },
  "evidence_scores": {
    "spike_visibility": number between 0 and 1,
    "anther_visibility": number between 0 and 1,
    "grain_fullness_score": number between 0 and 1,
    "flag_leaf_booting_strength": number between 0 and 1,
    "tillering_strength": number between 0 and 1,
    "stem_elongation_strength": number between 0 and 1,
    "yellowing_degree": number between 0 and 1,
    "noise_degree": number between 0 and 1
  },
  "rule_check": {
    "boundary_pair_if_any": "none or label_a/label_b",
    "key_visible_evidence": ["short evidence strings"],
    "missing_or_occluded_evidence": ["short missing or occluded evidence strings"]
  },
  "image_quality": "good" | "medium" | "poor",
  "predicted_stage": one of ["seedling","tillering","stem_elongation","flag_leaf_booting","heading","flowering","grain_filling","maturity"],
  "second_candidate": one of the same labels,
  "confidence": number between 0 and 1,
  "short_reason": "one concise sentence based only on visible evidence",
  "uncertainty_reason": "one concise sentence, or empty string"
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Wheat-KGCA evidence dictionary, adjacency matrix, and boundary-aware VLM prompt.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/paper_full_experiments/wheat_kgca_final"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rules = pd.DataFrame(STAGE_RULES)
    rules.to_csv(args.out_dir / "agronomic_evidence_dictionary.csv", index=False, encoding="utf-8-sig")

    stages = [row["stage_id"] for row in STAGE_RULES]
    matrix = pd.DataFrame(0, index=stages, columns=stages)
    for stage in stages:
        matrix.loc[stage, stage] = 1
    for left, right in ADJACENT_EDGES:
        matrix.loc[left, right] = 1
        matrix.loc[right, left] = 1
    matrix.to_csv(args.out_dir / "adjacent_stage_matrix.csv", encoding="utf-8-sig")

    (args.out_dir / "boundary_aware_vlm_prompt.txt").write_text(PROMPT, encoding="utf-8")
    (args.out_dir / "evidence_assets_summary.json").write_text(
        json.dumps(
            {
                "stage_count": len(STAGE_RULES),
                "adjacent_edges": ADJACENT_EDGES,
                "files": [
                    "agronomic_evidence_dictionary.csv",
                    "adjacent_stage_matrix.csv",
                    "boundary_aware_vlm_prompt.txt",
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote Wheat-KGCA evidence assets to: {args.out_dir}")


if __name__ == "__main__":
    main()
