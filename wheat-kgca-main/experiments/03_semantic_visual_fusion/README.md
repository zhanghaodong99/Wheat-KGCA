# Experiment 03: Semantic-Visual Fusion

This folder contains the main Wheat-KGCA training path after the VLM and visual-backbone comparison experiments.

## Purpose

The experiment tests whether structured VLM agronomic evidence can improve a strong ConvNeXtV2 visual baseline when it is introduced as a controlled semantic signal rather than as a direct classifier.

## Included Mainline Code

- `src/build_full_fusion_manifests.py`
- `src/train_image_level_semantic_fusion.py`
- `src/train_unified_aux_heads.py`

Shared implementations are stored under:

- `../../src/fusion/`
- `../../src/training/knowledge_losses.py`
- `../../src/fusion/semantic_gate.py`

## Main Variants

| ID | Variant | Purpose |
|---|---|---|
| F0 | Visual-only image-level control | Checks whether the training wrapper preserves the ConvNeXtV2 baseline. |
| F1 | Semantic vector without gate | Tests whether raw VLM semantics are useful. |
| F2 | Semantic residual gate | Main VLM-guided fusion route. |
| F3 | Unified adjacent auxiliary heads | Keeps one model, one 8-class main output, with internal adjacent-pair heads. |

## Run Order

Run from the repository root:

```powershell
.\experiments\03_semantic_visual_fusion\run_scripts\01_build_full_manifests.ps1
.\experiments\03_semantic_visual_fusion\run_scripts\02_extract_convnext_features.ps1
.\experiments\03_semantic_visual_fusion\run_scripts\03_train_visual_only_control.ps1
.\experiments\03_semantic_visual_fusion\run_scripts\04_train_semantic_gate.ps1
.\experiments\03_semantic_visual_fusion\run_scripts\05_train_unified_aux_heads.ps1
```

The scripts assume that image manifests, cached VLM semantic CSV files, and the visual baseline checkpoint are available under `outputs/`. See `../../checkpoints/README.md` and `../../data/README.md`.
