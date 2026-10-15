# Experiment 04: CAM Alignment

This folder contains the CAM and expert-spatial-prior code used to explain how Wheat-KGCA changes attention regions relative to the visual baseline.

## Purpose

The experiment supports the paper claim that expert/VLM evidence should not only change logits, but should also make the model attend to agronomically meaningful regions such as flag leaves, spikes, anthers, tillers, and grain-filling regions.

## Included Mainline Code

- `src/build_expert_spatial_prior_evidence.py`
- `src/wheat_kgca_build_evidence_assets.py`
- `src/wheat_kgca_select_final_samples.py`
- `src/wheat_kgca_cam_compare.py`

Shared CAM implementations are stored under:

- `../../src/fusion/stage2_cam/`

## Run Order

Run from the repository root:

```powershell
.\experiments\04_cam_alignment\run_scripts\01_build_evidence_assets.ps1
.\experiments\04_cam_alignment\run_scripts\02_build_expert_spatial_priors.ps1
.\experiments\04_cam_alignment\run_scripts\03_select_final_cam_samples.ps1
.\experiments\04_cam_alignment\run_scripts\04_generate_cam_compare.ps1
```

The selected CAM figures are used for the paper's explanation section and should be reported together with boundary-case examples.
