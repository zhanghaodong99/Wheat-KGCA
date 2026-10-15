# Experiment 05: Final Ablation

This folder contains the final result summarization scripts for the paper-submission version of Wheat-KGCA.

## Purpose

The final ablation should report only the main model path and the necessary controls:

| ID | Variant | Question |
|---|---|---|
| A0 | ConvNeXtV2 baseline | How strong is visual-only classification? |
| A1 | RCA | Does residual coordinate attention help? |
| A2 | Structured VLM semantic gate | Does structured agronomic evidence help? |
| A3 | Knowledge-constrained loss | Does adjacent-stage prior reduce unreasonable confusion? |
| A4 | Boundary-rule enhancement | Does revised expert knowledge improve boundary cases? |
| A5 | CAM analysis | Do attention regions become more agronomically meaningful? |

## Included Mainline Code

- `src/summarize_image_level_ablation.py`
- `src/summarize_full_fusion_search.py`
- `src/diagnose_error_mechanisms.py`

Additional reusable summary scripts are stored under:

- `../../src/paper_full_experiments/`

## Run Order

Run from the repository root:

```powershell
.\experiments\05_final_ablation\run_scripts\01_summarize_image_level_ablation.ps1
.\experiments\05_final_ablation\run_scripts\02_summarize_full_fusion_search.ps1
.\experiments\05_final_ablation\run_scripts\03_diagnose_error_mechanisms.ps1
```

The output tables should be copied to `results/tables/` and the final figures to `results/figures/` before paper submission.
