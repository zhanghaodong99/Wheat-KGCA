# Experiment 02: Visual Backbone Comparison

This experiment evaluates multiple visual backbones for eight-class wheat growth-stage recognition. It is the visual-only comparison before adding VLM semantic evidence, attention fusion, and boundary-aware constraints.

## Purpose

1. Select a strong visual backbone for later Wheat-KGCA fusion experiments.
2. Compare clean validation/test accuracy with noise-perturbed robustness.
3. Identify the main adjacent-stage confusions through confusion matrices.
4. Provide training curves and timing information for the paper's baseline section.

## Included Code

```text
src/
|-- visual_model_comparison.py
|-- train_src_models.py
`-- summarize_visual_model_runs.py

run_scripts/
|-- dry_run_visual_models.ps1
|-- run_single_src_model.ps1
|-- run_single_visual_model.ps1
`-- run_visual_models_sequence.ps1
```

Top-level reproducible entry scripts are also provided:

```text
scripts/02a_run_visual_model_single.ps1
scripts/02b_run_visual_backbone_sequence.ps1
scripts/02c_summarize_visual_results.ps1
```

## Visual Backbones

The packaged training code supports:

```text
convnextv2_tiny
inception_next_tiny
maxvit_tiny_rw_224
mobilenetv4_hybrid_m
repvit_tiny
resnet50
swinv2_tiny_window16_256
```

The separate visual comparison script also contains extended timm candidates used in earlier screening:

```text
dinov3_vits16
siglip2_base_p16
mambavision_t
eva02_small_p14
```

## Run

From the repository root:

```powershell
.\scripts\02a_run_visual_model_single.ps1 -Model convnextv2_tiny -Epochs 30 -EnvName wheat_gpu
```

Run all listed backbones:

```powershell
.\scripts\02b_run_visual_backbone_sequence.ps1 -Epochs 30 -EnvName wheat_gpu
```

Summarize completed runs:

```powershell
.\scripts\02c_summarize_visual_results.ps1 -EnvName wheat_gpu
```

## Results

The integrated results copied from the working experiments are organized as:

```text
results/
|-- tables/
|   |-- visual_model_final_summary.csv
|   |-- per_class_metrics.csv
|   `-- per_source_metrics.csv
|-- figures/
|   |-- fig_01_training_val_macro_f1.png
|   |-- fig_02_clean_noise_accuracy_bar.png
|   |-- fig_03_accuracy_drop_bar.png
|   `-- fig_04_training_time_bar.png
`-- confusion_matrices/
    |-- clean/
    `-- noise/
```

## Notes

This part does not use VLM calls and does not require any private API credential. It is a pure visual-model baseline and robustness comparison.
