Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path "$PSScriptRoot\..\..\.."
Set-Location $RepoRoot

python src\fusion\train_full_image_level_wheat_sage.py `
  --train-manifest outputs\paper_full_experiments\manifests\full_train_19877.csv `
  --val-manifest outputs\paper_full_experiments\manifests\full_val_2485.csv `
  --test-manifest outputs\paper_full_experiments\manifests\full_test_2485.csv `
  --semantic-train outputs\paper_full_experiments\vlm_semantics\full_train_19877\parsed_results.csv `
  --semantic-val outputs\paper_full_experiments\vlm_semantics\full_val_2485\parsed_results.csv `
  --semantic-test outputs\paper_full_experiments\vlm_semantics\full_test_2485\parsed_results.csv `
  --checkpoint outputs\src_model_training\src_single_convnextv2_tiny_20260713_092041\convnextv2_tiny\best_model.pth `
  --epochs 30 `
  --batch-size 32 `
  --num-workers 0 `
  --lr 0.0003 `
  --freeze-visual encoder `
  --aux-weight 0.2 `
  --gate-weight 0.02 `
  --gate-open-weight 0.03 `
  --gate-open-target 0.08 `
  --anchor-weight 0.20 `
  --switch-weight 0.05 `
  --delta-weight 0.005 `
  --low-conf-focus-weight 1.5 `
  --max-confidence 0.95 `
  --max-margin 0.65 `
  --high-confidence-threshold 0.97 `
  --high-margin-threshold 0.70 `
  --preserve-margin 0.05 `
  --eligibility-sharpness 12 `
  --delta-scale 2.0 `
  --gate-init-bias -0.5 `
  --min-effective-gate 0.02 `
  --selection-mode switch_gain `
  --amp `
  --out-dir outputs\paper_full_experiments\image_ablation_03b_relaxed_gate
