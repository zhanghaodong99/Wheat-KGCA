Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path "$PSScriptRoot\..\..\.."
Set-Location $RepoRoot

python src\fusion\stage6_hierarchical_heads\train_hierarchical_aux_heads.py `
  --train-manifest outputs\paper_full_experiments\manifests\full_train_19877.csv `
  --val-manifest outputs\paper_full_experiments\manifests\full_val_2485.csv `
  --test-manifest outputs\paper_full_experiments\manifests\full_test_2485.csv `
  --semantic-train outputs\paper_full_experiments\vlm_semantics\full_train_19877\parsed_results.csv `
  --semantic-val outputs\paper_full_experiments\vlm_semantics\full_val_2485\parsed_results.csv `
  --semantic-test outputs\paper_full_experiments\vlm_semantics\full_test_2485\parsed_results.csv `
  --checkpoint outputs\src_model_training\src_single_convnextv2_tiny_20260713_092041\convnextv2_tiny\best_model.pth `
  --out-dir outputs\paper_full_experiments\hierarchical_03_aux_heads_cam_valref_select_main_output `
  --epochs 30 `
  --batch-size 32 `
  --lambda-align 0.0025 `
  --lambda-pair 0.2 `
  --selection-target main `
  --primary-output main
