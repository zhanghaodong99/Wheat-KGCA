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
  --disable-semantic `
  --disable-gate `
  --amp `
  --out-dir outputs\paper_full_experiments\image_ablation_01_visual_only_head_tune
