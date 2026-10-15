Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path "$PSScriptRoot\..\..\.."
Set-Location $RepoRoot

python src\paper_full_experiments\wheat_kgca_cam_compare.py `
  --samples outputs\paper_full_experiments\wheat_kgca_final\wheat_kgca_selected_samples.csv `
  --semantic-test outputs\paper_full_experiments\vlm_semantics\full_test_2485\parsed_results.csv `
  --baseline-checkpoint outputs\src_model_training\src_single_convnextv2_tiny_20260713_092041\convnextv2_tiny\best_model.pth `
  --h03-checkpoint outputs\paper_full_experiments\hierarchical_03_aux_heads_cam_valref_select_main_output\best_model.pth `
  --out-dir outputs\paper_full_experiments\wheat_kgca_final\cam_compare `
  --target true `
  --limit 48
