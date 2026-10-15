Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path "$PSScriptRoot\..\..\.."
Set-Location $RepoRoot

python src\paper_full_experiments\diagnose_error_mechanisms.py `
  --manifest outputs\paper_full_experiments\manifests\full_test_2485.csv `
  --baseline-predictions outputs\paper_full_experiments\convnext_features\full_test_2485\convnext_predictions.csv `
  --wheat-sage-cam-predictions outputs\paper_full_experiments\hierarchical_03_aux_heads_cam_valref_select_main_output\test_predictions.csv `
  --skip-attention15 `
  --out-dir outputs\paper_full_experiments\error_diagnosis
