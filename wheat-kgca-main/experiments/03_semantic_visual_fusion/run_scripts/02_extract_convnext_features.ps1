Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path "$PSScriptRoot\..\..\.."
Set-Location $RepoRoot

$Manifests = @(
  "outputs\paper_full_experiments\manifests\full_train_19877.csv",
  "outputs\paper_full_experiments\manifests\full_val_2485.csv",
  "outputs\paper_full_experiments\manifests\full_test_2485.csv"
)

foreach ($Manifest in $Manifests) {
  python src\fusion\extract_convnext_features.py `
    --manifest $Manifest `
    --checkpoint outputs\src_model_training\src_single_convnextv2_tiny_20260713_092041\convnextv2_tiny\best_model.pth `
    --out-dir outputs\paper_full_experiments\convnext_features `
    --batch-size 64 `
    --num-workers 0
}
