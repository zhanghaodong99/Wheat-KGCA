Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path "$PSScriptRoot\..\..\.."
Set-Location $RepoRoot

python src\paper_full_experiments\summarize_image_level_ablation.py `
  --root outputs\paper_full_experiments `
  --out-csv results\tables\image_level_ablation_summary.csv `
  --out-md results\tables\image_level_ablation_summary.md
