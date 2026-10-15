Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path "$PSScriptRoot\..\..\.."
Set-Location $RepoRoot

python src\paper_full_experiments\summarize_full_fusion_search.py `
  --root outputs\paper_full_experiments `
  --out-csv results\tables\full_fusion_search_summary.csv `
  --out-md results\tables\full_fusion_search_summary.md
