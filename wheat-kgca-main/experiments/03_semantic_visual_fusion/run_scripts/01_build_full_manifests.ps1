Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path "$PSScriptRoot\..\..\.."
Set-Location $RepoRoot

python src\paper_full_experiments\build_full_fusion_manifests.py `
  --source data\data_split_811_full\wheat_split_811_manifest.csv `
  --out-dir outputs\paper_full_experiments\manifests
