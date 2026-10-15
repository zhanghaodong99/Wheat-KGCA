Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path "$PSScriptRoot\..\..\.."
Set-Location $RepoRoot

python src\paper_full_experiments\wheat_kgca_build_evidence_assets.py `
  --out-dir outputs\paper_full_experiments\wheat_kgca_final
