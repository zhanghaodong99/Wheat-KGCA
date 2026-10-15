Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path "$PSScriptRoot\..\..\.."
Set-Location $RepoRoot

python src\paper_full_experiments\build_expert_spatial_prior_evidence.py `
  --train-manifest outputs\paper_full_experiments\manifests\full_train_19877.csv `
  --val-manifest outputs\paper_full_experiments\manifests\full_val_2485.csv `
  --out-dir outputs\paper_full_experiments\expert_spatial_priors
