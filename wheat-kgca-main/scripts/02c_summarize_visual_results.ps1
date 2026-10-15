param(
    [string]$EnvName = "wheat_gpu",
    [switch]$UseCurrentPython
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path "$PSScriptRoot\.."
$Script = Join-Path $RepoRoot "src\evaluation\summarize_visual_model_runs.py"
$OutputRoot = Join-Path $RepoRoot "outputs\src_model_training"
$Manifest = Join-Path $RepoRoot "data\data_split_811_full\wheat_split_811_manifest.csv"
$AnalysisRoot = Join-Path $RepoRoot "_analysis_outputs\visual_model_supplementary_analysis"

$ArgsList = @(
    $Script,
    "--output-root", $OutputRoot,
    "--manifest", $Manifest,
    "--analysis-root", $AnalysisRoot
)

if ($UseCurrentPython) {
    python @ArgsList
} else {
    conda run -n $EnvName python @ArgsList
}
