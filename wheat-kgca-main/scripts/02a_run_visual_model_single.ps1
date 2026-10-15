param(
    [ValidateSet("convnextv2_tiny", "inception_next_tiny", "maxvit_tiny_rw_224", "mobilenetv4_hybrid_m", "repvit_tiny", "resnet50", "swinv2_tiny_window16_256")]
    [string]$Model = "convnextv2_tiny",
    [int]$Epochs = 30,
    [int]$BatchSize = 0,
    [int]$NumWorkers = 4,
    [string]$EnvName = "wheat_gpu",
    [switch]$UseCurrentPython,
    [switch]$DryRun,
    [switch]$NoPretrained,
    [switch]$NoAmp,
    [switch]$NoNoiseEval
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path "$PSScriptRoot\.."
$TrainScript = Join-Path $RepoRoot "src\training\train_src_models.py"
$SplitRoot = Join-Path $RepoRoot "data\data_split_811_full"
$OutputRoot = Join-Path $RepoRoot "outputs\src_model_training"

$ArgsList = @(
    $TrainScript,
    "--model", $Model,
    "--split-root", $SplitRoot,
    "--output-root", $OutputRoot,
    "--epochs", $Epochs,
    "--num-workers", $NumWorkers
)

if ($BatchSize -gt 0) {
    $ArgsList += @("--batch-size", $BatchSize)
}
if ($NoPretrained) {
    $ArgsList += "--no-pretrained"
}
if ($NoAmp) {
    $ArgsList += "--no-amp"
}
if ($NoNoiseEval) {
    $ArgsList += "--no-noise-eval"
}
if ($DryRun) {
    $ArgsList += "--dry-run"
}

Write-Host "Repository: $RepoRoot"
Write-Host "Model: $Model"
Write-Host "Output: $OutputRoot"

if ($UseCurrentPython) {
    python @ArgsList
} else {
    conda run -n $EnvName python @ArgsList
}
