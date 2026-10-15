param(
    [ValidateSet(
        "convnextv2_tiny",
        "inception_next_tiny",
        "maxvit_tiny_rw_224",
        "mobilenetv4_hybrid_m",
        "repvit_tiny",
        "resnet50",
        "swinv2_tiny_window16_256"
    )]
    [string]$Model = "swinv2_tiny_window16_256",

    [int]$Epochs = 30,
    [int]$BatchSize = 0,
    [int]$NumWorkers = 4,
    [string]$EnvName = "wheat_gpu",
    [switch]$UseCurrentPython,
    [switch]$DryRun,
    [switch]$NoPretrained,
    [switch]$NoAmp,
    [switch]$NoNoiseEval,
    [int]$MaxTrainPerClass = 0,
    [int]$MaxValPerClass = 0,
    [int]$MaxTestPerClass = 0,
    [string]$RunPrefix = "src_single"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
$TrainScript = Join-Path $ProjectRoot "src\training\train_src_models.py"
$SplitRoot = Join-Path $ProjectRoot "data\data_split_811_full"
$OutputRoot = Join-Path $ProjectRoot "outputs\src_model_training"

if (-not (Test-Path -LiteralPath $TrainScript)) {
    throw "Training script not found: $TrainScript"
}
if (-not (Test-Path -LiteralPath $SplitRoot)) {
    throw "Split root not found: $SplitRoot"
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunName = "${RunPrefix}_${Model}_${Timestamp}"

$ArgsList = @(
    $TrainScript,
    "--models", $Model,
    "--split-root", $SplitRoot,
    "--output-root", $OutputRoot,
    "--run-name", $RunName,
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--num-workers", "$NumWorkers",
    "--device", "auto",
    "--max-train-per-class", "$MaxTrainPerClass",
    "--max-val-per-class", "$MaxValPerClass",
    "--max-test-per-class", "$MaxTestPerClass"
)

if ($NoPretrained) { $ArgsList += "--no-pretrained" } else { $ArgsList += "--pretrained" }
if ($NoAmp) { $ArgsList += "--no-amp" } else { $ArgsList += "--amp" }
if ($NoNoiseEval) { $ArgsList += "--no-noise-eval" } else { $ArgsList += "--noise-eval" }
if ($DryRun) { $ArgsList += "--dry-run" }

Write-Host "Project root: $ProjectRoot"
Write-Host "Local src model: $Model"
Write-Host "Run name: $RunName"
Write-Host "Output root: $OutputRoot"

Push-Location $ProjectRoot
try {
    if ($UseCurrentPython) {
        & python @ArgsList
    }
    else {
        & conda run -n $EnvName python @ArgsList
    }
}
finally {
    Pop-Location
}

if (-not $DryRun) {
    $RunDir = Join-Path $OutputRoot $RunName
    Write-Host "Run directory: $RunDir"
    Write-Host "Summary file: $(Join-Path $RunDir 'local_model_comparison_summary.md')"
}
