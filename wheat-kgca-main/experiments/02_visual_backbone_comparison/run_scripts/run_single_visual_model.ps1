param(
    [ValidateSet(
        "dinov3_vits16",
        "siglip2_base_p16",
        "mambavision_t",
        "convnextv2_tiny",
        "mobilenetv4_hybrid_m",
        "swinv2_tiny",
        "eva02_small_p14",
        "resnet50"
    )]
    [string]$Model = "convnextv2_tiny",

    [int]$Epochs = 30,
    [int]$BatchSize = 32,
    [int]$NumWorkers = 4,
    [ValidateSet("adamw", "sgd")]
    [string]$Optimizer = "adamw",
    [double]$Lr = 0.0003,
    [double]$WeightDecay = 0.05,
    [ValidateSet("cosine", "step", "none")]
    [string]$Scheduler = "cosine",
    [ValidateSet("val_macro_f1", "val_accuracy")]
    [string]$SelectionMetric = "val_macro_f1",
    [int]$EarlyStoppingPatience = 0,
    [int]$Seed = 42,
    [int]$NoiseSeed = 20260710,
    [string]$Device = "auto",
    [string]$EnvName = "wheat_gpu",
    [switch]$UseCurrentPython,
    [switch]$DryRun,
    [switch]$NoPretrained,
    [switch]$NoAmp,
    [switch]$NoNoiseEval,
    [int]$MaxTrainPerClass = 0,
    [int]$MaxValPerClass = 0,
    [int]$MaxTestPerClass = 0,
    [string]$RunPrefix = "visual_single",
    [string]$MambaVisionCheckpoint = ""
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
$TrainScript = Join-Path $ProjectRoot "src\training\visual_model_comparison.py"
$SplitRoot = Join-Path $ProjectRoot "data\data_split_811_full"
$SourceManifest = Join-Path $SplitRoot "wheat_split_811_manifest.csv"
$OutputRoot = Join-Path $ProjectRoot "outputs\visual_model_comparison"

if (-not (Test-Path -LiteralPath $TrainScript)) {
    throw "Training script not found: $TrainScript"
}
if (-not (Test-Path -LiteralPath $SplitRoot)) {
    throw "Split root not found: $SplitRoot"
}
if (-not (Test-Path -LiteralPath $SourceManifest)) {
    throw "Source manifest not found: $SourceManifest"
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunName = "${RunPrefix}_${Model}_${Timestamp}"

$ArgsList = @(
    $TrainScript,
    "--models", $Model,
    "--split-root", $SplitRoot,
    "--source-manifest", $SourceManifest,
    "--output-root", $OutputRoot,
    "--run-name", $RunName,
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--num-workers", "$NumWorkers",
    "--optimizer", $Optimizer,
    "--lr", "$Lr",
    "--weight-decay", "$WeightDecay",
    "--scheduler", $Scheduler,
    "--selection-metric", $SelectionMetric,
    "--early-stopping-patience", "$EarlyStoppingPatience",
    "--seed", "$Seed",
    "--noise-seed", "$NoiseSeed",
    "--device", $Device,
    "--max-train-per-class", "$MaxTrainPerClass",
    "--max-val-per-class", "$MaxValPerClass",
    "--max-test-per-class", "$MaxTestPerClass"
)

if ($NoPretrained) {
    $ArgsList += "--no-pretrained"
}
else {
    $ArgsList += "--pretrained"
}

if ($NoAmp) {
    $ArgsList += "--no-amp"
}
else {
    $ArgsList += "--amp"
}

if ($NoNoiseEval) {
    $ArgsList += "--no-noise-eval"
}
else {
    $ArgsList += "--noise-eval"
}

if ($DryRun) {
    $ArgsList += "--dry-run"
}

if ($Model -eq "mambavision_t") {
    if ($MambaVisionCheckpoint.Trim().Length -gt 0) {
        $ArgsList += @("--mambavision-checkpoint", $MambaVisionCheckpoint)
    }
    elseif ($env:MAMBAVISION_T_CHECKPOINT) {
        $ArgsList += @("--mambavision-checkpoint", $env:MAMBAVISION_T_CHECKPOINT)
    }
    else {
        Write-Warning "mambavision_t requires the official checkpoint. Pass -MambaVisionCheckpoint or set MAMBAVISION_T_CHECKPOINT."
    }
}

Write-Host "Project root: $ProjectRoot"
Write-Host "Model: $Model"
Write-Host "Run name: $RunName"
Write-Host "Output root: $OutputRoot"

Push-Location $ProjectRoot
try {
    if ($UseCurrentPython) {
        Write-Host "Running with current python."
        & python @ArgsList
    }
    else {
        Write-Host "Running with conda env: $EnvName"
        & conda run -n $EnvName python @ArgsList
    }
}
finally {
    Pop-Location
}

if (-not $DryRun) {
    $RunDir = Join-Path $OutputRoot $RunName
    Write-Host "Run directory: $RunDir"
    Write-Host "Summary file: $(Join-Path $RunDir 'visual_model_comparison_summary.md')"
}
