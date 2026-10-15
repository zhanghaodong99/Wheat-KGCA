param(
    [string[]]$Models = @(
        "convnextv2_tiny",
        "convnextv2_tiny_eca",
        "convnextv2_tiny_se",
        "convnextv2_tiny_spatial",
        "convnextv2_tiny_coordinate",
        "convnextv2_tiny_mlp_head"
    ),
    [string]$SplitRoot = "D:\2026.6.22\wheat_original\data\data_split_811_full",
    [int]$Epochs = 3,
    [int]$BatchSize = 16,
    [int]$NumWorkers = 0,
    [int]$MaxTrainPerClass = 40,
    [int]$MaxValPerClass = 20,
    [int]$MaxTestPerClass = 20,
    [string]$EnvName = "wheat_gpu",
    [switch]$UseCurrentPython,
    [switch]$NoPretrained,
    [switch]$NoAmp
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path "$PSScriptRoot\.."
$TrainScript = Join-Path $ProjectRoot "src\training\train_src_models.py"
$OutputRoot = Join-Path $ProjectRoot "outputs\convnextv2_module_smoke"
$RunName = "convnextv2_modules_smoke_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
$ModelText = $Models -join ","

$ArgsList = @(
    $TrainScript,
    "--models", $ModelText,
    "--split-root", $SplitRoot,
    "--output-root", $OutputRoot,
    "--run-name", $RunName,
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--num-workers", "$NumWorkers",
    "--max-train-per-class", "$MaxTrainPerClass",
    "--max-val-per-class", "$MaxValPerClass",
    "--max-test-per-class", "$MaxTestPerClass",
    "--selection-metric", "val_macro_f1",
    "--continue-on-error",
    "--no-noise-eval"
)

if ($NoPretrained) {
    $ArgsList += "--no-pretrained"
} else {
    $ArgsList += "--pretrained"
}

if ($NoAmp) {
    $ArgsList += "--no-amp"
} else {
    $ArgsList += "--amp"
}

Write-Host "Project root: $ProjectRoot"
Write-Host "Models: $ModelText"
Write-Host "Split root: $SplitRoot"
Write-Host "Run name: $RunName"
Write-Host "Output root: $OutputRoot"

Push-Location $ProjectRoot
try {
    if ($UseCurrentPython) {
        python @ArgsList
    } else {
        conda run -n $EnvName python @ArgsList
    }
}
finally {
    Pop-Location
}
