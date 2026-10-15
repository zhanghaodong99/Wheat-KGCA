param(
    [string[]]$Models = @(
        "convnextv2_tiny",
        "convnextv2_tiny_res_coordinate",
        "convnextv2_tiny_res_eca2"
    ),
    [string]$SplitRoot = "D:\2026.6.22\wheat_original\data\data_split_811_full",
    [int]$Epochs = 10,
    [int]$BatchSize = 16,
    [int]$NumWorkers = 0,
    [int]$MaxTrainPerClass = 200,
    [int]$MaxValPerClass = 0,
    [int]$MaxTestPerClass = 0,
    [string]$EnvName = "wheat_gpu",
    [switch]$UseCurrentPython,
    [switch]$NoPretrained,
    [switch]$NoAmp,
    [switch]$NoNoiseEval
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path "$PSScriptRoot\.."
$TrainScript = Join-Path $ProjectRoot "src\training\train_src_models.py"
$OutputRoot = Join-Path $ProjectRoot "outputs\convnextv2_top_module_validation"
$RunName = "convnextv2_top_modules_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
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
    "--continue-on-error"
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

if ($NoNoiseEval) {
    $ArgsList += "--no-noise-eval"
} else {
    $ArgsList += "--noise-eval"
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
