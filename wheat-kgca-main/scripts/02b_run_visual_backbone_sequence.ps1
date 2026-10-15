param(
    [string[]]$Models = @(
        "convnextv2_tiny",
        "inception_next_tiny",
        "maxvit_tiny_rw_224",
        "mobilenetv4_hybrid_m",
        "repvit_tiny",
        "resnet50",
        "swinv2_tiny_window16_256"
    ),
    [int]$Epochs = 30,
    [int]$NumWorkers = 4,
    [string]$EnvName = "wheat_gpu",
    [switch]$UseCurrentPython,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Runner = Join-Path $PSScriptRoot "02a_run_visual_model_single.ps1"

foreach ($Model in $Models) {
    Write-Host ""
    Write-Host "===== Running visual backbone: $Model ====="
    $CallArgs = @(
        "-Model", $Model,
        "-Epochs", $Epochs,
        "-NumWorkers", $NumWorkers,
        "-EnvName", $EnvName
    )
    if ($UseCurrentPython) {
        $CallArgs += "-UseCurrentPython"
    }
    if ($DryRun) {
        $CallArgs += "-DryRun"
    }
    & $Runner @CallArgs
}
