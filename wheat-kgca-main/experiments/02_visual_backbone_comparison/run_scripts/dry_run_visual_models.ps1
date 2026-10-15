param(
    [string[]]$Models = @(
        "dinov3_vits16",
        "siglip2_base_p16",
        "mambavision_t",
        "convnextv2_tiny",
        "mobilenetv4_hybrid_m",
        "swinv2_tiny",
        "eva02_small_p14",
        "resnet50"
    ),
    [string]$EnvName = "wheat_gpu",
    [switch]$UseCurrentPython,
    [string]$MambaVisionCheckpoint = ""
)

$ErrorActionPreference = "Stop"

$Runner = Join-Path $PSScriptRoot "run_visual_models_sequence.ps1"
if (-not (Test-Path -LiteralPath $Runner)) {
    throw "Sequence runner not found: $Runner"
}

$ArgsList = @(
    "-Models", $Models,
    "-Epochs", "1",
    "-BatchSize", "2",
    "-NumWorkers", "0",
    "-EnvName", $EnvName,
    "-DryRun",
    "-RunPrefix", "visual_dryrun"
)

if ($UseCurrentPython) { $ArgsList += "-UseCurrentPython" }
if ($MambaVisionCheckpoint.Trim().Length -gt 0) {
    $ArgsList += @("-MambaVisionCheckpoint", $MambaVisionCheckpoint)
}

& powershell -ExecutionPolicy Bypass -File $Runner @ArgsList
