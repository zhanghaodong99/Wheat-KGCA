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
    [int]$Epochs = 30,
    [int]$BatchSize = 32,
    [int]$NumWorkers = 4,
    [string]$EnvName = "wheat_gpu",
    [switch]$UseCurrentPython,
    [switch]$DryRun,
    [switch]$NoPretrained,
    [switch]$NoAmp,
    [switch]$NoNoiseEval,
    [switch]$StopOnError,
    [int]$MaxTrainPerClass = 0,
    [int]$MaxValPerClass = 0,
    [int]$MaxTestPerClass = 0,
    [string]$RunPrefix = "visual_seq",
    [string]$MambaVisionCheckpoint = ""
)

$ErrorActionPreference = "Stop"

$Runner = Join-Path $PSScriptRoot "run_single_visual_model.ps1"
if (-not (Test-Path -LiteralPath $Runner)) {
    throw "Single-model runner not found: $Runner"
}

$AllowedModels = @(
    "dinov3_vits16",
    "siglip2_base_p16",
    "mambavision_t",
    "convnextv2_tiny",
    "mobilenetv4_hybrid_m",
    "swinv2_tiny",
    "eva02_small_p14",
    "resnet50"
)

$Models = @(
    foreach ($Item in $Models) {
        foreach ($Part in ($Item -split ",")) {
            $Name = $Part.Trim()
            if ($Name.Length -gt 0) {
                $Name
            }
        }
    }
)

foreach ($Model in $Models) {
    if ($AllowedModels -notcontains $Model) {
        throw "Unknown model: $Model. Allowed: $($AllowedModels -join ', ')"
    }
}

$Failed = @()

foreach ($Model in $Models) {
    Write-Host ""
    Write-Host "================================================================================"
    Write-Host "Starting model: $Model"
    Write-Host "================================================================================"

    $CallArgs = @(
        "-Model", $Model,
        "-Epochs", "$Epochs",
        "-BatchSize", "$BatchSize",
        "-NumWorkers", "$NumWorkers",
        "-EnvName", $EnvName,
        "-MaxTrainPerClass", "$MaxTrainPerClass",
        "-MaxValPerClass", "$MaxValPerClass",
        "-MaxTestPerClass", "$MaxTestPerClass",
        "-RunPrefix", $RunPrefix
    )

    if ($UseCurrentPython) { $CallArgs += "-UseCurrentPython" }
    if ($DryRun) { $CallArgs += "-DryRun" }
    if ($NoPretrained) { $CallArgs += "-NoPretrained" }
    if ($NoAmp) { $CallArgs += "-NoAmp" }
    if ($NoNoiseEval) { $CallArgs += "-NoNoiseEval" }
    if ($MambaVisionCheckpoint.Trim().Length -gt 0) {
        $CallArgs += @("-MambaVisionCheckpoint", $MambaVisionCheckpoint)
    }

    try {
        & powershell -ExecutionPolicy Bypass -File $Runner @CallArgs
    }
    catch {
        $Failed += $Model
        Write-Error "Model failed: $Model. Error: $_"
        if ($StopOnError) {
            throw
        }
    }
}

Write-Host ""
Write-Host "================================================================================"
if ($Failed.Count -eq 0) {
    Write-Host "All requested models finished."
}
else {
    Write-Warning "Failed models: $($Failed -join ', ')"
}
Write-Host "================================================================================"
