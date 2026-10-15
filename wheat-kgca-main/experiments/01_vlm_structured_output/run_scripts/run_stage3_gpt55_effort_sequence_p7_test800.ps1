param(
  [string]$Manifest = ".\wheat_original\outputs\vlm_comparison\manifests\vlm_model_test_balanced_800.csv",
  [string[]]$Efforts = @("low", "medium", "high"),
  [switch]$StopOnScriptFailure
)

$ErrorActionPreference = "Stop"

conda activate wheat_gpu
Set-Location "D:\2026.6.22"

if (-not $env:OPENAI_API_KEY) {
  throw "OPENAI_API_KEY is empty. Set it first, for example: [Environment]::SetEnvironmentVariable('OPENAI_API_KEY', 'your_openai_key', 'User')"
}

$baseUrl = if ($env:OPENAI_BASE_URL) { $env:OPENAI_BASE_URL } else { "https://subai-sg.lpkhbj.com/v1" }
$model = if ($env:OPENAI_VLM_MODEL) { $env:OPENAI_VLM_MODEL } else { "gpt-5.5" }
$outputRoot = ".\wheat_original\outputs\vlm_comparison\model_comparison"

function Get-EffortConfig {
  param([string]$Effort)
  switch ($Effort) {
    "low" {
      return @{
        Timeout = 180
        Sleep = 2
        RetryAttempts = 5
        RetrySleep = 20
        StopAfterConsecutiveErrors = 10
      }
    }
    "medium" {
      return @{
        Timeout = 240
        Sleep = 2
        RetryAttempts = 5
        RetrySleep = 20
        StopAfterConsecutiveErrors = 10
      }
    }
    "high" {
      return @{
        Timeout = 300
        Sleep = 2
        RetryAttempts = 6
        RetrySleep = 20
        StopAfterConsecutiveErrors = 5
      }
    }
    default {
      throw "Unsupported effort: $Effort. Use low, medium, or high."
    }
  }
}

function Get-LatestStage3Dir {
  if (-not (Test-Path $outputRoot)) {
    return $null
  }
  return Get-ChildItem -LiteralPath $outputRoot -Directory |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
}

function Print-Stage3Summary {
  param(
    [string]$Effort,
    [string]$Stage3Dir
  )

  $summaryPath = Join-Path $Stage3Dir "stage3_model_summary.csv"
  if (-not (Test-Path $summaryPath)) {
    Write-Host "[$Effort] No stage3_model_summary.csv found yet: $summaryPath"
    return $null
  }

  $row = Import-Csv -LiteralPath $summaryPath | Select-Object -First 1
  Write-Host ""
  Write-Host "===== $Effort summary ====="
  Write-Host "Run dir: $Stage3Dir"
  Write-Host ("Status: {0}" -f $row.status)
  Write-Host ("Total/OK/Errors: {0}/{1}/{2}" -f $row.total, $row.ok, $row.errors)
  Write-Host ("Acc: {0}; Macro-F1: {1}; Balanced Acc: {2}" -f $row.accuracy_valid_label_rows, $row.macro_f1_valid_label_rows, $row.balanced_accuracy_valid_label_rows)
  Write-Host ("JSON/Schema/Evidence: {0}/{1}/{2}" -f $row.json_valid_rate, $row.schema_valid_rate, $row.evidence_complete_rate)
  Write-Host ("Mean latency: {0}s" -f $row.mean_latency_s)
  return $row
}

Write-Host "GPT effort sequence run"
Write-Host "Model: $model"
Write-Host "Base URL: $baseUrl"
Write-Host "Manifest: $Manifest"
Write-Host "Efforts: $($Efforts -join ', ')"
Write-Host ""

foreach ($effort in $Efforts) {
  $config = Get-EffortConfig -Effort $effort

  $beforeLatest = Get-LatestStage3Dir
  Write-Host ""
  Write-Host "Running effort: $effort"
  Write-Host ("timeout={0}, sleep={1}, retry-attempts={2}, retry-sleep={3}, stop-after-consecutive-errors={4}" -f `
    $config.Timeout, $config.Sleep, $config.RetryAttempts, $config.RetrySleep, $config.StopAfterConsecutiveErrors)

  try {
    python .\wheat_original\src\stage3_vlm_model_comparison.py `
      --provider openai `
      --api-key-env OPENAI_API_KEY `
      --api-base-url $baseUrl `
      --manifest $Manifest `
      --models $model `
      --prompt-ids P7 `
      --reasoning-effort $effort `
      --openai-image-detail high `
      --openai-api-mode chat_completions `
      --timeout $config.Timeout `
      --sleep $config.Sleep `
      --retry-attempts $config.RetryAttempts `
      --retry-sleep $config.RetrySleep `
      --stop-after-consecutive-errors $config.StopAfterConsecutiveErrors `
      --continue-on-error
  }
  catch {
    Write-Host ""
    Write-Host "[$effort] script-level failure:"
    Write-Host $_
    if ($StopOnScriptFailure) {
      throw
    }
  }

  $afterLatest = Get-LatestStage3Dir
  if ($null -eq $afterLatest) {
    Write-Host "No stage3 output directory found after running $effort. Continuing."
    continue
  }
  if ($null -ne $beforeLatest -and $afterLatest.FullName -eq $beforeLatest.FullName) {
    Write-Host "Warning: latest stage3 directory did not change. Using latest directory anyway."
  }

  $summary = Print-Stage3Summary -Effort $effort -Stage3Dir $afterLatest.FullName
}

Write-Host ""
Write-Host "GPT effort sequence finished."
