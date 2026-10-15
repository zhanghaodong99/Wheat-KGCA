Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path "$PSScriptRoot\.."
Set-Location $RepoRoot

$ApiBaseUrl = $env:OPENAI_BASE_URL
if ([string]::IsNullOrWhiteSpace($ApiBaseUrl)) {
  $ApiBaseUrl = "https://api.openai.com/v1"
}

python src\vlm\stage3_vlm_model_comparison.py `
  --manifest experiments\01_vlm_structured_output\manifests\vlm_model_test_balanced_800.csv `
  --prompt-dir experiments\01_vlm_structured_output\prompts `
  --prompt-ids P11 `
  --provider openai `
  --api-key-env OPENAI_API_KEY `
  --api-base-url $ApiBaseUrl `
  --models gpt-6 `
  --openai-api-mode responses `
  --openai-image-detail high `
  --reasoning-effort low `
  --timeout 300 `
  --retry-attempts 6 `
  --retry-sleep 20 `
  --stop-after-consecutive-errors 10
