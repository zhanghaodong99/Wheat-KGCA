$ErrorActionPreference = "Stop"

conda activate wheat_gpu
Set-Location "D:\2026.6.22"

if (-not $env:OPENAI_API_KEY) {
  throw "OPENAI_API_KEY is empty. Set it first, for example: [Environment]::SetEnvironmentVariable('OPENAI_API_KEY', 'your_openai_key', 'User')"
}

$baseUrl = if ($env:OPENAI_BASE_URL) { $env:OPENAI_BASE_URL } else { "https://subai-sg.lpkhbj.com/v1" }
$model = if ($env:OPENAI_VLM_MODEL) { $env:OPENAI_VLM_MODEL } else { "gpt-5.5" }

python .\wheat_original\src\stage3_vlm_model_comparison.py `
  --provider openai `
  --api-key-env OPENAI_API_KEY `
  --api-base-url $baseUrl `
  --models $model `
  --prompt-ids P7 `
  --reasoning-effort high `
  --openai-image-detail high `
  --openai-api-mode chat_completions `
  --timeout 300 `
  --sleep 2 `
  --retry-attempts 6 `
  --retry-sleep 20 `
  --stop-after-consecutive-errors 10 `
  --continue-on-error
