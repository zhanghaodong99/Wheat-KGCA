$ErrorActionPreference = "Stop"

conda activate wheat_gpu
Set-Location "D:\2026.6.22"

if (-not $env:MOONSHOT_API_KEY) {
  throw "MOONSHOT_API_KEY is empty. Set it first, for example: `$env:MOONSHOT_API_KEY='your_kimi_key'"
}

python .\wheat_original\src\stage3_vlm_model_comparison.py `
  --provider kimi `
  --api-key-env MOONSHOT_API_KEY `
  --api-base-url https://api.moonshot.cn/v1 `
  --models kimi-k2.7-code-highspeed,kimi-k2.7-code,kimi-k2.6 `
  --prompt-ids P7 `
  --limit 5 `
  --timeout 180 `
  --sleep 0.2 `
  --continue-on-error

