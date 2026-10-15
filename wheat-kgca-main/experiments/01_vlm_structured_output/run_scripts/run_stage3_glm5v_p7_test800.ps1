$ErrorActionPreference = "Stop"

conda activate wheat_gpu
Set-Location "D:\2026.6.22"

if (-not $env:BIGMODEL_API_KEY) {
  throw "BIGMODEL_API_KEY is empty. Set it first, for example: `$env:BIGMODEL_API_KEY='your_glm_key'"
}

python .\wheat_original\src\stage3_vlm_model_comparison.py `
  --provider bigmodel `
  --api-key-env BIGMODEL_API_KEY `
  --api-base-url https://open.bigmodel.cn/api/paas/v4 `
  --models glm-5v-turbo `
  --prompt-ids P7 `
  --timeout 180 `
  --sleep 0.2 `
  --retry-attempts 3 `
  --retry-sleep 5 `
  --stop-after-consecutive-errors 10 `
  --continue-on-error
