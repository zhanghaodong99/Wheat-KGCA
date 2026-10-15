$ErrorActionPreference = "Stop"

conda activate wheat_gpu
Set-Location "D:\2026.6.22"

python .\wheat_original\src\stage2_vlm_prompt_screening.py `
  --manifest .\wheat_original\outputs\vlm_comparison\manifests\vlm_prompt_dev_400.csv `
  --prompt-ids P1,P2,P4,P6 `
  --model qwen3.7-plus `
  --temperature 0 `
  --sleep 0.2
