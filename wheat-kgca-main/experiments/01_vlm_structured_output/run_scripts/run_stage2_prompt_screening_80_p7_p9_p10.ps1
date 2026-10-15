$ErrorActionPreference = "Stop"

conda activate wheat_gpu
Set-Location "D:\2026.6.22"

python .\wheat_original\src\build_vlm_prompt_dev_stage10.py `
  --manifest .\wheat_original\data\data_split_811_full\wheat_split_811_manifest.csv `
  --out-dir .\wheat_original\outputs\vlm_comparison\manifests `
  --output-name vlm_prompt_dev_80_stage10.csv `
  --split val `
  --per-stage 10 `
  --seed 20260707

python .\wheat_original\src\stage2_vlm_prompt_screening.py `
  --manifest .\wheat_original\outputs\vlm_comparison\manifests\vlm_prompt_dev_80_stage10.csv `
  --prompt-ids P7,P9,P10 `
  --model qwen3.7-plus `
  --temperature 0 `
  --sleep 0.2
