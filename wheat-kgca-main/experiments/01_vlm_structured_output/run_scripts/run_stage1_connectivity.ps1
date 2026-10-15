$ErrorActionPreference = "Stop"

Set-Location "D:\2026.6.22"

python .\wheat_original\src\stage1_vlm_connectivity.py `
  --manifest .\wheat_original\outputs\vlm_comparison\manifests\vlm_connectivity_40.csv `
  --prompt-file .\wheat_original\outputs\vlm_comparison\prompts\P4_organ_scan_rule_arbitration.txt `
  --output-root .\wheat_original\outputs\vlm_comparison\connectivity `
  --model qwen3.7-plus `
  --temperature 0 `
  --sleep 0.2
