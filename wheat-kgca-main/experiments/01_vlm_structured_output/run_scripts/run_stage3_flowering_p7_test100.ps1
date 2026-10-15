conda activate wheat_gpu
cd D:\2026.6.22
python .\wheat_original\src\stage3_vlm_model_comparison.py `
  --manifest .\wheat_original\outputs\vlm_comparison\manifests\vlm_model_test_flowering_100.csv `
  --models qwen3.7-plus `
  --prompt-ids P7 `
  --temperature 0 `
  --sleep 0.2
