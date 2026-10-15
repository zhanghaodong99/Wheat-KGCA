# Part 01: VLM Structured Output and Comparison Experiments

This folder contains the code and materials for the VLM structured agronomic evidence experiment.

## Purpose

The experiment evaluates whether vision-language models can produce stable structured evidence for wheat growth-stage recognition. The VLM is not used as the final classifier. Its role is to generate image-specific agronomic evidence, including spike status, anther visibility, flag-leaf/booting clues, tillering clues, stem-elongation clues, plant color, image quality, confidence, and uncertainty.

## Folder Structure

| Folder | Content |
|---|---|
| `src` | Python scripts for fixed subset construction, VLM calling, prompt screening, model comparison, and result summarization |
| `prompts` | Prompt versions P1, P2, P4, P6, P7, P8, P9, and P10 |
| `manifests` | Fixed image lists for connectivity, prompt development, and VLM model comparison |
| `run_scripts` | PowerShell scripts used to run the VLM experiments |
| `results/tables` | Final CSV result tables and summary tables |
| `results/figures` | Final performance figures |
| `results/confusion_matrices` | VLM confusion matrix figures |
| `docs` | Experiment notes and paper-draft documentation |

## Main Scripts

| Script | Function |
|---|---|
| `src/run_vlm_connectivity_test.py` | Core VLM runner: reads image manifest, calls VLM API, parses JSON, validates schema, and computes metrics |
| `src/stage1_vlm_connectivity.py` | Stage 1 connectivity test wrapper |
| `src/stage2_vlm_prompt_screening.py` | Stage 2 prompt screening |
| `src/stage3_vlm_model_comparison.py` | Stage 3 VLM model comparison |
| `src/summarize_vlm_prompt_comparison.py` | Prompt result summarization |

## Key Results

The VLM-only comparison showed that direct VLM classification was weaker than task-specific visual models, but structured output quality was stable:

| Setting | Main Finding |
|---|---|
| Prompt screening | Structured prompts achieved stable JSON output and complete evidence fields |
| VLM-only comparison | GPT-5.5 High + P7 reached 61.25% accuracy and 0.5950 Macro-F1 on test800 |
| Output quality | JSON valid rate, schema valid rate, and evidence completeness were all 1.0000 for the main completed runs |
| Interpretation | VLM is more suitable as a structured agronomic evidence generator than as an independent classifier |

The full tables are in `results/tables`.

## Reproduction

Install dependencies:

```powershell
pip install -r requirements.txt
```

Set the provider key and base URL according to your local account. The examples below intentionally use placeholders only.

```powershell
$env:OPENAI_API_KEY="<replace_with_your_openai_key>"
$env:OPENAI_BASE_URL="https://api.openai.com/v1"

$env:DASHSCOPE_API_KEY="<replace_with_your_dashscope_key>"

$env:MOONSHOT_API_KEY="<replace_with_your_moonshot_key>"
$env:MOONSHOT_BASE_URL="https://api.moonshot.cn/v1"

$env:ZHIPU_API_KEY="<replace_with_your_zhipu_key>"
$env:ZHIPU_BASE_URL="https://open.bigmodel.cn/api/paas/v4"
```

The scripts read the selected key through `--api-key-env`. If the provider endpoint is not the default, pass it with `--api-base-url`.

Example runs:

```powershell
python src/run_vlm_connectivity_test.py --manifest manifests/vlm_connectivity_40.csv --prompt prompts/P4_organ_scan_rule_arbitration.txt
python src/stage2_vlm_prompt_screening.py --manifest manifests/vlm_prompt_dev_400.csv --prompt-dir prompts --prompt-ids P6,P7,P8
python src/stage3_vlm_model_comparison.py --manifest manifests/vlm_model_test_balanced_800.csv --prompt-dir prompts --prompt-ids P7 --provider openai --api-key-env OPENAI_API_KEY --api-base-url https://api.openai.com/v1 --models gpt-5.5
```

The original PowerShell commands are preserved in `run_scripts`.

## GPT-6 Comparison

The repository includes a GPT-6 comparison entry. From the repository root, run:

```powershell
.\scripts\01b_run_gpt6_vlm_comparison.ps1
```

or from this experiment folder:

```powershell
.\run_scripts\run_stage3_gpt6_p7_test800.ps1
```

The script uses `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL`. If `OPENAI_BASE_URL` is not set, it defaults to `https://api.openai.com/v1`.

## Important Methodological Note

The files in this part document the earlier VLM structured-output experiments. They are valid for supporting the structured VLM evidence module. However, the flag-leaf booting boundary rule was later revised. For the final paper experiment, VLM semantics should be regenerated using the latest boundary-aware prompt before being used in the final fusion model.
