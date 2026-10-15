# Wheat-KGCA Code Repository

This repository contains the source code and reproducibility assets for Wheat-KGCA, an eight-class wheat growth-stage recognition method for field RGB images.

## Overview

Wheat-KGCA integrates expert-defined agronomic rules, structured VLM evidence, residual coordinate attention (RCA), semantic-visual fusion, knowledge-constrained loss, CAM-based spatial analysis, and final ablation experiments for eight-class wheat growth-stage recognition.

This is a clean code package modeled on the FieldBCQ repository layout. It includes source code, scripts, configs, prompts, fixed manifests, selected experiment instructions, and compact result assets. Raw image data, generated training outputs, and model checkpoints are kept outside the repository. No unrelated DOCX files are included.

## Repository Structure

```text
wheat-kgca-main/
|-- README.md
|-- requirements.txt
|-- environment.yml
|-- configs/
|-- data/
|   |-- README.md
|   `-- manifests/
|-- src/
|   |-- datasets/
|   |-- models/
|   |-- vlm/
|   |-- fusion/
|   |-- training/
|   `-- evaluation/
|-- scripts/
|-- experiments/
|   |-- 01_vlm_structured_output/
|   |-- 02_visual_backbone_comparison/
|   |-- 03_semantic_visual_fusion/
|   |-- 04_cam_alignment/
|   `-- 05_final_ablation/
|-- results/
|-- checkpoints/                     # instructions only; weights are external
`-- docs/
    |-- architecture/
    |   |-- architecture_overview.md
    |   `-- wheat_kgca_overview.svg
    |-- annotation_rules.md
    |-- reproducibility_notes.md
    `-- wheat_kgca_modules.md
```

## Architecture

See [`docs/architecture/architecture_overview.md`](docs/architecture/architecture_overview.md) for the current framework, training/inference separation, and source-code mapping. The editable SVG diagram is available at [`docs/architecture/wheat_kgca_overview.svg`](docs/architecture/wheat_kgca_overview.svg).

## Experiments

| ID | Experiment | Status |
|---|---|---|
| 01 | VLM structured agronomic evidence and VLM-only comparison | Prepared |
| 02 | Visual backbone comparison | Prepared |
| 03 | RCA semantic-visual fusion | Mainline scripts included |
| 04 | CAM alignment and spatial explanation | Mainline scripts included |
| 05 | Final ablation and error diagnosis | Mainline scripts included |

## Installation

```bash
pip install -r requirements.txt
```

For conda users:

```bash
conda env create -f environment.yml
conda activate wheat-kgca
```

## Data Preparation

See `data/README.md`.

The expected dataset layout is:

```text
data/
`-- images/
    |-- train/
    |-- val/
    `-- test/
```

Each split should contain the eight class folders:

```text
seedling
tillering
stem_elongation
flag_leaf_booting
heading
flowering
grain_filling
maturity
```

## Reproducing Part 01

Set your VLM provider variables in the local shell environment. Do not commit private credentials.

### Provider Configuration

The VLM scripts read credentials from environment variables. Replace the placeholder values locally before running the experiment.

PowerShell:

```powershell
$env:OPENAI_API_KEY="<replace_with_your_openai_key>"
$env:OPENAI_BASE_URL="https://api.openai.com/v1"

$env:DASHSCOPE_API_KEY="<replace_with_your_dashscope_key>"

$env:MOONSHOT_API_KEY="<replace_with_your_moonshot_key>"
$env:MOONSHOT_BASE_URL="https://api.moonshot.cn/v1"

$env:ZHIPU_API_KEY="<replace_with_your_zhipu_key>"
$env:ZHIPU_BASE_URL="https://open.bigmodel.cn/api/paas/v4"
```

Linux/macOS:

```bash
export OPENAI_API_KEY="<replace_with_your_openai_key>"
export OPENAI_BASE_URL="https://api.openai.com/v1"

export DASHSCOPE_API_KEY="<replace_with_your_dashscope_key>"

export MOONSHOT_API_KEY="<replace_with_your_moonshot_key>"
export MOONSHOT_BASE_URL="https://api.moonshot.cn/v1"

export ZHIPU_API_KEY="<replace_with_your_zhipu_key>"
export ZHIPU_BASE_URL="https://open.bigmodel.cn/api/paas/v4"
```

Use `--api-key-env` to select which environment variable should be read. Use `--api-base-url` when the provider endpoint differs from the default.

Run a small connectivity test:

```bash
python src/vlm/run_vlm_connectivity_test.py --manifest experiments/01_vlm_structured_output/manifests/vlm_connectivity_40.csv --prompt experiments/01_vlm_structured_output/prompts/P4_organ_scan_rule_arbitration.txt
```

Run prompt screening:

```bash
python src/vlm/stage2_vlm_prompt_screening.py --manifest experiments/01_vlm_structured_output/manifests/vlm_prompt_dev_400.csv --prompt-dir experiments/01_vlm_structured_output/prompts --prompt-ids P6,P7,P8
```

Run VLM model comparison:

```bash
python src/vlm/stage3_vlm_model_comparison.py --manifest experiments/01_vlm_structured_output/manifests/vlm_model_test_balanced_800.csv --prompt-dir experiments/01_vlm_structured_output/prompts --prompt-ids P7 --provider openai --api-key-env OPENAI_API_KEY --api-base-url https://api.openai.com/v1 --models gpt-5.5
```

Run GPT-6 comparison:

```bash
python src/vlm/stage3_vlm_model_comparison.py --manifest experiments/01_vlm_structured_output/manifests/vlm_model_test_balanced_800.csv --prompt-dir experiments/01_vlm_structured_output/prompts --prompt-ids P7 --provider openai --api-key-env OPENAI_API_KEY --api-base-url https://api.openai.com/v1 --models gpt-6 --openai-api-mode responses --openai-image-detail high
```

## Reproducing Part 02

Part 02 compares visual backbones under the same wheat growth-stage split. It provides the visual-only baseline used before semantic fusion and boundary-aware modules.

Run one visual backbone:

```powershell
.\scripts\02a_run_visual_model_single.ps1 -Model convnextv2_tiny -Epochs 30 -EnvName wheat_gpu
```

Run the backbone sequence:

```powershell
.\scripts\02b_run_visual_backbone_sequence.ps1 -Epochs 30 -EnvName wheat_gpu
```

Summarize visual results:

```powershell
.\scripts\02c_summarize_visual_results.ps1 -EnvName wheat_gpu
```

Generated run outputs are written to ignored output directories and are not included in this clean repository copy.

## Notes for Reviewers

1. The VLM structured-output experiment demonstrates that VLMs can produce stable, parseable agronomic evidence.
2. VLM-only classification is not the final model; VLM evidence is used as an auxiliary semantic signal.
3. The visual backbone comparison identifies the strongest visual-only backbone and the remaining adjacent-stage confusion.
4. Final paper results should use the latest boundary-aware prompt and revised label manifests.
5. Large image data and checkpoints should be provided separately if required by the journal.
6. Large checkpoints are not included. See `checkpoints/README.md` for the expected checkpoint layout and metadata.
