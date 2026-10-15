Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location "$PSScriptRoot\..\experiments\01_vlm_structured_output"
python src\run_vlm_connectivity_test.py --manifest manifests\vlm_connectivity_40.csv --prompt prompts\P4_organ_scan_rule_arbitration.txt

