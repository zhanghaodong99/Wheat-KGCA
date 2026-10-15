# Reproducibility Notes

## Randomness

Training scripts should expose a `--seed` argument and report the seed in the output directory.

## External VLM APIs

VLM results may vary by model version and API provider. For reproducibility:

1. Record model name and provider.
2. Record prompt file.
3. Record decoding parameters.
4. Save raw responses and parsed JSON.
5. Report JSON validity, schema validity, evidence completeness, and contradiction rate.

## Data Paths

Use relative paths or `DATA_ROOT` instead of machine-specific absolute paths.

