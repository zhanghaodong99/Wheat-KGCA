# Checkpoints

Large model checkpoints are intentionally not committed to this repository.

Expected local layout:

```text
outputs/
|-- src_model_training/
|   `-- src_single_convnextv2_tiny_20260713_092041/
|       `-- convnextv2_tiny/
|           `-- best_model.pth
`-- paper_full_experiments/
    `-- hierarchical_03_aux_heads_cam_valref_select_main_output/
        `-- best_model.pth
```

For paper submission, provide the final checkpoint through a release asset, Zenodo, Hugging Face, or another permanent download link.

Recommended checkpoint metadata:

| Field | Value |
|---|---|
| Model name | Wheat-KGCA-RCA |
| Backbone | ConvNeXtV2-Tiny |
| Classes | seedling, tillering, stem_elongation, flag_leaf_booting, heading, flowering, grain_filling, maturity |
| VLM prompt | `P11_flag_leaf_boundary_strict.txt` |
| Training split | revised train/val/test manifests |
| Main metric | report Accuracy, Macro-F1, Balanced Accuracy |
| File | `best_model.pth` |
| Download URL | replace before submission |
