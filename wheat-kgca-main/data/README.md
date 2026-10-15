# Data Preparation

The full image dataset is not included in this code package by default.

## Expected Layout

```text
data/
└── images/
    ├── train/
    │   ├── seedling/
    │   ├── tillering/
    │   ├── stem_elongation/
    │   ├── flag_leaf_booting/
    │   ├── heading/
    │   ├── flowering/
    │   ├── grain_filling/
    │   └── maturity/
    ├── val/
    └── test/
```

## Manifests

Manifests should contain at least:

```text
image_id,image_path,split,label,source
```

For portability, `image_path` should be relative to the repository root or to a user-defined `DATA_ROOT`.

Avoid absolute local paths such as:

```text
D:\project\data\...
```

Use relative paths instead:

```text
data/images/train/seedling/example.jpg
```

