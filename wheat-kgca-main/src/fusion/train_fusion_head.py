from __future__ import annotations

import argparse
import json
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "fusion" / "fusion_head"

NUMERIC_EVIDENCE_COLS = [
    "confidence",
    "image_quality",
    "spike_visibility",
    "anther_visibility",
    "grain_fullness_score",
    "flag_leaf_booting_strength",
    "tillering_strength",
    "stem_elongation_strength",
    "yellowing_degree",
    "noise_degree",
    "contradiction_count",
]

CATEGORICAL_EVIDENCE_COLS = [
    "spike_status",
    "anther_visible",
    "grain_fullness",
    "flag_leaf_booting_clue",
    "tillering_clue",
    "stem_elongation_clue",
    "plant_color",
    "field_noise",
]

GPT_STAGE_COLS = ["predicted_stage", "second_candidate"]


def image_key(path_value: str) -> str:
    return PureWindowsPath(str(path_value)).name.lower()


def load_visual(npz_path: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray], list[str]]:
    data = np.load(npz_path, allow_pickle=True)
    class_names = [str(x) for x in data["class_names"].tolist()]
    labels = data["labels"].astype(int)
    frame = pd.DataFrame(
        {
            "row_idx": np.arange(len(labels)),
            "image_id": data["image_id"].astype(str),
            "image_key": data["image_key"].astype(str),
            "true_label": [class_names[i] for i in labels],
        }
    )
    arrays = {
        "features": data["features"],
        "logits": data["logits"],
        "probs": data["probs"],
    }
    return frame, arrays, class_names


def load_semantic(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "image_key" not in df.columns:
        df["image_key"] = df["image_path"].map(image_key)
    if "confidence" in df.columns and "gpt_confidence" not in df.columns:
        df = df.rename(columns={"confidence": "gpt_confidence"})
    return df


def semantic_features(df: pd.DataFrame, include_gpt_stage: bool) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    for col in NUMERIC_EVIDENCE_COLS:
        source_col = "gpt_confidence" if col == "confidence" else col
        if source_col in df.columns:
            out_col = "gpt_confidence" if col == "confidence" else col
            features[out_col] = pd.to_numeric(df[source_col], errors="coerce")

    cat_cols = [col for col in CATEGORICAL_EVIDENCE_COLS if col in df.columns]
    if include_gpt_stage:
        cat_cols += [col for col in GPT_STAGE_COLS if col in df.columns]
    if cat_cols:
        features = pd.concat([features, pd.get_dummies(df[cat_cols].fillna("missing"), prefix=cat_cols)], axis=1)

    return features.replace([np.inf, -np.inf], np.nan).fillna(-1.0).astype(float)


def build_split(
    visual_npz: Path,
    semantic_csv: Path,
    visual_mode: str,
    include_semantic: bool,
    include_gpt_stage: bool,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, list[str]]:
    visual_frame, arrays, class_names = load_visual(visual_npz)
    semantic = load_semantic(semantic_csv)
    merged = visual_frame.merge(semantic, on="image_key", how="inner", suffixes=("", "_semantic"))
    if merged.empty:
        raise RuntimeError(f"No matched rows for visual={visual_npz} and semantic={semantic_csv}")

    row_idx = merged["row_idx"].to_numpy(dtype=int)
    parts: list[pd.DataFrame] = []

    if visual_mode in {"logits", "features", "features_logits"}:
        if visual_mode in {"logits", "features_logits"}:
            logits = pd.DataFrame(arrays["logits"][row_idx], columns=[f"logit_{name}" for name in class_names], index=merged.index)
            parts.append(logits)
        if visual_mode in {"features", "features_logits"}:
            feats = pd.DataFrame(arrays["features"][row_idx], columns=[f"feat_{i:04d}" for i in range(arrays["features"].shape[1])], index=merged.index)
            parts.append(feats)

    if include_semantic:
        parts.append(semantic_features(merged, include_gpt_stage=include_gpt_stage))

    if not parts:
        raise ValueError("No feature parts selected.")

    x = pd.concat(parts, axis=1)
    y = merged["true_label"].astype(str).to_numpy()
    meta = merged[["image_id", "image_key", "true_label"]].copy()
    return x, y, meta, class_names


def align_columns(train_x: pd.DataFrame, *others: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    train_cols = train_x.columns
    aligned = [train_x]
    for frame in others:
        aligned.append(frame.reindex(columns=train_cols, fill_value=0.0))
    return tuple(aligned)


def evaluate_method(
    name: str,
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    val_x: pd.DataFrame,
    val_y: np.ndarray,
    test_x: pd.DataFrame,
    test_y: np.ndarray,
    test_meta: pd.DataFrame,
    out_dir: Path,
) -> dict[str, float | str]:
    train_x, val_x, test_x = align_columns(train_x, val_x, test_x)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=4000, C=0.5, solver="lbfgs"),
    )
    clf.fit(train_x, train_y)
    val_pred = clf.predict(val_x)
    test_pred = clf.predict(test_x)

    method_dir = out_dir / name
    method_dir.mkdir(parents=True, exist_ok=True)
    pred_df = test_meta.copy()
    pred_df["pred_label"] = test_pred
    pred_df["correct"] = pred_df["true_label"].astype(str).to_numpy() == test_pred.astype(str)
    pred_df.to_csv(method_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    labels = sorted(np.unique(np.concatenate([train_y, val_y, test_y])))
    cm = confusion_matrix(test_y, test_pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(method_dir / "test_confusion_matrix.csv", encoding="utf-8-sig")

    return {
        "method": name,
        "train_rows": int(len(train_y)),
        "val_rows": int(len(val_y)),
        "test_rows": int(len(test_y)),
        "feature_dim": int(train_x.shape[1]),
        "val_accuracy": float(accuracy_score(val_y, val_pred)),
        "val_macro_f1": float(f1_score(val_y, val_pred, average="macro")),
        "val_balanced_accuracy": float(balanced_accuracy_score(val_y, val_pred)),
        "test_accuracy": float(accuracy_score(test_y, test_pred)),
        "test_macro_f1": float(f1_score(test_y, test_pred, average="macro")),
        "test_balanced_accuracy": float(balanced_accuracy_score(test_y, test_pred)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ConvNeXt-GPT fusion heads from visual npz and VLM parsed CSVs.")
    parser.add_argument("--visual-train", type=Path, required=True)
    parser.add_argument("--visual-val", type=Path, required=True)
    parser.add_argument("--visual-test", type=Path, required=True)
    parser.add_argument("--semantic-train", type=Path, required=True)
    parser.add_argument("--semantic-val", type=Path, required=True)
    parser.add_argument("--semantic-test", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--include-gpt-stage", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    method_specs = [
        ("visual_logits_only", "logits", False),
        ("visual_features_logits", "features_logits", False),
        ("semantic_only", "none", True),
        ("fusion_logits_semantic", "logits", True),
        ("fusion_features_logits_semantic", "features_logits", True),
    ]

    rows = []
    for name, visual_mode, include_semantic in method_specs:
        train_x, train_y, _, _ = build_split(
            args.visual_train,
            args.semantic_train,
            visual_mode=visual_mode,
            include_semantic=include_semantic,
            include_gpt_stage=args.include_gpt_stage,
        )
        val_x, val_y, _, _ = build_split(
            args.visual_val,
            args.semantic_val,
            visual_mode=visual_mode,
            include_semantic=include_semantic,
            include_gpt_stage=args.include_gpt_stage,
        )
        test_x, test_y, test_meta, _ = build_split(
            args.visual_test,
            args.semantic_test,
            visual_mode=visual_mode,
            include_semantic=include_semantic,
            include_gpt_stage=args.include_gpt_stage,
        )
        rows.append(evaluate_method(name, train_x, train_y, val_x, val_y, test_x, test_y, test_meta, args.out_dir))

    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.out_dir / "fusion_head_metrics.csv", index=False, encoding="utf-8-sig")
    (args.out_dir / "fusion_head_config.json").write_text(
        json.dumps({k: str(v) for k, v in vars(args).items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(metrics.to_string(index=False))
    print(f"Output: {args.out_dir}")


if __name__ == "__main__":
    main()
