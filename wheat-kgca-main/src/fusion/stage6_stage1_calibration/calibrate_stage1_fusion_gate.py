from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion.train_fusion_head import align_columns, build_split  # noqa: E402


DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "fusion" / "stage6_stage1_calibration"
DEFAULT_TRAIN_VISUAL = PROJECT_ROOT / "outputs" / "fusion" / "convnext_features" / "fusion_train_800" / "convnext_features.npz"
DEFAULT_VAL_VISUAL = PROJECT_ROOT / "outputs" / "fusion" / "convnext_features" / "fusion_val_400" / "convnext_features.npz"
DEFAULT_TEST_VISUAL = PROJECT_ROOT / "outputs" / "fusion" / "convnext_features" / "fusion_test_800" / "convnext_features.npz"
DEFAULT_TRAIN_SEMANTIC = (
    PROJECT_ROOT
    / "outputs"
    / "fusion"
    / "vlm_semantics"
    / "stage3_20260715_111841"
    / "gpt-5.5"
    / "P7"
    / "gpt-5.5_P7_fusion_train_800"
    / "parsed_results.csv"
)
DEFAULT_VAL_SEMANTIC = (
    PROJECT_ROOT
    / "outputs"
    / "fusion"
    / "vlm_semantics"
    / "stage3_20260715_150354"
    / "gpt-5.5"
    / "P7"
    / "gpt-5.5_P7_fusion_val_400"
    / "parsed_results.csv"
)
DEFAULT_TEST_SEMANTIC = (
    PROJECT_ROOT
    / "outputs"
    / "fusion"
    / "vlm_semantics"
    / "stage3_20260715_173238"
    / "gpt-5.5"
    / "P7"
    / "gpt-5.5_P7_fusion_test_800"
    / "parsed_results.csv"
)


def load_visual_reference(npz_path: Path) -> pd.DataFrame:
    data = np.load(npz_path, allow_pickle=True)
    class_names = [str(value) for value in data["class_names"].tolist()]
    probs = data["probs"].astype(np.float64)
    logits = data["logits"].astype(np.float64)
    labels = data["labels"].astype(int)
    pred_idx = probs.argmax(axis=1)
    frame = pd.DataFrame(
        {
            "image_id": data["image_id"].astype(str),
            "image_key": data["image_key"].astype(str),
            "true_label": [class_names[index] for index in labels],
            "conv_label": [class_names[index] for index in pred_idx],
            "conv_confidence": probs.max(axis=1),
            "conv_margin": np.sort(probs, axis=1)[:, -1] - np.sort(probs, axis=1)[:, -2],
        }
    )
    for class_index, class_name in enumerate(class_names):
        frame[f"conv_prob_{class_name}"] = probs[:, class_index]
        frame[f"conv_logit_{class_name}"] = logits[:, class_index]
    return frame


def metrics(frame: pd.DataFrame, labels: list[str]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(frame["true_label"], frame["pred_label"])),
        "macro_f1": float(f1_score(frame["true_label"], frame["pred_label"], average="macro", labels=labels)),
        "balanced_accuracy": float(balanced_accuracy_score(frame["true_label"], frame["pred_label"])),
    }


def movement(frame: pd.DataFrame) -> dict[str, int]:
    switched = frame["pred_label"] != frame["conv_label"]
    conv_correct = frame["conv_label"] == frame["true_label"]
    pred_correct = frame["pred_label"] == frame["true_label"]
    return {
        "switch_count": int(switched.sum()),
        "corrected_count": int((switched & ~conv_correct & pred_correct).sum()),
        "introduced_error_count": int((switched & conv_correct & ~pred_correct).sum()),
        "wrong_to_wrong_switch_count": int((switched & ~conv_correct & ~pred_correct).sum()),
        "net_correct_change": int(pred_correct.sum() - conv_correct.sum()),
    }


def predict_fusion(
    clf,
    x: pd.DataFrame,
    meta: pd.DataFrame,
    visual_reference: pd.DataFrame,
) -> pd.DataFrame:
    classes = [str(value) for value in clf.classes_]
    probabilities = clf.predict_proba(x)
    pred_index = probabilities.argmax(axis=1)
    output = meta[["image_id", "image_key", "true_label"]].copy()
    output["fusion_label"] = [classes[index] for index in pred_index]
    output["fusion_confidence"] = probabilities.max(axis=1)
    for class_index, class_name in enumerate(classes):
        output[f"fusion_prob_{class_name}"] = probabilities[:, class_index]
    output = output.merge(
        visual_reference.drop(columns=["true_label"]),
        on=["image_id", "image_key"],
        how="inner",
        validate="one_to_one",
    )
    if len(output) != len(meta):
        raise RuntimeError("Fusion predictions and visual reference did not align one-to-one.")
    output["fusion_prob_conv_label"] = [
        row[f"fusion_prob_{row['conv_label']}"] for _, row in output.iterrows()
    ]
    output["fusion_prob_gain_over_conv_label"] = output["fusion_confidence"] - output["fusion_prob_conv_label"]
    output["conv_correct"] = output["conv_label"] == output["true_label"]
    output["fusion_correct"] = output["fusion_label"] == output["true_label"]
    return output


def apply_gate(predictions: pd.DataFrame, min_gain: float, max_conv_confidence: float) -> pd.DataFrame:
    output = predictions.copy()
    switch = (
        (output["fusion_label"] != output["conv_label"])
        & (output["fusion_prob_gain_over_conv_label"] >= min_gain)
        & (output["conv_confidence"] <= max_conv_confidence)
    )
    output["pred_label"] = np.where(switch, output["fusion_label"], output["conv_label"])
    output["switched"] = switch
    output["correct"] = output["pred_label"] == output["true_label"]
    return output


def tune_gate(val_predictions: pd.DataFrame, labels: list[str], max_switch_rate: float) -> tuple[dict[str, float], pd.DataFrame]:
    gain_values = sorted(
        set([-0.10, -0.05, -0.02, 0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30])
        | set(np.quantile(val_predictions["fusion_prob_gain_over_conv_label"], np.linspace(0.05, 0.95, 10)).round(4))
    )
    confidence_values = [0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.98, 1.01]
    total = len(val_predictions)
    rows = []
    best_key = None
    best_config = {"min_gain": 0.0, "max_conv_confidence": 1.01}
    for min_gain in gain_values:
        for max_confidence in confidence_values:
            gated = apply_gate(val_predictions, float(min_gain), float(max_confidence))
            metric_values = metrics(gated, labels)
            move = movement(gated)
            switch_rate = move["switch_count"] / total
            row = {
                "min_gain": float(min_gain),
                "max_conv_confidence": float(max_confidence),
                "switch_rate": float(switch_rate),
                **metric_values,
                **move,
            }
            rows.append(row)
            if switch_rate > max_switch_rate:
                continue
            key = (
                metric_values["macro_f1"],
                metric_values["accuracy"],
                move["net_correct_change"],
                -move["introduced_error_count"],
                -move["switch_count"],
            )
            if best_key is None or key > best_key:
                best_key = key
                best_config = {"min_gain": float(min_gain), "max_conv_confidence": float(max_confidence)}
    return best_config, pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate a conservative gate around Stage 1 logits+semantic fusion.")
    parser.add_argument("--visual-train", type=Path, default=DEFAULT_TRAIN_VISUAL)
    parser.add_argument("--visual-val", type=Path, default=DEFAULT_VAL_VISUAL)
    parser.add_argument("--visual-test", type=Path, default=DEFAULT_TEST_VISUAL)
    parser.add_argument("--semantic-train", type=Path, default=DEFAULT_TRAIN_SEMANTIC)
    parser.add_argument("--semantic-val", type=Path, default=DEFAULT_VAL_SEMANTIC)
    parser.add_argument("--semantic-test", type=Path, default=DEFAULT_TEST_SEMANTIC)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-switch-rate", type=float, default=0.03)
    parser.add_argument("--c", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_x, train_y, _, class_names = build_split(
        args.visual_train, args.semantic_train, visual_mode="logits", include_semantic=True, include_gpt_stage=False
    )
    val_x, val_y, val_meta, _ = build_split(
        args.visual_val, args.semantic_val, visual_mode="logits", include_semantic=True, include_gpt_stage=False
    )
    test_x, test_y, test_meta, _ = build_split(
        args.visual_test, args.semantic_test, visual_mode="logits", include_semantic=True, include_gpt_stage=False
    )
    train_x, val_x, test_x = align_columns(train_x, val_x, test_x)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=4000, C=args.c, solver="lbfgs"),
    )
    clf.fit(train_x, train_y)

    val_predictions = predict_fusion(clf, val_x, val_meta, load_visual_reference(args.visual_val))
    test_predictions = predict_fusion(clf, test_x, test_meta, load_visual_reference(args.visual_test))
    best_config, val_curve = tune_gate(val_predictions, class_names, args.max_switch_rate)
    val_gated = apply_gate(val_predictions, **best_config)
    test_gated = apply_gate(test_predictions, **best_config)
    test_fusion_raw = test_predictions.copy()
    test_fusion_raw["pred_label"] = test_fusion_raw["fusion_label"]
    test_fusion_raw["correct"] = test_fusion_raw["pred_label"] == test_fusion_raw["true_label"]
    test_visual = test_predictions.copy()
    test_visual["pred_label"] = test_visual["conv_label"]
    test_visual["correct"] = test_visual["pred_label"] == test_visual["true_label"]

    val_predictions.to_csv(args.out_dir / "val_raw_fusion_predictions.csv", index=False, encoding="utf-8-sig")
    test_predictions.to_csv(args.out_dir / "test_raw_fusion_predictions.csv", index=False, encoding="utf-8-sig")
    val_gated.to_csv(args.out_dir / "val_gated_predictions.csv", index=False, encoding="utf-8-sig")
    test_gated.to_csv(args.out_dir / "test_gated_predictions.csv", index=False, encoding="utf-8-sig")
    val_curve.to_csv(args.out_dir / "val_gate_curve.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        confusion_matrix(test_gated["true_label"], test_gated["pred_label"], labels=class_names),
        index=class_names,
        columns=class_names,
    ).to_csv(args.out_dir / "test_gated_confusion_matrix.csv", encoding="utf-8-sig")

    summary = {
        "max_switch_rate": args.max_switch_rate,
        "c": args.c,
        "feature_dim": int(train_x.shape[1]),
        "best_gate": best_config,
        "visual_test": {**metrics(test_visual, class_names), **movement(test_visual)},
        "raw_stage1_fusion_test": {**metrics(test_fusion_raw, class_names), **movement(test_fusion_raw)},
        "gated_stage1_fusion_val": {**metrics(val_gated, class_names), **movement(val_gated)},
        "gated_stage1_fusion_test": {**metrics(test_gated, class_names), **movement(test_gated)},
    }
    summary["raw_fusion_test_change_vs_visual"] = (
        summary["raw_stage1_fusion_test"]["accuracy"] - summary["visual_test"]["accuracy"]
    )
    summary["gated_test_change_vs_visual"] = (
        summary["gated_stage1_fusion_test"]["accuracy"] - summary["visual_test"]["accuracy"]
    )
    (args.out_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
