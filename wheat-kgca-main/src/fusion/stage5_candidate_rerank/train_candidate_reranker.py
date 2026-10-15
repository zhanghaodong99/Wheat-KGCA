from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion.train_semantic_visual_module import SemanticVectorizer  # noqa: E402
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "fusion" / "stage5_candidate_rerank" / "candidate_reranker"

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

STAGE_ORDER = [
    "seedling",
    "tillering",
    "stem_elongation",
    "flag_leaf_booting",
    "heading",
    "flowering",
    "grain_filling",
    "maturity",
]
STAGE_POSITION = {name: index for index, name in enumerate(STAGE_ORDER)}


def image_key(path_value: str) -> str:
    return PureWindowsPath(str(path_value)).name.lower()


def load_visual(npz_path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    data = np.load(npz_path, allow_pickle=True)
    class_names = [str(value) for value in data["class_names"].tolist()]
    labels = data["labels"].astype(np.int64)
    logits = data["logits"].astype(np.float32)
    probs = data["probs"].astype(np.float32)
    frame = pd.DataFrame(
        {
            "row_idx": np.arange(len(labels)),
            "image_id": data["image_id"].astype(str),
            "image_key": data["image_key"].astype(str),
            "true_idx": labels,
            "true_label": [class_names[index] for index in labels],
            "conv_idx": probs.argmax(axis=1),
            "conv_label": [class_names[index] for index in probs.argmax(axis=1)],
            "conv_confidence": probs.max(axis=1),
        }
    )
    return frame, logits, probs, labels, class_names


def load_semantic(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[(frame["status"] == "ok") & (frame["schema_valid"])].copy()
    if "image_key" not in frame.columns:
        frame["image_key"] = frame["image_path"].map(image_key)
    if "confidence" in frame.columns:
        frame = frame.rename(columns={"confidence": "gpt_confidence"})
    return frame


def split_features(
    visual_path: Path,
    semantic_path: Path,
    vectorizer: SemanticVectorizer,
    fit_vectorizer: bool,
) -> dict[str, Any]:
    visual_frame, logits, probs, labels, class_names = load_visual(visual_path)
    semantic = load_semantic(semantic_path)
    merged = visual_frame.merge(semantic, on="image_key", how="inner", validate="one_to_one", suffixes=("", "_semantic"))
    if len(merged) != len(visual_frame):
        raise RuntimeError(f"Unmatched visual/semantic rows: visual={len(visual_frame)}, merged={len(merged)}")
    order = merged["row_idx"].to_numpy(dtype=int)
    semantic_matrix = vectorizer.fit_transform(merged) if fit_vectorizer else vectorizer.transform(merged)
    return {
        "frame": merged.reset_index(drop=True),
        "logits": logits[order],
        "probs": probs[order],
        "labels": labels[order],
        "semantic": semantic_matrix,
        "class_names": class_names,
    }


def stage_distance(a: str, b: str) -> int:
    if a not in STAGE_POSITION or b not in STAGE_POSITION:
        return 9
    return abs(STAGE_POSITION[a] - STAGE_POSITION[b])


def candidate_indices(row: pd.Series, probs: np.ndarray, class_to_idx: dict[str, int], top_k: int) -> list[int]:
    order = np.argsort(probs)[::-1][:top_k].tolist()
    for column in ("predicted_stage", "second_candidate"):
        value = str(row.get(column, "")).strip()
        if value in class_to_idx:
            order.append(class_to_idx[value])
    seen = set()
    unique = []
    for index in order:
        if index not in seen:
            seen.add(index)
            unique.append(index)
    return unique


def entropy(probs: np.ndarray) -> float:
    clipped = np.clip(probs, 1e-8, 1.0)
    return float(-(clipped * np.log(clipped)).sum())


def build_candidate_table(split: dict[str, Any], vectorizer_columns: list[str], top_k: int) -> pd.DataFrame:
    frame = split["frame"]
    logits = split["logits"]
    probs = split["probs"]
    labels = split["labels"]
    semantic = split["semantic"]
    class_names = split["class_names"]
    class_to_idx = {name: index for index, name in enumerate(class_names)}

    rows: list[dict[str, Any]] = []
    for row_index, row in frame.iterrows():
        sample_probs = probs[row_index]
        sample_logits = logits[row_index]
        top_order = np.argsort(sample_probs)[::-1]
        top1 = int(top_order[0])
        top2 = int(top_order[1])
        conv_score_reference = float(sample_probs[top1])
        sample_entropy = entropy(sample_probs)
        predicted_stage = str(row.get("predicted_stage", "")).strip()
        second_candidate = str(row.get("second_candidate", "")).strip()
        gpt_confidence = float(row.get("gpt_confidence", 0.0))
        for candidate_idx in candidate_indices(row, sample_probs, class_to_idx, top_k):
            candidate_label = class_names[candidate_idx]
            candidate_rank_array = np.where(top_order == candidate_idx)[0]
            candidate_rank = int(candidate_rank_array[0] + 1) if len(candidate_rank_array) else len(class_names) + 1
            base = {
                "image_id": row["image_id"],
                "true_label": class_names[int(labels[row_index])],
                "candidate_label": candidate_label,
                "conv_label": class_names[top1],
                "visual_top2_label": class_names[top2],
                "vlm_label": predicted_stage,
                "vlm_second": second_candidate,
                "candidate_is_true": int(candidate_idx == int(labels[row_index])),
                "candidate_is_conv": int(candidate_idx == top1),
                "candidate_is_visual_top2": int(candidate_idx == top2),
                "candidate_is_vlm": int(candidate_label == predicted_stage),
                "candidate_is_vlm_second": int(candidate_label == second_candidate),
                "candidate_prob": float(sample_probs[candidate_idx]),
                "candidate_logit": float(sample_logits[candidate_idx]),
                "candidate_rank": candidate_rank,
                "conv_confidence": conv_score_reference,
                "visual_margin": float(sample_probs[top1] - sample_probs[top2]),
                "visual_entropy": sample_entropy,
                "gpt_confidence": gpt_confidence,
                "candidate_stage_pos": STAGE_POSITION.get(candidate_label, -1),
                "conv_stage_pos": STAGE_POSITION.get(class_names[top1], -1),
                "vlm_stage_pos": STAGE_POSITION.get(predicted_stage, -1),
                "candidate_conv_distance": stage_distance(candidate_label, class_names[top1]),
                "candidate_vlm_distance": stage_distance(candidate_label, predicted_stage),
                "conv_vlm_distance": stage_distance(class_names[top1], predicted_stage),
                "candidate_is_adjacent_to_conv": int(stage_distance(candidate_label, class_names[top1]) == 1),
                "candidate_is_adjacent_to_vlm": int(stage_distance(candidate_label, predicted_stage) == 1),
            }
            for semantic_index, semantic_column in enumerate(vectorizer_columns):
                base[f"semantic__{semantic_column}"] = float(semantic[row_index, semantic_index])
            rows.append(base)
    return pd.DataFrame(rows)


def feature_columns(candidate_frame: pd.DataFrame) -> list[str]:
    excluded = {
        "image_id",
        "true_label",
        "candidate_label",
        "conv_label",
        "visual_top2_label",
        "vlm_label",
        "vlm_second",
        "candidate_is_true",
    }
    return [column for column in candidate_frame.columns if column not in excluded]


def grouped_choice(
    candidate_frame: pd.DataFrame,
    score_column: str,
    threshold: float,
) -> pd.DataFrame:
    rows = []
    for image_id, group in candidate_frame.groupby("image_id", sort=False):
        conv_rows = group[group["candidate_is_conv"] == 1]
        if conv_rows.empty:
            chosen = group.sort_values(score_column, ascending=False).iloc[0]
        else:
            conv = conv_rows.iloc[0]
            best = group.sort_values(score_column, ascending=False).iloc[0]
            chosen = best if float(best[score_column] - conv[score_column]) >= threshold else conv
        rows.append(
            {
                "image_id": image_id,
                "true_label": chosen["true_label"],
                "pred_label": chosen["candidate_label"],
                "conv_label": chosen["conv_label"],
                "vlm_label": chosen["vlm_label"],
                "switched": chosen["candidate_label"] != chosen["conv_label"],
                "candidate_score": float(chosen[score_column]),
            }
        )
    return pd.DataFrame(rows)


def metric_dict(predictions: pd.DataFrame, class_names: list[str]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(predictions["true_label"], predictions["pred_label"])),
        "macro_f1": float(f1_score(predictions["true_label"], predictions["pred_label"], average="macro", labels=class_names)),
        "balanced_accuracy": float(balanced_accuracy_score(predictions["true_label"], predictions["pred_label"])),
    }


def movement(predictions: pd.DataFrame) -> dict[str, int]:
    conv_correct = predictions["conv_label"] == predictions["true_label"]
    pred_correct = predictions["pred_label"] == predictions["true_label"]
    switched = predictions["switched"]
    return {
        "switch_count": int(switched.sum()),
        "corrected_count": int((switched & ~conv_correct & pred_correct).sum()),
        "introduced_error_count": int((switched & conv_correct & ~pred_correct).sum()),
        "wrong_to_wrong_switch_count": int((switched & ~conv_correct & ~pred_correct).sum()),
        "net_correct_change": int(pred_correct.sum() - conv_correct.sum()),
    }


def tune_threshold(
    val_candidates: pd.DataFrame,
    score_column: str,
    class_names: list[str],
    max_switch_rate: float,
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    thresholds = sorted(
        set([0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50])
        | set(np.quantile(val_candidates[score_column], np.linspace(0.05, 0.95, 10)).round(4).tolist())
    )
    rows = []
    best_threshold = thresholds[-1]
    best_predictions = None
    best_key = None
    total_images = val_candidates["image_id"].nunique()
    for threshold in thresholds:
        predictions = grouped_choice(val_candidates, score_column, threshold)
        values = metric_dict(predictions, class_names)
        move = movement(predictions)
        switch_rate = move["switch_count"] / total_images
        row = {
            "threshold": threshold,
            "switch_rate": switch_rate,
            **values,
            **move,
        }
        rows.append(row)
        if switch_rate > max_switch_rate:
            continue
        key = (values["macro_f1"], values["accuracy"], -move["introduced_error_count"], -move["switch_count"], threshold)
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = threshold
            best_predictions = predictions
    if best_predictions is None:
        best_predictions = grouped_choice(val_candidates, score_column, thresholds[-1])
    return best_threshold, best_predictions, pd.DataFrame(rows)


def oracle_coverage(candidate_frame: pd.DataFrame) -> float:
    covered = candidate_frame.groupby("image_id")["candidate_is_true"].max()
    return float(covered.mean())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a conservative frozen candidate reranker.")
    parser.add_argument("--visual-train", type=Path, default=DEFAULT_TRAIN_VISUAL)
    parser.add_argument("--visual-val", type=Path, default=DEFAULT_VAL_VISUAL)
    parser.add_argument("--visual-test", type=Path, default=DEFAULT_TEST_VISUAL)
    parser.add_argument("--semantic-train", type=Path, default=DEFAULT_TRAIN_SEMANTIC)
    parser.add_argument("--semantic-val", type=Path, default=DEFAULT_VAL_SEMANTIC)
    parser.add_argument("--semantic-test", type=Path, default=DEFAULT_TEST_SEMANTIC)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-switch-rate", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=20260718)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    vectorizer = SemanticVectorizer()
    train = split_features(args.visual_train, args.semantic_train, vectorizer, True)
    val = split_features(args.visual_val, args.semantic_val, vectorizer, False)
    test = split_features(args.visual_test, args.semantic_test, vectorizer, False)
    class_names = train["class_names"]

    train_candidates = build_candidate_table(train, vectorizer.columns, args.top_k)
    val_candidates = build_candidate_table(val, vectorizer.columns, args.top_k)
    test_candidates = build_candidate_table(test, vectorizer.columns, args.top_k)
    features = feature_columns(train_candidates)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            C=0.35,
            random_state=args.seed,
        ),
    )
    model.fit(train_candidates[features], train_candidates["candidate_is_true"])
    for candidate_frame in (train_candidates, val_candidates, test_candidates):
        candidate_frame["rerank_score"] = model.predict_proba(candidate_frame[features])[:, 1]

    threshold, val_predictions, threshold_curve = tune_threshold(
        val_candidates,
        score_column="rerank_score",
        class_names=class_names,
        max_switch_rate=args.max_switch_rate,
    )
    test_predictions = grouped_choice(test_candidates, "rerank_score", threshold)
    train_predictions = grouped_choice(train_candidates, "rerank_score", threshold)

    train_candidates.to_csv(args.out_dir / "train_candidate_scores.csv", index=False, encoding="utf-8-sig")
    val_candidates.to_csv(args.out_dir / "val_candidate_scores.csv", index=False, encoding="utf-8-sig")
    test_candidates.to_csv(args.out_dir / "test_candidate_scores.csv", index=False, encoding="utf-8-sig")
    train_predictions.to_csv(args.out_dir / "train_predictions.csv", index=False, encoding="utf-8-sig")
    val_predictions.to_csv(args.out_dir / "val_predictions.csv", index=False, encoding="utf-8-sig")
    test_predictions.to_csv(args.out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    threshold_curve.to_csv(args.out_dir / "val_threshold_curve.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        confusion_matrix(test_predictions["true_label"], test_predictions["pred_label"], labels=class_names),
        index=class_names,
        columns=class_names,
    ).to_csv(args.out_dir / "test_confusion_matrix.csv", encoding="utf-8-sig")

    baseline_test = test_predictions.copy()
    baseline_test["pred_label"] = baseline_test["conv_label"]
    summary = {
        "top_k": args.top_k,
        "max_switch_rate": args.max_switch_rate,
        "selected_threshold": float(threshold),
        "feature_dim": len(features),
        "candidate_oracle_coverage": {
            "train": oracle_coverage(train_candidates),
            "val": oracle_coverage(val_candidates),
            "test": oracle_coverage(test_candidates),
        },
        "baseline_visual_test": metric_dict(baseline_test, class_names),
        "reranker_train": {**metric_dict(train_predictions, class_names), **movement(train_predictions)},
        "reranker_val": {**metric_dict(val_predictions, class_names), **movement(val_predictions)},
        "reranker_test": {**metric_dict(test_predictions, class_names), **movement(test_predictions)},
    }
    summary["test_accuracy_change_vs_visual"] = (
        summary["reranker_test"]["accuracy"] - summary["baseline_visual_test"]["accuracy"]
    )
    summary["test_macro_f1_change_vs_visual"] = (
        summary["reranker_test"]["macro_f1"] - summary["baseline_visual_test"]["macro_f1"]
    )
    (args.out_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "feature_columns.json").write_text(json.dumps(features, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
