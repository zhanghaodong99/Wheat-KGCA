from __future__ import annotations

import argparse
import json
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "fusion" / "semantic_visual_module_v2"

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


def image_key(path_value: str) -> str:
    return PureWindowsPath(str(path_value)).name.lower()


class SemanticVectorizer:
    def __init__(self) -> None:
        self.columns: list[str] = []

    def _build(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        source_conf = "gpt_confidence" if "gpt_confidence" in df.columns else "confidence"
        for col in NUMERIC_EVIDENCE_COLS:
            if col == "confidence" and source_conf in df.columns:
                out["gpt_confidence"] = pd.to_numeric(df[source_conf], errors="coerce")
            elif col == "image_quality" and col in df.columns:
                quality_map = {"poor": 0.0, "low": 0.0, "fair": 0.5, "medium": 0.5, "good": 1.0}
                out[col] = df[col].astype(str).str.lower().map(quality_map)
            elif col in df.columns:
                out[col] = pd.to_numeric(df[col], errors="coerce")

        cat_cols = [col for col in CATEGORICAL_EVIDENCE_COLS if col in df.columns]
        if cat_cols:
            out = pd.concat([out, pd.get_dummies(df[cat_cols].fillna("missing"), prefix=cat_cols)], axis=1)
        return out.replace([np.inf, -np.inf], np.nan).fillna(-1.0).astype(float)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        out = self._build(df)
        self.columns = out.columns.tolist()
        return out.to_numpy(dtype=np.float32)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        out = self._build(df)
        out = out.reindex(columns=self.columns, fill_value=0.0)
        return out.to_numpy(dtype=np.float32)


def load_visual(npz_path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    data = np.load(npz_path, allow_pickle=True)
    class_names = [str(x) for x in data["class_names"].tolist()]
    labels = data["labels"].astype(np.int64)
    frame = pd.DataFrame(
        {
            "row_idx": np.arange(len(labels)),
            "image_id": data["image_id"].astype(str),
            "image_key": data["image_key"].astype(str),
            "true_label": [class_names[i] for i in labels],
        }
    )
    features = data["features"].astype(np.float32)
    logits = data["logits"].astype(np.float32)
    probs = data["probs"].astype(np.float32)
    return frame, features, logits, probs, labels, class_names


def load_semantic(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "image_key" not in df.columns:
        df["image_key"] = df["image_path"].map(image_key)
    if "confidence" in df.columns and "gpt_confidence" not in df.columns:
        df = df.rename(columns={"confidence": "gpt_confidence"})
    return df


def build_split(
    visual_npz: Path,
    semantic_csv: Path,
    vectorizer: SemanticVectorizer,
    fit_vectorizer: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    visual_frame, features, logits, probs, labels, class_names = load_visual(visual_npz)
    semantic = load_semantic(semantic_csv)
    merged = visual_frame.merge(semantic, on="image_key", how="inner", suffixes=("", "_semantic"))
    if merged.empty:
        raise RuntimeError(f"No matched rows for {visual_npz} and {semantic_csv}")
    if len(merged) != len(visual_frame):
        raise RuntimeError(f"Matched rows {len(merged)} do not equal visual rows {len(visual_frame)} for {visual_npz}")
    row_idx = merged["row_idx"].to_numpy(dtype=int)
    semantic_array = vectorizer.fit_transform(merged) if fit_vectorizer else vectorizer.transform(merged)
    meta = merged[["image_id", "image_key", "true_label"]].copy()
    return features[row_idx], logits[row_idx], probs[row_idx], semantic_array, labels[row_idx], meta, class_names


class FusionDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        logits: np.ndarray,
        probs: np.ndarray,
        semantic: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self.features = torch.from_numpy(features).float()
        self.logits = torch.from_numpy(logits).float()
        self.probs = torch.from_numpy(probs).float()
        self.semantic = torch.from_numpy(semantic).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.features[index], self.logits[index], self.probs[index], self.semantic[index], self.labels[index]


class ConservativeSemanticVisualModule(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        semantic_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        max_confidence: float = 0.85,
        max_margin: float = 0.35,
        eligibility_sharpness: float = 18.0,
        delta_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.max_confidence = max_confidence
        self.max_margin = max_margin
        self.eligibility_sharpness = eligibility_sharpness
        self.delta_scale = delta_scale
        self.visual_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.semantic_encoder = nn.Sequential(
            nn.Linear(semantic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.film = nn.Linear(hidden_dim, hidden_dim * 2)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, num_classes),
            nn.Tanh(),
        )
        self.visual_aux = nn.Linear(hidden_dim, num_classes)
        nn.init.constant_(self.gate[-1].bias, -3.0)
        nn.init.zeros_(self.delta_head[-2].weight)
        nn.init.zeros_(self.delta_head[-2].bias)

    def forward(self, features: torch.Tensor, logits: torch.Tensor, probs: torch.Tensor, semantic: torch.Tensor):
        visual_token = self.visual_proj(features)
        semantic_token = self.semantic_encoder(semantic)
        gamma, beta = self.film(semantic_token).chunk(2, dim=1)
        guided_visual = visual_token * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * torch.tanh(beta)

        top2 = torch.topk(probs, k=2, dim=1).values
        max_conf = top2[:, 0:1]
        margin = top2[:, 0:1] - top2[:, 1:2]
        entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
        uncertainty = torch.cat([max_conf, margin, entropy], dim=1)

        gate_input = torch.cat([visual_token, semantic_token, uncertainty], dim=1)
        raw_gate = torch.sigmoid(self.gate(gate_input))
        low_conf = torch.sigmoid((self.max_confidence - max_conf) * self.eligibility_sharpness)
        low_margin = torch.sigmoid((self.max_margin - margin) * self.eligibility_sharpness)
        eligibility = torch.maximum(low_conf, low_margin)
        effective_gate = raw_gate * eligibility

        delta_input = torch.cat([guided_visual, semantic_token, uncertainty], dim=1)
        delta_logits = self.delta_scale * self.delta_head(delta_input)
        fused_logits = logits + effective_gate * delta_logits
        return fused_logits, self.visual_aux(visual_token), raw_gate, effective_gate, eligibility, delta_logits


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def weighted_ce(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    losses = F.cross_entropy(logits, labels, reduction="none")
    weights = weights.squeeze(1)
    return (losses * weights).sum() / weights.sum().clamp_min(1e-6)


def high_confidence_mask(probs: torch.Tensor, confidence_threshold: float, margin_threshold: float) -> torch.Tensor:
    top2 = torch.topk(probs, k=2, dim=1).values
    max_conf = top2[:, 0:1]
    margin = top2[:, 0:1] - top2[:, 1:2]
    return ((max_conf >= confidence_threshold) & (margin >= margin_threshold)).float()


def preserve_margin_loss(fused_logits: torch.Tensor, base_logits: torch.Tensor, high_conf: torch.Tensor, margin: float) -> torch.Tensor:
    base_pred = base_logits.argmax(dim=1)
    base_score = fused_logits.gather(1, base_pred[:, None])
    mask = torch.ones_like(fused_logits, dtype=torch.bool)
    mask.scatter_(1, base_pred[:, None], False)
    competitor_score = fused_logits.masked_fill(~mask, -1e9).max(dim=1, keepdim=True).values
    losses = F.relu(margin - (base_score - competitor_score))
    return (losses * high_conf).sum() / high_conf.sum().clamp_min(1.0)


def train_one_epoch(
    model: ConservativeSemanticVisualModule,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    aux_weight: float,
    gate_weight: float,
    anchor_weight: float,
    switch_weight: float,
    delta_weight: float,
    low_conf_focus_weight: float,
    high_confidence_threshold: float,
    high_margin_threshold: float,
    preserve_margin: float,
) -> dict[str, float]:
    model.train()
    totals: dict[str, list[float]] = {
        "loss": [],
        "ce": [],
        "aux": [],
        "gate": [],
        "anchor": [],
        "switch": [],
        "delta": [],
        "raw_gate": [],
        "effective_gate": [],
        "eligibility": [],
    }
    for features, base_logits, probs, semantic, labels in loader:
        features = features.to(device)
        base_logits = base_logits.to(device)
        probs = probs.to(device)
        semantic = semantic.to(device)
        labels = labels.to(device)
        fused_logits, visual_logits, raw_gate, effective_gate, eligibility, delta_logits = model(
            features, base_logits, probs, semantic
        )
        high_conf = high_confidence_mask(probs, high_confidence_threshold, high_margin_threshold)
        sample_weights = 1.0 + low_conf_focus_weight * eligibility.detach()
        ce_loss = weighted_ce(fused_logits, labels, sample_weights)
        aux_loss = F.cross_entropy(visual_logits, labels)
        gate_loss = (raw_gate * high_conf).sum() / high_conf.sum().clamp_min(1.0)
        anchor_loss = (
            F.kl_div(F.log_softmax(fused_logits, dim=1), probs, reduction="none").sum(dim=1, keepdim=True) * high_conf
        ).sum() / high_conf.sum().clamp_min(1.0)
        switch_loss = preserve_margin_loss(fused_logits, base_logits, high_conf, preserve_margin)
        delta_loss = ((delta_logits.pow(2).mean(dim=1, keepdim=True) * effective_gate) * high_conf).sum() / high_conf.sum().clamp_min(1.0)
        loss = (
            ce_loss
            + aux_weight * aux_loss
            + gate_weight * gate_loss
            + anchor_weight * anchor_loss
            + switch_weight * switch_loss
            + delta_weight * delta_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        totals["loss"].append(float(loss.item()))
        totals["ce"].append(float(ce_loss.item()))
        totals["aux"].append(float(aux_loss.item()))
        totals["gate"].append(float(gate_loss.item()))
        totals["anchor"].append(float(anchor_loss.item()))
        totals["switch"].append(float(switch_loss.item()))
        totals["delta"].append(float(delta_loss.item()))
        totals["raw_gate"].append(float(raw_gate.detach().mean().item()))
        totals["effective_gate"].append(float(effective_gate.detach().mean().item()))
        totals["eligibility"].append(float(eligibility.detach().mean().item()))
    return {key: float(np.mean(values)) for key, values in totals.items()}


@torch.no_grad()
def predict(
    model: ConservativeSemanticVisualModule,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    labels_all = []
    raw_gates = []
    effective_gates = []
    eligibilities = []
    fused_confidences = []
    for features, base_logits, probs, semantic, labels in loader:
        fused_logits, _, raw_gate, effective_gate, eligibility, _ = model(
            features.to(device),
            base_logits.to(device),
            probs.to(device),
            semantic.to(device),
        )
        fused_probs = F.softmax(fused_logits, dim=1)
        preds.append(fused_probs.argmax(dim=1).cpu().numpy())
        fused_confidences.append(fused_probs.max(dim=1).values.cpu().numpy())
        labels_all.append(labels.numpy())
        raw_gates.append(raw_gate.cpu().numpy())
        effective_gates.append(effective_gate.cpu().numpy())
        eligibilities.append(eligibility.cpu().numpy())
    return (
        np.concatenate(preds),
        np.concatenate(labels_all),
        np.concatenate(raw_gates).reshape(-1),
        np.concatenate(effective_gates).reshape(-1),
        np.concatenate(eligibilities).reshape(-1),
        np.concatenate(fused_confidences).reshape(-1),
    )


def metrics(labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
    }


def movement(labels: np.ndarray, base_preds: np.ndarray, preds: np.ndarray) -> dict[str, int]:
    switched = preds != base_preds
    base_correct = base_preds == labels
    pred_correct = preds == labels
    return {
        "switch_count": int(switched.sum()),
        "corrected_count": int((switched & ~base_correct & pred_correct).sum()),
        "introduced_error_count": int((switched & base_correct & ~pred_correct).sum()),
        "wrong_to_wrong_switch_count": int((switched & ~base_correct & ~pred_correct).sum()),
        "net_correct_change": int(pred_correct.sum() - base_correct.sum()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train conservative Wheat-SAGE semantic residual module.")
    parser.add_argument("--visual-train", type=Path, required=True)
    parser.add_argument("--visual-val", type=Path, required=True)
    parser.add_argument("--visual-test", type=Path, required=True)
    parser.add_argument("--semantic-train", type=Path, required=True)
    parser.add_argument("--semantic-val", type=Path, required=True)
    parser.add_argument("--semantic-test", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--aux-weight", type=float, default=0.2)
    parser.add_argument("--gate-weight", type=float, default=0.8)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--switch-weight", type=float, default=0.5)
    parser.add_argument("--delta-weight", type=float, default=0.05)
    parser.add_argument("--low-conf-focus-weight", type=float, default=1.0)
    parser.add_argument("--max-confidence", type=float, default=0.85)
    parser.add_argument("--max-margin", type=float, default=0.35)
    parser.add_argument("--eligibility-sharpness", type=float, default=18.0)
    parser.add_argument("--delta-scale", type=float, default=1.0)
    parser.add_argument("--high-confidence-threshold", type=float, default=0.90)
    parser.add_argument("--high-margin-threshold", type=float, default=0.40)
    parser.add_argument("--preserve-margin", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    vectorizer = SemanticVectorizer()
    train_f, train_l, train_p, train_s, train_y, _, class_names = build_split(
        args.visual_train, args.semantic_train, vectorizer, True
    )
    val_f, val_l, val_p, val_s, val_y, _, _ = build_split(args.visual_val, args.semantic_val, vectorizer, False)
    test_f, test_l, test_p, test_s, test_y, test_meta, _ = build_split(args.visual_test, args.semantic_test, vectorizer, False)

    train_loader = DataLoader(
        FusionDataset(train_f, train_l, train_p, train_s, train_y),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(FusionDataset(val_f, val_l, val_p, val_s, val_y), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(FusionDataset(test_f, test_l, test_p, test_s, test_y), batch_size=args.batch_size, shuffle=False)

    model = ConservativeSemanticVisualModule(
        feature_dim=train_f.shape[1],
        semantic_dim=train_s.shape[1],
        num_classes=len(class_names),
        hidden_dim=args.hidden_dim,
        max_confidence=args.max_confidence,
        max_margin=args.max_margin,
        eligibility_sharpness=args.eligibility_sharpness,
        delta_scale=args.delta_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    base_val_pred = val_p.argmax(axis=1)
    base_test_pred = test_p.argmax(axis=1)
    best_state = None
    best_key = None
    history = []
    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.aux_weight,
            args.gate_weight,
            args.anchor_weight,
            args.switch_weight,
            args.delta_weight,
            args.low_conf_focus_weight,
            args.high_confidence_threshold,
            args.high_margin_threshold,
            args.preserve_margin,
        )
        val_pred, val_labels, val_raw_gate, val_effective_gate, val_eligibility, _ = predict(model, val_loader, device)
        val_metrics = metrics(val_labels, val_pred)
        val_move = movement(val_labels, base_val_pred, val_pred)
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_stats.items()},
            "val_raw_gate_mean": float(val_raw_gate.mean()),
            "val_effective_gate_mean": float(val_effective_gate.mean()),
            "val_eligibility_mean": float(val_eligibility.mean()),
            **{f"val_{key}": value for key, value in val_metrics.items()},
            **{f"val_{key}": value for key, value in val_move.items()},
        }
        history.append(row)
        key = (
            val_metrics["macro_f1"],
            val_metrics["accuracy"],
            val_move["net_correct_change"],
            -val_move["introduced_error_count"],
            -val_move["switch_count"],
        )
        if best_key is None or key > best_key:
            best_key = key
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        print(
            f"epoch={epoch:03d} loss={train_stats['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f} raw_gate={val_raw_gate.mean():.4f} "
            f"eff_gate={val_effective_gate.mean():.4f} switch={val_move['switch_count']} "
            f"net={val_move['net_correct_change']}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    test_pred, test_labels, test_raw_gate, test_effective_gate, test_eligibility, test_confidence = predict(
        model, test_loader, device
    )
    test_metrics = metrics(test_labels, test_pred)
    test_move = movement(test_labels, base_test_pred, test_pred)
    base_metrics = metrics(test_y, base_test_pred)

    pred_labels = [class_names[i] for i in test_pred]
    true_labels = [class_names[i] for i in test_labels]
    base_labels = [class_names[i] for i in base_test_pred]
    pred_df = test_meta.copy()
    pred_df["convnext_pred_label"] = base_labels
    pred_df["convnext_confidence"] = test_p.max(axis=1)
    pred_df["pred_label"] = pred_labels
    pred_df["fused_confidence"] = test_confidence
    pred_df["raw_gate"] = test_raw_gate
    pred_df["effective_gate"] = test_effective_gate
    pred_df["eligibility"] = test_eligibility
    pred_df["switched"] = pred_df["pred_label"] != pred_df["convnext_pred_label"]
    pred_df["convnext_correct"] = pred_df["convnext_pred_label"] == pred_df["true_label"]
    pred_df["correct"] = np.array(pred_labels) == np.array(true_labels)
    pred_df.to_csv(args.out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        confusion_matrix(true_labels, pred_labels, labels=class_names),
        index=class_names,
        columns=class_names,
    ).to_csv(args.out_dir / "test_confusion_matrix.csv", encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(args.out_dir / "history.csv", index=False, encoding="utf-8-sig")

    summary = {
        "method": "conservative_semantic_residual_v2",
        "selection": "best validation macro_f1, then accuracy, net_correct_change, fewer introduced errors, fewer switches",
        "semantic_dim": int(train_s.shape[1]),
        "feature_dim": int(train_f.shape[1]),
        "class_names": class_names,
        "loss_weights": {
            "aux_weight": args.aux_weight,
            "gate_weight": args.gate_weight,
            "anchor_weight": args.anchor_weight,
            "switch_weight": args.switch_weight,
            "delta_weight": args.delta_weight,
            "low_conf_focus_weight": args.low_conf_focus_weight,
        },
        "gate_config": {
            "max_confidence": args.max_confidence,
            "max_margin": args.max_margin,
            "eligibility_sharpness": args.eligibility_sharpness,
            "high_confidence_threshold": args.high_confidence_threshold,
            "high_margin_threshold": args.high_margin_threshold,
            "preserve_margin": args.preserve_margin,
            "delta_scale": args.delta_scale,
        },
        "test_raw_gate_mean": float(test_raw_gate.mean()),
        "test_effective_gate_mean": float(test_effective_gate.mean()),
        "test_eligibility_mean": float(test_eligibility.mean()),
        "convnext_test": base_metrics,
        "wheat_sage_v2_test": {**test_metrics, **test_move},
        "test_accuracy_change_vs_convnext": float(test_metrics["accuracy"] - base_metrics["accuracy"]),
        "test_macro_f1_change_vs_convnext": float(test_metrics["macro_f1"] - base_metrics["macro_f1"]),
    }
    (args.out_dir / "semantic_visual_module_v2_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.out_dir / "semantic_vector_columns.json").write_text(
        json.dumps(vectorizer.columns, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Output: {args.out_dir}")


if __name__ == "__main__":
    main()
