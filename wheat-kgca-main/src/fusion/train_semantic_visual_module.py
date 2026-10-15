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
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "fusion" / "semantic_visual_module"

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


def load_visual(npz_path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str]]:
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
    visual = np.concatenate([data["features"].astype(np.float32), data["logits"].astype(np.float32)], axis=1)
    probs = data["probs"].astype(np.float32)
    return frame, visual, probs, labels, class_names


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    visual_frame, visual_array, probs, labels, class_names = load_visual(visual_npz)
    semantic = load_semantic(semantic_csv)
    merged = visual_frame.merge(semantic, on="image_key", how="inner", suffixes=("", "_semantic"))
    if merged.empty:
        raise RuntimeError(f"No matched rows for {visual_npz} and {semantic_csv}")
    row_idx = merged["row_idx"].to_numpy(dtype=int)
    semantic_array = vectorizer.fit_transform(merged) if fit_vectorizer else vectorizer.transform(merged)
    meta = merged[["image_id", "image_key", "true_label"]].copy()
    return visual_array[row_idx], semantic_array, probs[row_idx], labels[row_idx], meta, class_names


class FusionDataset(Dataset):
    def __init__(self, visual: np.ndarray, semantic: np.ndarray, probs: np.ndarray, labels: np.ndarray):
        self.visual = torch.from_numpy(visual).float()
        self.semantic = torch.from_numpy(semantic).float()
        self.probs = torch.from_numpy(probs).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.visual[index], self.semantic[index], self.probs[index], self.labels[index]


class SemanticVisualModule(nn.Module):
    def __init__(self, visual_dim: int, semantic_dim: int, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
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
        self.delta = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.visual_aux = nn.Linear(hidden_dim, num_classes)
        nn.init.constant_(self.gate[-1].bias, -2.0)

    def forward(self, visual: torch.Tensor, semantic: torch.Tensor, probs: torch.Tensor):
        visual_token = self.visual_proj(visual)
        semantic_token = self.semantic_encoder(semantic)
        gamma, beta = self.film(semantic_token).chunk(2, dim=1)
        guided_visual = visual_token * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * torch.tanh(beta)

        top2 = torch.topk(probs, k=2, dim=1).values
        max_conf = top2[:, 0:1]
        margin = top2[:, 0:1] - top2[:, 1:2]
        entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
        uncertainty = torch.cat([max_conf, margin, entropy], dim=1)

        gate_input = torch.cat([visual_token, semantic_token, uncertainty], dim=1)
        gate = torch.sigmoid(self.gate(gate_input))
        delta = self.delta(gate_input)
        fused = visual_token + gate * delta * guided_visual
        return self.classifier(fused), self.visual_aux(visual_token), gate


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def train_one_epoch(
    model: SemanticVisualModule,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    aux_weight: float,
    gate_weight: float,
) -> float:
    model.train()
    losses = []
    for visual, semantic, probs, labels in loader:
        visual = visual.to(device)
        semantic = semantic.to(device)
        probs = probs.to(device)
        labels = labels.to(device)
        logits, visual_logits, gate = model(visual, semantic, probs)
        high_conf = (probs.max(dim=1).values > 0.95).float().unsqueeze(1)
        loss = (
            F.cross_entropy(logits, labels)
            + aux_weight * F.cross_entropy(visual_logits, labels)
            + gate_weight * (gate * high_conf).mean()
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses))


@torch.no_grad()
def predict(model: SemanticVisualModule, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    preds = []
    labels_all = []
    gates = []
    for visual, semantic, probs, labels in loader:
        logits, _, gate = model(visual.to(device), semantic.to(device), probs.to(device))
        preds.append(logits.argmax(dim=1).cpu().numpy())
        labels_all.append(labels.numpy())
        gates.append(gate.cpu().numpy())
    return np.concatenate(preds), np.concatenate(labels_all), float(np.concatenate(gates).mean())


def metrics(labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a semantic-guided visual fusion module, not simple concatenation.")
    parser.add_argument("--visual-train", type=Path, required=True)
    parser.add_argument("--visual-val", type=Path, required=True)
    parser.add_argument("--visual-test", type=Path, required=True)
    parser.add_argument("--semantic-train", type=Path, required=True)
    parser.add_argument("--semantic-val", type=Path, required=True)
    parser.add_argument("--semantic-test", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--aux-weight", type=float, default=0.3)
    parser.add_argument("--gate-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    vectorizer = SemanticVectorizer()
    train_v, train_s, train_p, train_y, _, class_names = build_split(args.visual_train, args.semantic_train, vectorizer, True)
    val_v, val_s, val_p, val_y, _, _ = build_split(args.visual_val, args.semantic_val, vectorizer, False)
    test_v, test_s, test_p, test_y, test_meta, _ = build_split(args.visual_test, args.semantic_test, vectorizer, False)

    train_loader = DataLoader(FusionDataset(train_v, train_s, train_p, train_y), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(FusionDataset(val_v, val_s, val_p, val_y), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(FusionDataset(test_v, test_s, test_p, test_y), batch_size=args.batch_size, shuffle=False)

    model = SemanticVisualModule(
        visual_dim=train_v.shape[1],
        semantic_dim=train_s.shape[1],
        num_classes=len(class_names),
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device, args.aux_weight, args.gate_weight)
        val_pred, val_labels, val_gate = predict(model, val_loader, device)
        val_metrics = metrics(val_labels, val_pred)
        row = {"epoch": epoch, "loss": loss, "val_gate_mean": val_gate, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        if val_metrics["macro_f1"] > best_val:
            best_val = val_metrics["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch={epoch:03d} loss={loss:.4f} val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['macro_f1']:.4f} gate={val_gate:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    test_pred, test_labels, test_gate = predict(model, test_loader, device)
    test_metrics = metrics(test_labels, test_pred)

    pred_labels = [class_names[i] for i in test_pred]
    true_labels = [class_names[i] for i in test_labels]
    pred_df = test_meta.copy()
    pred_df["pred_label"] = pred_labels
    pred_df["correct"] = np.array(pred_labels) == np.array(true_labels)
    pred_df.to_csv(args.out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(confusion_matrix(true_labels, pred_labels, labels=class_names), index=class_names, columns=class_names).to_csv(
        args.out_dir / "test_confusion_matrix.csv",
        encoding="utf-8-sig",
    )
    pd.DataFrame(history).to_csv(args.out_dir / "history.csv", index=False, encoding="utf-8-sig")

    summary = {
        "test_gate_mean": test_gate,
        "semantic_dim": int(train_s.shape[1]),
        "visual_dim": int(train_v.shape[1]),
        "class_names": class_names,
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }
    (args.out_dir / "semantic_visual_module_metrics.json").write_text(
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
