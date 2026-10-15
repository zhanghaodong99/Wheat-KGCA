from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
    }


def apply_calibration(logits: torch.Tensor, log_temperature: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    temperature = torch.exp(log_temperature).clamp(0.05, 20.0)
    return logits / temperature + bias


def fit_temperature_bias(
    logits_np: np.ndarray,
    labels_np: np.ndarray,
    l2_bias: float,
    use_bias: bool,
    max_iter: int,
) -> tuple[float, np.ndarray, dict[str, float]]:
    logits = torch.from_numpy(logits_np).float()
    labels = torch.from_numpy(labels_np).long()
    log_temperature = torch.zeros((), requires_grad=True)
    bias = torch.zeros(logits.shape[1], requires_grad=use_bias)
    parameters = [log_temperature]
    if use_bias:
        parameters.append(bias)
    optimizer = torch.optim.LBFGS(parameters, lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        calibrated = apply_calibration(logits, log_temperature, bias)
        loss = F.cross_entropy(calibrated, labels)
        if use_bias and l2_bias > 0:
            loss = loss + float(l2_bias) * (bias * bias).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        calibrated = apply_calibration(logits, log_temperature, bias)
        predictions = calibrated.argmax(dim=1).numpy()
        result = metrics(labels_np, predictions)
    return float(torch.exp(log_temperature.detach()).item()), bias.detach().numpy(), result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit val-only temperature/bias calibration and evaluate test logits.")
    parser.add_argument("--logits-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--logit-key", choices=["fused_logits", "visual_logits"], default="fused_logits")
    parser.add_argument("--l2-grid", type=str, default="0,0.001,0.01,0.1,1.0")
    parser.add_argument("--max-iter", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.logits_dir / "export_config.json").read_text(encoding="utf-8"))
    class_names = [str(value) for value in config["class_names"]]
    val = np.load(args.logits_dir / "val_logits.npz", allow_pickle=True)
    test = np.load(args.logits_dir / "test_logits.npz", allow_pickle=True)
    val_logits = val[args.logit_key].astype(np.float32)
    val_labels = val["labels"].astype(np.int64)
    test_logits = test[args.logit_key].astype(np.float32)
    test_labels = test["labels"].astype(np.int64)

    base_val_pred = val_logits.argmax(axis=1)
    base_test_pred = test_logits.argmax(axis=1)
    candidates = [
        {
            "name": "identity",
            "temperature": 1.0,
            "bias": np.zeros(val_logits.shape[1], dtype=np.float32),
            "val": metrics(val_labels, base_val_pred),
        }
    ]
    for use_bias in [False, True]:
        for l2 in [float(value) for value in args.l2_grid.split(",") if value.strip()]:
            if not use_bias and l2 != 0:
                continue
            temperature, bias, val_metrics = fit_temperature_bias(
                val_logits,
                val_labels,
                l2_bias=l2,
                use_bias=use_bias,
                max_iter=args.max_iter,
            )
            candidates.append(
                {
                    "name": f"{'temp_bias' if use_bias else 'temperature'}_l2_{l2:g}",
                    "temperature": temperature,
                    "bias": bias,
                    "val": val_metrics,
                    "l2_bias": l2,
                    "use_bias": use_bias,
                }
            )

    best = max(candidates, key=lambda item: (item["val"]["macro_f1"], item["val"]["accuracy"]))
    bias = np.asarray(best["bias"], dtype=np.float32)
    calibrated_test = test_logits / float(best["temperature"]) + bias.reshape(1, -1)
    calibrated_test_pred = calibrated_test.argmax(axis=1)
    test_metrics = metrics(test_labels, calibrated_test_pred)
    base_test_metrics = metrics(test_labels, base_test_pred)

    meta = pd.read_csv(args.logits_dir / "test_meta.csv")
    meta["base_pred_label"] = [class_names[index] for index in base_test_pred]
    meta["calibrated_pred_label"] = [class_names[index] for index in calibrated_test_pred]
    meta["base_correct"] = meta["stage_id"] == meta["base_pred_label"]
    meta["calibrated_correct"] = meta["stage_id"] == meta["calibrated_pred_label"]
    meta.to_csv(args.out_dir / "test_predictions_calibrated.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        confusion_matrix(meta["stage_id"], meta["calibrated_pred_label"], labels=class_names),
        index=class_names,
        columns=class_names,
    ).to_csv(args.out_dir / "test_calibrated_confusion_matrix.csv", encoding="utf-8-sig")

    serializable_candidates = []
    for item in candidates:
        serializable_candidates.append(
            {
                key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in item.items()
            }
        )
    summary = {
        "logits_dir": str(args.logits_dir),
        "logit_key": args.logit_key,
        "selection_metric": "val macro_f1, then val accuracy",
        "best_candidate": {
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in best.items()
        },
        "base_test": base_test_metrics,
        "calibrated_test": test_metrics,
        "accuracy_change": test_metrics["accuracy"] - base_test_metrics["accuracy"],
        "fix_count": int(((~meta["base_correct"]) & meta["calibrated_correct"]).sum()),
        "break_count": int((meta["base_correct"] & (~meta["calibrated_correct"])).sum()),
        "candidates": serializable_candidates,
    }
    (args.out_dir / "calibration_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
