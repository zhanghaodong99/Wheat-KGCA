from __future__ import annotations

import argparse
import csv
import gc
import importlib
import json
import os
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import PIL
import sklearn
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_SPLIT_ROOT = PROJECT_ROOT / "data" / "data_split_811_full"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "src_model_training"


MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "convnextv2_tiny": {
        "module": "src.models.convnextv2_tiny",
        "paper_name": "ConvNeXt V2 Tiny",
        "input_size": 224,
        "default_batch_size": 32,
    },
    "convnextv2_tiny_eca": {
        "module": "src.models.convnextv2_tiny_eca",
        "paper_name": "ConvNeXt V2 Tiny + ECA",
        "input_size": 224,
        "default_batch_size": 32,
    },
    "convnextv2_tiny_se": {
        "module": "src.models.convnextv2_tiny_se",
        "paper_name": "ConvNeXt V2 Tiny + SE",
        "input_size": 224,
        "default_batch_size": 32,
    },
    "convnextv2_tiny_spatial": {
        "module": "src.models.convnextv2_tiny_spatial",
        "paper_name": "ConvNeXt V2 Tiny + Spatial Gate",
        "input_size": 224,
        "default_batch_size": 32,
    },
    "convnextv2_tiny_coordinate": {
        "module": "src.models.convnextv2_tiny_coordinate",
        "paper_name": "ConvNeXt V2 Tiny + Coordinate Attention",
        "input_size": 224,
        "default_batch_size": 32,
    },
    "convnextv2_tiny_mlp_head": {
        "module": "src.models.convnextv2_tiny_mlp_head",
        "paper_name": "ConvNeXt V2 Tiny + MLP Head",
        "input_size": 224,
        "default_batch_size": 32,
    },
    "convnextv2_tiny_res_eca": {
        "module": "src.models.convnextv2_tiny_res_eca",
        "paper_name": "ConvNeXt V2 Tiny + Residual ECA",
        "input_size": 224,
        "default_batch_size": 32,
    },
    "convnextv2_tiny_res_eca2": {
        "module": "src.models.convnextv2_tiny_res_eca2",
        "paper_name": "ConvNeXt V2 Tiny + Stacked Residual ECA",
        "input_size": 224,
        "default_batch_size": 32,
    },
    "convnextv2_tiny_res_coordinate": {
        "module": "src.models.convnextv2_tiny_res_coordinate",
        "paper_name": "ConvNeXt V2 Tiny + Residual Coordinate Attention",
        "input_size": 224,
        "default_batch_size": 32,
    },
    "convnextv2_tiny_res_cbam": {
        "module": "src.models.convnextv2_tiny_res_cbam",
        "paper_name": "ConvNeXt V2 Tiny + Residual CBAM-lite",
        "input_size": 224,
        "default_batch_size": 32,
    },
    "convnextv2_tiny_res_dwconv": {
        "module": "src.models.convnextv2_tiny_res_dwconv",
        "paper_name": "ConvNeXt V2 Tiny + Residual Depthwise Refinement",
        "input_size": 224,
        "default_batch_size": 32,
    },
    "inception_next_tiny": {
        "module": "src.models.inception_next_tiny",
        "paper_name": "InceptionNeXt Tiny",
        "input_size": 224,
        "default_batch_size": 32,
    },
    "maxvit_tiny_rw_224": {
        "module": "src.models.maxvit_tiny_rw_224",
        "paper_name": "MaxViT Tiny RW 224",
        "input_size": 224,
        "default_batch_size": 24,
    },
    "mobilenetv4_hybrid_m": {
        "module": "src.models.mobilenetv4_hybrid_m",
        "paper_name": "MobileNetV4 Hybrid Medium",
        "input_size": 224,
        "default_batch_size": 64,
    },
    "repvit_tiny": {
        "module": "src.models.repvit_tiny",
        "paper_name": "RepViT Tiny",
        "input_size": 224,
        "default_batch_size": 64,
    },
    "resnet50": {
        "module": "src.models.resnet50",
        "paper_name": "ResNet50",
        "input_size": 224,
        "default_batch_size": 64,
    },
    "swinv2_tiny_window16_256": {
        "module": "src.models.swinv2_tiny_window16_256",
        "paper_name": "SwinV2 Tiny Window16 256",
        "input_size": 256,
        "default_batch_size": 32,
    },
}


class ImageFolderWithMetadata(datasets.ImageFolder):
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str, int]:
        image, target = super().__getitem__(index)
        image_path = self.samples[index][0]
        return image, target, image_path, index


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = torch.cuda.is_available()


def parse_csv_items(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_models(text: str) -> list[str]:
    models = parse_csv_items(text)
    unknown = [model for model in models if model not in MODEL_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown models: {unknown}. Available: {sorted(MODEL_REGISTRY)}"
        )
    if not models:
        raise ValueError("At least one model must be selected.")
    if len(set(models)) != len(models):
        raise ValueError(f"Duplicate models are not allowed: {models}")
    return models


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_matrix_csv(path: Path, matrix: np.ndarray, labels: list[str]) -> None:
    rows: list[dict[str, Any]] = []
    for row_index, label in enumerate(labels):
        row: dict[str, Any] = {"true_label": label}
        for column_index, pred_label in enumerate(labels):
            row[pred_label] = int(matrix[row_index, column_index])
        rows.append(row)
    write_csv(path, rows)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def build_transforms(input_size: int) -> tuple[Any, Any]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def limit_samples_per_class(
    dataset: ImageFolderWithMetadata,
    max_per_class: int,
    seed: int,
) -> None:
    if max_per_class <= 0:
        return
    by_class: dict[int, list[int]] = {}
    for index, (_, target) in enumerate(dataset.samples):
        by_class.setdefault(int(target), []).append(index)
    rng = random.Random(seed)
    selected: list[int] = []
    for target in sorted(by_class):
        indices = by_class[target]
        rng.shuffle(indices)
        selected.extend(indices[:max_per_class])
    selected.sort()
    dataset.samples = [dataset.samples[index] for index in selected]
    dataset.imgs = dataset.samples
    dataset.targets = [target for _, target in dataset.samples]


def build_datasets(
    split_root: Path,
    input_size: int,
    seed: int,
    max_train_per_class: int,
    max_val_per_class: int,
    max_test_per_class: int,
) -> tuple[ImageFolderWithMetadata, ImageFolderWithMetadata, ImageFolderWithMetadata]:
    train_transform, eval_transform = build_transforms(input_size)
    train_dataset = ImageFolderWithMetadata(split_root / "train", transform=train_transform)
    val_dataset = ImageFolderWithMetadata(split_root / "val", transform=eval_transform)
    test_dataset = ImageFolderWithMetadata(split_root / "test", transform=eval_transform)
    if train_dataset.classes != val_dataset.classes or train_dataset.classes != test_dataset.classes:
        raise ValueError(
            "Train/val/test class names differ: "
            f"{train_dataset.classes} / {val_dataset.classes} / {test_dataset.classes}"
        )
    limit_samples_per_class(train_dataset, max_train_per_class, seed + 1)
    limit_samples_per_class(val_dataset, max_val_per_class, seed + 2)
    limit_samples_per_class(test_dataset, max_test_per_class, seed + 3)
    return train_dataset, val_dataset, test_dataset


def build_loaders(
    datasets_tuple: tuple[ImageFolderWithMetadata, ImageFolderWithMetadata, ImageFolderWithMetadata],
    batch_size: int,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset, val_dataset, test_dataset = datasets_tuple
    generator = torch.Generator()
    generator.manual_seed(seed)
    shared: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        shared["prefetch_factor"] = 2
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **shared)
    val_loader = DataLoader(val_dataset, shuffle=False, **shared)
    test_loader = DataLoader(test_dataset, shuffle=False, **shared)
    return train_loader, val_loader, test_loader


def build_local_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    module = importlib.import_module(MODEL_REGISTRY[model_name]["module"])
    if not hasattr(module, "build_model"):
        raise AttributeError(f"{MODEL_REGISTRY[model_name]['module']} has no build_model().")
    return module.build_model(num_classes=num_classes, pretrained=pretrained)


def forward_logits(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    output = model(images)
    if isinstance(output, (tuple, list)):
        output = output[0]
    if hasattr(output, "logits"):
        output = output.logits
    return output


def apply_deterministic_field_degradation(
    images: torch.Tensor,
    sample_indices: torch.Tensor,
    noise_seed: int,
) -> tuple[torch.Tensor, list[str]]:
    corrupted = images.clone()
    modes: list[str] = []
    height, width = corrupted.shape[-2:]
    for batch_index, sample_index_value in enumerate(sample_indices.tolist()):
        sample_index = int(sample_index_value)
        generator = torch.Generator(device=images.device)
        generator.manual_seed(noise_seed + sample_index * 1009)
        mode_index = int(torch.randint(0, 4, (1,), generator=generator, device=images.device).item())
        if mode_index == 0:
            noise = torch.randn(
                corrupted[batch_index].shape,
                generator=generator,
                device=images.device,
                dtype=images.dtype,
            )
            corrupted[batch_index] += noise * 0.35
            modes.append("gaussian_noise")
        elif mode_index == 1:
            factor = 0.5 + float(torch.rand((1,), generator=generator, device=images.device).item())
            corrupted[batch_index] *= factor
            modes.append("brightness")
        elif mode_index == 2:
            corrupted[batch_index] = F.avg_pool2d(
                corrupted[batch_index].unsqueeze(0),
                kernel_size=5,
                stride=1,
                padding=2,
            ).squeeze(0)
            modes.append("average_blur")
        else:
            cut_height = max(1, round(height * 0.18))
            cut_width = max(1, round(width * 0.18))
            max_top = max(1, height - cut_height + 1)
            max_left = max(1, width - cut_width + 1)
            top = int(torch.randint(0, max_top, (1,), generator=generator, device=images.device).item())
            left = int(torch.randint(0, max_left, (1,), generator=generator, device=images.device).item())
            corrupted[batch_index, :, top : top + cut_height, left : left + cut_width] = 0.0
            modes.append("cutout")
    return torch.clamp(corrupted, -2.5, 2.5), modes


def classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
) -> dict[str, Any]:
    labels = list(range(len(class_names)))
    if not y_true:
        return {
            "accuracy": 0.0,
            "accuracy_percent": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "balanced_accuracy": 0.0,
            "samples": 0,
        }
    acc = accuracy_score(y_true, y_pred)
    return {
        "accuracy": round(float(acc), 6),
        "accuracy_percent": round(float(acc) * 100, 2),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 6),
        "weighted_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 6),
        "samples": len(y_true),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    amp_enabled: bool,
    description: str,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_count = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    progress = tqdm(loader, desc=description, dynamic_ncols=True, leave=False)
    for images, labels, _, _ in progress:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = forward_logits(model, images)
            loss = criterion(logits, labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(predictions.detach().cpu().tolist())
        progress.set_postfix(
            loss=f"{total_loss / max(total_count, 1):.4f}",
            acc=f"{accuracy_score(y_true, y_pred):.4f}",
        )
    metrics = classification_metrics(y_true, y_pred, list(range(max(y_true + y_pred) + 1)))
    return {
        "loss": round(total_loss / max(total_count, 1), 6),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    class_names: list[str],
    device: torch.device,
    amp_enabled: bool,
    degraded: bool,
    noise_seed: int,
    description: str,
    collect_rows: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    with torch.no_grad():
        progress = tqdm(loader, desc=description, dynamic_ncols=True, leave=False)
        for images, labels, image_paths, sample_indices in progress:
            images = images.to(device, non_blocking=device.type == "cuda")
            labels_device = labels.to(device, non_blocking=device.type == "cuda")
            degradation_modes = ["clean"] * len(labels)
            if degraded:
                images, degradation_modes = apply_deterministic_field_degradation(
                    images, sample_indices.to(device), noise_seed
                )
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = forward_logits(model, images)
                loss = criterion(logits, labels_device)
            probs = torch.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)
            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
            true_batch = labels.detach().cpu().tolist()
            pred_batch = predictions.detach().cpu().tolist()
            y_true.extend(true_batch)
            y_pred.extend(pred_batch)
            if collect_rows:
                conf_batch = probs.max(dim=1).values.detach().cpu().tolist()
                for row_index, (true_idx, pred_idx) in enumerate(zip(true_batch, pred_batch)):
                    rows.append(
                        {
                            "image_path": str(image_paths[row_index]),
                            "true_idx": int(true_idx),
                            "true_label": class_names[int(true_idx)],
                            "pred_idx": int(pred_idx),
                            "pred_label": class_names[int(pred_idx)],
                            "correct": int(true_idx == pred_idx),
                            "confidence": round(float(conf_batch[row_index]), 6),
                            "degradation": degradation_modes[row_index],
                        }
                    )
    metrics = classification_metrics(y_true, y_pred, class_names)
    metrics["loss"] = round(total_loss / max(total_count, 1), 6)
    elapsed = time.perf_counter() - start
    metrics["fps"] = round(total_count / elapsed, 2) if elapsed > 0 else 0.0
    labels_range = list(range(len(class_names)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels_range)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels_range,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return metrics, rows, matrix, report


def build_optimizer(args: argparse.Namespace, model: nn.Module) -> optim.Optimizer:
    if args.optimizer == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def build_scheduler(args: argparse.Namespace, optimizer: optim.Optimizer) -> Any:
    if args.scheduler == "none":
        return None
    if args.scheduler == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=args.step_gamma,
        )
    return optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )


def selection_value(metrics: dict[str, Any], selection_metric: str) -> float:
    if selection_metric == "val_accuracy":
        return float(metrics["accuracy"])
    return float(metrics["macro_f1"])


def environment_info(device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "pillow": PIL.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
    }


def model_summary_row(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "ok",
        "model": report["model"],
        "paper_name": report["paper_name"],
        "model_dir": report["model_dir"],
        "pretrained": report["pretrained"],
        "params_M": report["params_M"],
        "epochs_completed": report["epochs_completed"],
        "best_epoch": report["best_epoch"],
        "best_val_metric": report["best_selection_value"],
        "clean_accuracy": report["clean"]["accuracy"],
        "clean_accuracy_percent": report["clean"]["accuracy_percent"],
        "clean_macro_f1": report["clean"]["macro_f1"],
        "clean_weighted_f1": report["clean"]["weighted_f1"],
        "clean_balanced_accuracy": report["clean"]["balanced_accuracy"],
        "noise_accuracy": report["noise"].get("accuracy", ""),
        "noise_accuracy_percent": report["noise"].get("accuracy_percent", ""),
        "noise_macro_f1": report["noise"].get("macro_f1", ""),
        "noise_weighted_f1": report["noise"].get("weighted_f1", ""),
        "noise_balanced_accuracy": report["noise"].get("balanced_accuracy", ""),
        "accuracy_drop_percent": report["accuracy_drop_percent"],
        "train_time_s": report["train_time_s"],
        "peak_gpu_memory_mb": report["peak_gpu_memory_mb"],
    }


def write_summary_markdown(run_root: Path, run_config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    lines.append("# src/models 本地视觉模型训练汇总")
    lines.append("")
    lines.append(f"生成时间：{datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## 运行配置")
    lines.append("")
    lines.append(f"- 数据集：`{run_config['split_root']}`")
    lines.append(f"- 模型：`{', '.join(run_config['models'])}`")
    lines.append(f"- Epochs：{run_config['epochs']}")
    lines.append(f"- Batch size：{run_config['batch_size']}")
    lines.append(f"- Device：{run_config['device']}")
    lines.append(f"- 预训练：{run_config['pretrained']}")
    lines.append("")
    lines.append("## 结果表")
    lines.append("")
    lines.append("| Model | Status | Clean Acc | Clean Macro-F1 | Noise Acc | Noise Macro-F1 | ΔAcc | Best Epoch | Params(M) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        if row.get("status") != "ok":
            lines.append(f"| {row.get('model', '')} | error |  |  |  |  |  |  |  |")
            continue
        lines.append(
            "| {model} | ok | {clean_accuracy_percent} | {clean_macro_f1} | "
            "{noise_accuracy_percent} | {noise_macro_f1} | {accuracy_drop_percent} | "
            "{best_epoch} | {params_M} |".format(**row)
        )
    lines.append("")
    lines.append("## 输出文件")
    lines.append("")
    lines.append("- `local_model_comparison.csv`：模型横向指标")
    lines.append("- `<model>/metrics_full.json`：单模型完整指标")
    lines.append("- `<model>/history.csv`：训练历史")
    lines.append("- `<model>/confusion_matrix_clean.csv`：clean 混淆矩阵")
    lines.append("- `<model>/confusion_matrix_noise.csv`：noise 混淆矩阵")
    (run_root / "local_model_comparison_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def train_model(
    model_name: str,
    args: argparse.Namespace,
    run_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    spec = MODEL_REGISTRY[model_name]
    model_dir = run_root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    input_size = int(args.input_size or spec["input_size"])
    batch_size = int(args.batch_size or spec["default_batch_size"])
    seed_everything(args.seed)
    datasets_tuple = build_datasets(
        split_root=args.split_root,
        input_size=input_size,
        seed=args.seed,
        max_train_per_class=args.max_train_per_class,
        max_val_per_class=args.max_val_per_class,
        max_test_per_class=args.max_test_per_class,
    )
    train_dataset, val_dataset, test_dataset = datasets_tuple
    class_names = train_dataset.classes
    loaders = build_loaders(
        datasets_tuple,
        batch_size=batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        device=device,
    )
    train_loader, val_loader, test_loader = loaders
    model = build_local_model(model_name, len(class_names), args.pretrained).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=True) if amp_enabled else None
    model_config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "paper_name": spec["paper_name"],
        "module": spec["module"],
        "input_size": input_size,
        "pretrained": args.pretrained,
        "class_names": class_names,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "params": parameter_count,
        "device": str(device),
        "amp": amp_enabled,
        "epochs": args.epochs,
        "batch_size": batch_size,
        "num_workers": args.num_workers,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "label_smoothing": args.label_smoothing,
        "selection_metric": args.selection_metric,
        "seed": args.seed,
        "noise_eval": args.noise_eval,
        "noise_seed": args.noise_seed,
        "split_root": str(args.split_root.resolve()),
        "sample_limits_per_class": {
            "train": args.max_train_per_class,
            "val": args.max_val_per_class,
            "test": args.max_test_per_class,
        },
    }
    write_json(model_dir / "run_config.json", model_config)
    write_json(model_dir / "class_mapping.json", {str(i): name for i, name in enumerate(class_names)})
    history: list[dict[str, Any]] = []
    best_value = -float("inf")
    best_epoch = 0
    best_val_metrics: dict[str, Any] = {}
    start_train = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, args.epochs + 1):
        current_lr = float(optimizer.param_groups[0]["lr"])
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            description=f"{model_name} train {epoch:03d}/{args.epochs:03d}",
        )
        val_metrics, _, _, _ = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            class_names=class_names,
            device=device,
            amp_enabled=amp_enabled,
            degraded=False,
            noise_seed=args.noise_seed,
            description=f"{model_name} val {epoch:03d}/{args.epochs:03d}",
            collect_rows=False,
        )
        if scheduler is not None:
            scheduler.step()
        value = selection_value(val_metrics, args.selection_metric)
        improved = value > best_value
        history_row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "selection_value": value,
            "is_best": int(improved),
        }
        history.append(history_row)
        write_csv(model_dir / "history.csv", history)
        if improved:
            best_value = value
            best_epoch = epoch
            best_val_metrics = val_metrics
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model": model_name,
                    "epoch": epoch,
                    "selection_metric": args.selection_metric,
                    "selection_value": value,
                    "class_names": class_names,
                    "input_size": input_size,
                },
                model_dir / "best_model.pth",
            )
        print(
            f"[{model_name}] epoch={epoch} lr={current_lr:.6g} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"best={best_value:.4f}"
        )
    train_time = time.perf_counter() - start_train
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model": model_name,
            "epoch": history[-1]["epoch"],
            "class_names": class_names,
            "input_size": input_size,
        },
        model_dir / "last_model.pth",
    )
    checkpoint = torch.load(model_dir / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    clean_metrics, clean_rows, clean_matrix, clean_report = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        class_names=class_names,
        device=device,
        amp_enabled=amp_enabled,
        degraded=False,
        noise_seed=args.noise_seed,
        description=f"{model_name} test clean",
        collect_rows=True,
    )
    noise_metrics: dict[str, Any] = {}
    noise_rows: list[dict[str, Any]] = []
    noise_matrix = np.zeros_like(clean_matrix)
    noise_report: dict[str, Any] = {}
    if args.noise_eval:
        noise_metrics, noise_rows, noise_matrix, noise_report = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            class_names=class_names,
            device=device,
            amp_enabled=amp_enabled,
            degraded=True,
            noise_seed=args.noise_seed,
            description=f"{model_name} test noise",
            collect_rows=True,
        )
    write_csv(model_dir / "test_predictions_clean.csv", clean_rows)
    write_matrix_csv(model_dir / "confusion_matrix_clean.csv", clean_matrix, class_names)
    write_json(model_dir / "classification_report_clean.json", clean_report)
    if args.noise_eval:
        write_csv(model_dir / "test_predictions_noise.csv", noise_rows)
        write_matrix_csv(model_dir / "confusion_matrix_noise.csv", noise_matrix, class_names)
        write_json(model_dir / "classification_report_noise.json", noise_report)
    peak_gpu_memory_mb = 0.0
    if device.type == "cuda":
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    report = {
        "model": model_name,
        "paper_name": spec["paper_name"],
        "module": spec["module"],
        "model_dir": str(model_dir.resolve()),
        "input_size": input_size,
        "pretrained": args.pretrained,
        "params_M": round(parameter_count / 1e6, 3),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "selection_metric": args.selection_metric,
        "best_selection_value": round(best_value, 6),
        "best_val": best_val_metrics,
        "train_time_s": round(train_time, 2),
        "peak_gpu_memory_mb": round(peak_gpu_memory_mb, 2),
        "clean": clean_metrics,
        "noise": noise_metrics,
        "accuracy_drop_percent": round(
            float(clean_metrics["accuracy_percent"])
            - float(noise_metrics.get("accuracy_percent", clean_metrics["accuracy_percent"])),
            2,
        ),
        "outputs": {
            "best_checkpoint": "best_model.pth",
            "last_checkpoint": "last_model.pth",
            "history": "history.csv",
            "clean_predictions": "test_predictions_clean.csv",
            "noise_predictions": "test_predictions_noise.csv" if args.noise_eval else "",
        },
    }
    write_json(model_dir / "metrics_full.json", report)
    del model, optimizer, scheduler, scaler
    del train_loader, val_loader, test_loader
    del train_dataset, val_dataset, test_dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train local models under wheat_original/src/models on data_split_811_full."
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(MODEL_REGISTRY),
        help=f"Comma-separated local model names. Available: {','.join(MODEL_REGISTRY)}",
    )
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=0, help="0 uses model default.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--scheduler", choices=["cosine", "step", "none"], default="cosine")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--step-size", type=int, default=10)
    parser.add_argument("--step-gamma", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument(
        "--selection-metric",
        choices=["val_macro_f1", "val_accuracy"],
        default="val_macro_f1",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-seed", type=int, default=20260710)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--input-size", type=int, default=0, help="0 uses model default.")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--noise-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-per-class", type=int, default=0)
    parser.add_argument("--max-val-per-class", type=int, default=0)
    parser.add_argument("--max-test-per-class", type=int, default=0)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace, model_names: list[str]) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.batch_size < 0:
        raise ValueError("--batch-size cannot be negative.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    if args.lr <= 0:
        raise ValueError("--lr must be positive.")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("--label-smoothing must be in [0, 1).")
    for split in ("train", "val", "test"):
        split_dir = args.split_root / split
        if not split_dir.exists():
            raise FileNotFoundError(split_dir)
    for model_name in model_names:
        module = importlib.import_module(MODEL_REGISTRY[model_name]["module"])
        if not hasattr(module, "build_model"):
            raise AttributeError(f"{MODEL_REGISTRY[model_name]['module']} has no build_model().")


def main() -> None:
    args = parse_args()
    args.split_root = args.split_root.resolve()
    args.output_root = args.output_root.resolve()
    model_names = parse_models(args.models)
    validate_args(args, model_names)
    device = resolve_device(args.device)
    preview = {
        "models": model_names,
        "split_root": str(args.split_root),
        "output_root": str(args.output_root),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "label_smoothing": args.label_smoothing,
        "selection_metric": args.selection_metric,
        "pretrained": args.pretrained,
        "amp": args.amp,
        "noise_eval": args.noise_eval,
        "seed": args.seed,
        "noise_seed": args.noise_seed,
        "device": str(device),
        "input_size": args.input_size,
        "model_specs": {name: MODEL_REGISTRY[name] for name in model_names},
        "sample_limits_per_class": {
            "train": args.max_train_per_class,
            "val": args.max_val_per_class,
            "test": args.max_test_per_class,
        },
    }
    print(json.dumps(preview, ensure_ascii=False, indent=2))
    if args.dry_run:
        print("Dry run complete. No output directory or model was created.")
        return
    run_name = args.run_name.strip() or f"src_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_root = args.output_root / run_name
    run_root.mkdir(parents=True, exist_ok=False)
    run_config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
        **preview,
    }
    write_json(run_root / "run_config.json", run_config)
    write_json(run_root / "environment.json", environment_info(device))
    rows: list[dict[str, Any]] = []
    for model_name in model_names:
        print("=" * 88)
        print(f"Starting local model: {model_name}")
        print("=" * 88)
        try:
            report = train_model(model_name=model_name, args=args, run_root=run_root, device=device)
            row = model_summary_row(report)
            rows.append(row)
            print(
                f"[DONE] {model_name}: clean_acc={row['clean_accuracy']:.4f}, "
                f"clean_macro_f1={row['clean_macro_f1']:.4f}"
            )
        except Exception as error:  # noqa: BLE001
            error_text = f"{type(error).__name__}: {error}"
            rows.append(
                {
                    "status": "error",
                    "model": model_name,
                    "model_dir": str((run_root / model_name).resolve()),
                    "error": error_text,
                }
            )
            (run_root / f"{model_name}_error.txt").write_text(
                traceback.format_exc(),
                encoding="utf-8",
            )
            print(f"[ERROR] {model_name}: {error_text}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if not args.continue_on_error:
                write_csv(run_root / "local_model_comparison.csv", rows)
                write_summary_markdown(run_root, run_config, rows)
                raise
        write_csv(run_root / "local_model_comparison.csv", rows)
        write_summary_markdown(run_root, run_config, rows)
    print(f"Local src/models training complete: {run_root}")
    print(f"Summary: {run_root / 'local_model_comparison_summary.md'}")


if __name__ == "__main__":
    main()
