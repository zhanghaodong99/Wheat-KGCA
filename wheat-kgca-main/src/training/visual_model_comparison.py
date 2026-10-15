from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import random
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
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
from torchvision import datasets
from tqdm import tqdm


DEFAULT_SPLIT_ROOT = PROJECT_ROOT / "data" / "data_split_811_full"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "visual_model_comparison"
DEFAULT_SOURCE_MANIFEST = DEFAULT_SPLIT_ROOT / "wheat_split_811_manifest.csv"

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "dinov3_vits16": {
        "backend": "timm",
        "model_id": "vit_small_patch16_dinov3.lvd1689m",
        "role": "self-supervised foundation model",
        "year": 2025,
        "official_url": "https://github.com/facebookresearch/dinov3",
        "upstream_recipe": "DINOv3 official LVD-1689M pretrained checkpoint; downstream wheat fine-tuning is controlled.",
    },
    "siglip2_base_p16": {
        "backend": "timm",
        "model_id": "vit_base_patch16_siglip_224.v2_webli",
        "role": "vision-language pretrained encoder",
        "year": 2025,
        "official_url": "https://huggingface.co/google/siglip2-base-patch16-224",
        "upstream_recipe": "Google SigLIP 2 WebLI pretrained checkpoint; downstream wheat fine-tuning is controlled.",
    },
    "mambavision_t": {
        "backend": "mambavision",
        "model_id": "mamba_vision_T",
        "role": "Mamba-Transformer hybrid",
        "year": 2024,
        "official_url": "https://github.com/NVlabs/MambaVision",
        "checkpoint_url": "https://huggingface.co/nvidia/MambaVision-T-1K/resolve/main/mambavision_tiny_1k.pth.tar",
        "input_size": (3, 224, 224),
        "interpolation": "bicubic",
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "crop_pct": 1.0,
        "crop_mode": "center",
        "upstream_recipe": "Official ImageNet-1K recipe: 310 epochs, LAMB, lr=0.005, wd=0.05, warmup=20, mixup=0.8, cutmix=1.0.",
    },
    "convnextv2_tiny": {
        "backend": "timm",
        "model_id": "convnextv2_tiny.fcmae_ft_in22k_in1k",
        "role": "modern convolutional baseline",
        "year": 2023,
        "official_url": "https://github.com/facebookresearch/ConvNeXt-V2",
        "upstream_recipe": "Official Tiny fine-tuning recipe: 300 epochs, AdamW, blr=8e-4, wd=0.05, warmup=40, layer decay=0.9.",
    },
    "mobilenetv4_hybrid_m": {
        "backend": "timm",
        "model_id": "mobilenetv4_hybrid_medium.e500_r224_in1k",
        "role": "lightweight deployment baseline",
        "year": 2024,
        "official_url": "https://arxiv.org/abs/2404.10518",
        "upstream_recipe": "timm E500 ImageNet-1K checkpoint at its native 224 resolution.",
    },
    "swinv2_tiny": {
        "backend": "timm",
        "model_id": "swinv2_tiny_window16_256.ms_in1k",
        "role": "hierarchical Transformer baseline",
        "year": 2022,
        "official_url": "https://github.com/microsoft/Swin-Transformer",
        "upstream_recipe": "Official 256/window16 recipe: 300 epochs, AdamW, lr=5e-4, wd=0.05, warmup=20, drop path=0.2.",
    },
    "eva02_small_p14": {
        "backend": "timm",
        "model_id": "eva02_small_patch14_336.mim_in22k_ft_in1k",
        "role": "language-aligned masked-image pretrained Transformer",
        "year": 2024,
        "official_url": "https://github.com/baaivision/EVA/tree/master/EVA-02",
        "upstream_recipe": "Official EVA-02 Small ImageNet-22K to ImageNet-1K checkpoint; authors recommend a shorter schedule and smaller LR for transfer.",
    },
    "resnet50": {
        "backend": "timm",
        "model_id": "resnet50.a1_in1k",
        "role": "traditional CNN reference",
        "year": 2015,
        "official_url": "https://github.com/huggingface/pytorch-image-models",
        "upstream_recipe": "timm A1 ImageNet-1K checkpoint; retained only as a historical reference.",
    },
}


class ClassifierAdapter(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, None]:
        output = self.model(inputs)
        if hasattr(output, "logits"):
            output = output.logits
        if isinstance(output, (tuple, list)):
            output = output[0]
        return output, None


def inspect_official_data_config(model_name: str) -> dict[str, Any]:
    spec = MODEL_SPECS[model_name]
    if spec["backend"] == "mambavision":
        return {
            "input_size": tuple(spec["input_size"]),
            "interpolation": spec["interpolation"],
            "mean": tuple(spec["mean"]),
            "std": tuple(spec["std"]),
            "crop_pct": spec["crop_pct"],
            "crop_mode": spec["crop_mode"],
        }
    model = timm.create_model(str(spec["model_id"]), pretrained=False, num_classes=8)
    data_config = timm.data.resolve_model_data_config(model)
    del model
    gc.collect()
    return data_config


class ImageFolderWithMetadata(datasets.ImageFolder):
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str, int]:
        image, target = super().__getitem__(index)
        image_path = self.samples[index][0]
        return image, target, image_path, index


def parse_csv_items(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_models(text: str) -> list[str]:
    model_names = parse_csv_items(text)
    unknown = [name for name in model_names if name not in MODEL_SPECS]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}. Available: {sorted(MODEL_SPECS)}")
    if len(set(model_names)) != len(model_names):
        raise ValueError(f"Duplicate model names are not allowed: {model_names}")
    if not model_names:
        raise ValueError("At least one model must be selected.")
    return model_names


def atomic_json_dump(path: Path, data: Any) -> None:
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
        for column_index, predicted_label in enumerate(labels):
            row[predicted_label] = int(matrix[row_index, column_index])
        rows.append(row)
    write_csv(path, rows)


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def build_transforms(data_config: dict[str, Any]) -> tuple[Any, Any]:
    train_transform = timm.data.create_transform(
        input_size=tuple(data_config["input_size"]),
        is_training=True,
        color_jitter=0.4,
        auto_augment="rand-m9-mstd0.5-inc1",
        interpolation=str(data_config["interpolation"]),
        mean=tuple(data_config["mean"]),
        std=tuple(data_config["std"]),
        re_prob=0.25,
        re_mode="pixel",
        re_count=1,
        hflip=0.5,
        vflip=0.0,
    )
    eval_transform = timm.data.create_transform(
        **data_config,
        is_training=False,
    )
    return train_transform, eval_transform


def class_names_from_split(split_root: Path) -> list[str]:
    return sorted(path.name for path in (split_root / "train").iterdir() if path.is_dir())


def build_model_and_data_config(
    model_name: str,
    num_classes: int,
    pretrained: bool,
    mambavision_checkpoint: Path | None,
) -> tuple[nn.Module, dict[str, Any]]:
    spec = MODEL_SPECS[model_name]
    if spec["backend"] == "timm":
        inner_model = timm.create_model(
            str(spec["model_id"]),
            pretrained=pretrained,
            num_classes=num_classes,
        )
        data_config = timm.data.resolve_model_data_config(inner_model)
        return ClassifierAdapter(inner_model), data_config

    try:
        from mambavision import create_model as create_mambavision_model
    except ImportError as error:
        raise RuntimeError(
            "MambaVision is not installed. Run: conda run -n wheat_gpu "
            "python -m pip install mambavision"
        ) from error

    model_path = ""
    if pretrained:
        if mambavision_checkpoint is None or not mambavision_checkpoint.exists():
            raise FileNotFoundError(
                "MambaVision pretrained checkpoint is required. Download the official "
                f"checkpoint from {spec['checkpoint_url']} and pass --mambavision-checkpoint."
            )
        model_path = str(mambavision_checkpoint.resolve())
    inner_model = create_mambavision_model(
        str(spec["model_id"]),
        pretrained=pretrained,
        model_path=model_path,
        num_classes=num_classes,
    )
    data_config = {
        "input_size": tuple(spec["input_size"]),
        "interpolation": spec["interpolation"],
        "mean": tuple(getattr(inner_model, "mean", spec["mean"])),
        "std": tuple(getattr(inner_model, "std", spec["std"])),
        "crop_pct": float(getattr(inner_model, "crop_pct", spec["crop_pct"])),
        "crop_mode": str(getattr(inner_model, "crop_mode", spec["crop_mode"])),
    }
    return ClassifierAdapter(inner_model), data_config


def limit_samples_per_class(
    dataset: ImageFolderWithMetadata,
    max_per_class: int,
    seed: int,
) -> None:
    if max_per_class <= 0:
        return

    by_class: dict[int, list[int]] = defaultdict(list)
    for index, (_, target) in enumerate(dataset.samples):
        by_class[int(target)].append(index)

    rng = random.Random(seed)
    selected_indices: list[int] = []
    for class_index in sorted(by_class):
        indices = by_class[class_index]
        rng.shuffle(indices)
        selected_indices.extend(indices[:max_per_class])
    selected_indices.sort()

    dataset.samples = [dataset.samples[index] for index in selected_indices]
    dataset.imgs = dataset.samples
    dataset.targets = [target for _, target in dataset.samples]


def build_datasets(
    split_root: Path,
    data_config: dict[str, Any],
    seed: int,
    max_train_per_class: int,
    max_val_per_class: int,
    max_test_per_class: int,
) -> tuple[ImageFolderWithMetadata, ImageFolderWithMetadata, ImageFolderWithMetadata]:
    train_transform, eval_transform = build_transforms(data_config)
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
    shared = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
        "worker_init_fn": seed_worker,
    }
    if num_workers > 0:
        shared["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **shared,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **shared)
    test_loader = DataLoader(test_dataset, shuffle=False, **shared)
    return train_loader, val_loader, test_loader


def normalize_manifest_path(path_value: str) -> str:
    path = Path(str(path_value))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return os.path.normcase(str(path.resolve()))


def load_source_type_map(manifest_path: Path) -> dict[str, str]:
    if not manifest_path.exists():
        return {}
    manifest = pd.read_csv(manifest_path)
    if "target_path" not in manifest or "source_type" not in manifest:
        return {}
    return {
        normalize_manifest_path(str(row.target_path)): str(row.source_type)
        for row in manifest.itertuples(index=False)
    }


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
            factor = 0.5 + float(
                torch.rand((1,), generator=generator, device=images.device).item()
            )
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
            top = int(
                torch.randint(0, max_top, (1,), generator=generator, device=images.device).item()
            )
            left = int(
                torch.randint(0, max_left, (1,), generator=generator, device=images.device).item()
            )
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
    accuracy = accuracy_score(y_true, y_pred)
    return {
        "accuracy": round(float(accuracy), 6),
        "accuracy_percent": round(float(accuracy) * 100, 2),
        "macro_f1": round(
            float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            6,
        ),
        "weighted_f1": round(
            float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
            6,
        ),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 6),
        "samples": len(y_true),
    }


def metrics_by_source(rows: list[dict[str, Any]], class_names: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    source_types = sorted({str(row.get("source_type", "")) for row in rows if row.get("source_type")})
    for source_type in source_types:
        source_rows = [row for row in rows if row.get("source_type") == source_type]
        metrics = classification_metrics(
            [int(row["true_idx"]) for row in source_rows],
            [int(row["pred_idx"]) for row in source_rows],
            class_names,
        )
        result[source_type] = metrics
    return result


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
            logits, _ = model(images)
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

    return {
        "loss": round(total_loss / max(total_count, 1), 6),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    class_names: list[str],
    source_type_map: dict[str, str],
    device: torch.device,
    amp_enabled: bool,
    degraded: bool,
    noise_seed: int,
    description: str,
    collect_rows: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    total_loss = 0.0
    total_count = 0
    start_time = time.perf_counter()

    with torch.inference_mode():
        progress = tqdm(loader, desc=description, dynamic_ncols=True, leave=False)
        for images, labels, image_paths, sample_indices in progress:
            images = images.to(device, non_blocking=device.type == "cuda")
            labels_device = labels.to(device, non_blocking=device.type == "cuda")
            degradation_modes = ["none"] * len(image_paths)
            if degraded:
                images, degradation_modes = apply_deterministic_field_degradation(
                    images,
                    sample_indices,
                    noise_seed,
                )

            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits, _ = model(images)
                loss = criterion(logits, labels_device)

            probabilities = torch.softmax(logits.float(), dim=1)
            predictions = logits.argmax(dim=1)
            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size

            true_values = labels.tolist()
            predicted_values = predictions.detach().cpu().tolist()
            y_true.extend(true_values)
            y_pred.extend(predicted_values)

            if collect_rows:
                for item_index, image_path in enumerate(image_paths):
                    true_index = int(true_values[item_index])
                    predicted_index = int(predicted_values[item_index])
                    normalized_path = os.path.normcase(str(Path(image_path).resolve()))
                    rows.append(
                        {
                            "image_path": str(Path(image_path).resolve()),
                            "source_type": source_type_map.get(normalized_path, "unknown"),
                            "true_idx": true_index,
                            "true_label": class_names[true_index],
                            "pred_idx": predicted_index,
                            "pred_label": class_names[predicted_index],
                            "correct": int(true_index == predicted_index),
                            "confidence": round(
                                float(probabilities[item_index, predicted_index].cpu().item()),
                                8,
                            ),
                            "degradation": degradation_modes[item_index],
                        }
                    )

    elapsed = time.perf_counter() - start_time
    metrics = classification_metrics(y_true, y_pred, class_names)
    metrics.update(
        {
            "loss": round(total_loss / max(total_count, 1), 6),
            "elapsed_s": round(elapsed, 3),
            "fps": round(total_count / elapsed, 2) if elapsed > 0 else 0.0,
        }
    )
    labels = list(range(len(class_names)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    if collect_rows:
        metrics["by_source"] = metrics_by_source(rows, class_names)
    return metrics, rows, matrix, report


def build_optimizer(args: argparse.Namespace, model: nn.Module) -> optim.Optimizer:
    if args.optimizer == "adamw":
        return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )


def build_scheduler(
    args: argparse.Namespace,
    optimizer: optim.Optimizer,
) -> optim.lr_scheduler.LRScheduler | None:
    if args.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs),
            eta_min=args.min_lr,
        )
    if args.scheduler == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, args.step_size),
            gamma=args.step_gamma,
        )
    return None


def selection_value(metrics: dict[str, Any], selection_metric: str) -> float:
    if selection_metric == "val_accuracy":
        return float(metrics["accuracy"])
    return float(metrics["macro_f1"])


def source_metric_rows(model_name: str, split_name: str, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for source_type, values in metrics.get("by_source", {}).items():
        rows.append(
            {
                "model": model_name,
                "evaluation": split_name,
                "source_type": source_type,
                **values,
            }
        )
    return rows


def model_summary_row(report: dict[str, Any]) -> dict[str, Any]:
    clean = report["clean"]
    noise = report.get("noise", {})
    return {
        "status": "ok",
        "model": report["model"],
        "official_model_id": report["official_model_id"],
        "input_size": report["input_size"],
        "pretrained": report["pretrained"],
        "params_M": report["params_M"],
        "trainable_params_M": report["trainable_params_M"],
        "best_epoch": report["best_epoch"],
        "selection_metric": report["selection_metric"],
        "best_selection_value": report["best_selection_value"],
        "best_val_accuracy": report["best_val"]["accuracy"],
        "best_val_macro_f1": report["best_val"]["macro_f1"],
        "clean_accuracy": clean["accuracy"],
        "clean_macro_f1": clean["macro_f1"],
        "clean_balanced_accuracy": clean["balanced_accuracy"],
        "clean_weighted_f1": clean["weighted_f1"],
        "noise_accuracy": noise.get("accuracy", ""),
        "noise_macro_f1": noise.get("macro_f1", ""),
        "noise_balanced_accuracy": noise.get("balanced_accuracy", ""),
        "accuracy_drop_percent": report.get("accuracy_drop_percent", ""),
        "train_time_s": report["train_time_s"],
        "clean_fps": clean["fps"],
        "noise_fps": noise.get("fps", ""),
        "peak_gpu_memory_mb": report["peak_gpu_memory_mb"],
        "model_dir": report["model_dir"],
        "error": "",
    }


def format_value(value: Any, digits: int = 4) -> str:
    if value == "" or value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def write_comparison_markdown(
    path: Path,
    run_config: dict[str, Any],
    rows: list[dict[str, Any]],
    final: bool,
) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    error_rows = [row for row in rows if row.get("status") != "ok"]
    ranked = sorted(
        ok_rows,
        key=lambda row: float(row.get("best_val_macro_f1", 0.0) or 0.0),
        reverse=True,
    )

    table_lines = [
        "| Model | Input | Params (M) | Best epoch | Val Macro-F1 | Test Acc | Test Macro-F1 | Noise Acc | Train time (s) | FPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranked:
        table_lines.append(
            "| {model} | {input_size} | {params} | {epoch} | {val_f1} | {test_acc} | "
            "{test_f1} | {noise_acc} | {train_time} | {fps} |".format(
                model=row["model"],
                input_size=row["input_size"],
                params=format_value(row["params_M"], 2),
                epoch=row["best_epoch"],
                val_f1=format_value(row["best_val_macro_f1"]),
                test_acc=format_value(row["clean_accuracy"]),
                test_f1=format_value(row["clean_macro_f1"]),
                noise_acc=format_value(row.get("noise_accuracy", "")),
                train_time=format_value(row["train_time_s"], 1),
                fps=format_value(row["clean_fps"], 2),
            )
        )
    if not ranked:
        table_lines.append("| No completed models | | | | | | | | | |")

    error_section = ""
    if error_rows:
        error_lines = ["## Failed Models", ""]
        for row in error_rows:
            error_lines.append(f"- `{row.get('model', 'unknown')}`: `{row.get('error', '')}`")
        error_section = "\n".join(error_lines)

    best_line = "No completed model yet."
    if ranked:
        best = ranked[0]
        best_line = (
            f"Best validation model: `{best['model']}` "
            f"(Val Macro-F1={float(best['best_val_macro_f1']):.4f})."
        )

    status_text = "complete" if final else "partial"
    markdown = f"""# Visual Model Comparison Summary

Status: **{status_text}**

## Unified Protocol

```json
{json.dumps(run_config, ensure_ascii=False, indent=2)}
```

## Results

{chr(10).join(table_lines)}

## Selection

{best_line}

- Downstream fusion backbone selection must use validation metrics, not test metrics.
- Noise evaluation uses deterministic per-sample corruption shared by all models in this run.
- This run is a new unified comparison. Do not mix its ranking with the historical RepVIT run.

{error_section}
"""
    path.write_text(markdown, encoding="utf-8")


def write_comparison_outputs(
    run_root: Path,
    run_config: dict[str, Any],
    rows: list[dict[str, Any]],
    final: bool,
) -> None:
    write_csv(run_root / "visual_model_comparison.csv", rows)
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    best = None
    if ok_rows:
        best = max(ok_rows, key=lambda row: float(row.get("best_val_macro_f1", 0.0) or 0.0))
    atomic_json_dump(
        run_root / "visual_model_comparison.json",
        {
            "status": "complete" if final else "partial",
            "run_config": run_config,
            "rows": rows,
            "best_by_validation_macro_f1": best,
        },
    )
    write_comparison_markdown(
        run_root / "visual_model_comparison_summary.md",
        run_config,
        rows,
        final,
    )


def train_model(
    model_name: str,
    args: argparse.Namespace,
    run_root: Path,
    source_type_map: dict[str, str],
    device: torch.device,
) -> dict[str, Any]:
    seed_everything(args.seed)
    spec = MODEL_SPECS[model_name]
    model_dir = run_root / model_name
    model_dir.mkdir(parents=True, exist_ok=False)

    expected_class_names = class_names_from_split(args.split_root)
    model, data_config = build_model_and_data_config(
        model_name=model_name,
        num_classes=len(expected_class_names),
        pretrained=args.pretrained,
        mambavision_checkpoint=args.mambavision_checkpoint,
    )
    model = model.to(device)
    input_size = int(data_config["input_size"][-1])

    datasets_tuple = build_datasets(
        split_root=args.split_root,
        data_config=data_config,
        seed=args.seed,
        max_train_per_class=args.max_train_per_class,
        max_val_per_class=args.max_val_per_class,
        max_test_per_class=args.max_test_per_class,
    )
    train_dataset, val_dataset, test_dataset = datasets_tuple
    class_names = train_dataset.classes
    if class_names != expected_class_names:
        raise ValueError(
            f"Dataset classes changed during model construction: {expected_class_names} / {class_names}"
        )
    train_loader, val_loader, test_loader = build_loaders(
        datasets_tuple,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        device=device,
    )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=True) if amp_enabled else None

    model_config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "official_model_id": spec["model_id"],
        "backend": spec["backend"],
        "role": spec["role"],
        "publication_year": spec["year"],
        "official_url": spec["official_url"],
        "upstream_recipe_reference": spec["upstream_recipe"],
        "official_data_config": data_config,
        "input_size": input_size,
        "pretrained": args.pretrained,
        "class_names": class_names,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "params": parameter_count,
        "trainable_params": trainable_parameter_count,
        "device": str(device),
        "amp": amp_enabled,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "momentum": args.momentum,
        "scheduler": args.scheduler,
        "min_lr": args.min_lr,
        "step_size": args.step_size,
        "step_gamma": args.step_gamma,
        "label_smoothing": args.label_smoothing,
        "selection_metric": args.selection_metric,
        "early_stopping_patience": args.early_stopping_patience,
        "seed": args.seed,
        "noise_eval": args.noise_eval,
        "noise_seed": args.noise_seed,
        "split_root": str(args.split_root.resolve()),
        "source_manifest": str(args.source_manifest.resolve()),
        "sample_limits_per_class": {
            "train": args.max_train_per_class,
            "val": args.max_val_per_class,
            "test": args.max_test_per_class,
        },
    }
    atomic_json_dump(model_dir / "run_config.json", model_config)
    atomic_json_dump(
        model_dir / "class_mapping.json",
        {str(index): name for index, name in enumerate(class_names)},
    )

    history: list[dict[str, Any]] = []
    best_value = -float("inf")
    best_epoch = 0
    best_val_metrics: dict[str, Any] = {}
    epochs_without_improvement = 0
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
            source_type_map=source_type_map,
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
            epochs_without_improvement = 0
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
        else:
            epochs_without_improvement += 1

        atomic_json_dump(
            model_dir / "training_progress.json",
            {
                "last_epoch": epoch,
                "best_epoch": best_epoch,
                "best_selection_value": best_value,
                "epochs_without_improvement": epochs_without_improvement,
            },
        )
        print(
            f"[{model_name}] epoch={epoch} lr={current_lr:.6g} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"best={best_value:.4f}"
        )

        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"[{model_name}] early stopping after {epoch} epochs; "
                f"no improvement for {epochs_without_improvement} epochs."
            )
            break

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

    best_checkpoint = torch.load(model_dir / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    clean_metrics, clean_rows, clean_matrix, clean_report = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        class_names=class_names,
        source_type_map=source_type_map,
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
            source_type_map=source_type_map,
            device=device,
            amp_enabled=amp_enabled,
            degraded=True,
            noise_seed=args.noise_seed,
            description=f"{model_name} test noise",
            collect_rows=True,
        )

    write_csv(model_dir / "test_predictions_clean.csv", clean_rows)
    write_matrix_csv(model_dir / "confusion_matrix_clean.csv", clean_matrix, class_names)
    atomic_json_dump(model_dir / "classification_report_clean.json", clean_report)
    if args.noise_eval:
        write_csv(model_dir / "test_predictions_noise.csv", noise_rows)
        write_matrix_csv(model_dir / "confusion_matrix_noise.csv", noise_matrix, class_names)
        atomic_json_dump(model_dir / "classification_report_noise.json", noise_report)

    all_source_rows = source_metric_rows(model_name, "clean", clean_metrics)
    if args.noise_eval:
        all_source_rows.extend(source_metric_rows(model_name, "noise", noise_metrics))
    write_csv(model_dir / "metrics_by_source.csv", all_source_rows)

    peak_gpu_memory_mb = 0.0
    if device.type == "cuda":
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    report = {
        "model": model_name,
        "official_model_id": spec["model_id"],
        "model_dir": str(model_dir.resolve()),
        "input_size": input_size,
        "pretrained": args.pretrained,
        "params_M": round(parameter_count / 1e6, 3),
        "trainable_params_M": round(trainable_parameter_count / 1e6, 3),
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
            "source_metrics": "metrics_by_source.csv",
        },
    }
    atomic_json_dump(model_dir / "metrics_full.json", report)

    del model, optimizer, scheduler, scaler
    del train_loader, val_loader, test_loader
    del train_dataset, val_dataset, test_dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def environment_info(device: torch.device) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "pillow": PIL.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and compare wheat visual backbones under one fixed protocol."
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(MODEL_SPECS),
        help=f"Comma-separated model names. Available: {','.join(MODEL_SPECS)}",
    )
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument(
        "--mambavision-checkpoint",
        type=Path,
        default=Path(os.environ["MAMBAVISION_T_CHECKPOINT"])
        if os.environ.get("MAMBAVISION_T_CHECKPOINT")
        else None,
        help="Official mambavision_tiny_1k.pth.tar path. Can also use MAMBAVISION_T_CHECKPOINT.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
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
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-seed", type=int, default=20260710)
    parser.add_argument("--device", type=str, default="auto")
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
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
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
    available_timm_models = set(timm.list_models())
    for model_name in model_names:
        spec = MODEL_SPECS[model_name]
        if spec["backend"] == "timm" and str(spec["model_id"]) not in available_timm_models:
            raise RuntimeError(f"timm model is unavailable: {spec['model_id']}")


def main() -> None:
    args = parse_args()
    args.split_root = args.split_root.resolve()
    args.source_manifest = args.source_manifest.resolve()
    args.output_root = args.output_root.resolve()
    if args.mambavision_checkpoint is not None:
        args.mambavision_checkpoint = args.mambavision_checkpoint.resolve()
    model_names = parse_models(args.models)
    validate_args(args, model_names)
    device = resolve_device(args.device)
    resolved_data_configs = {
        name: inspect_official_data_config(name) for name in model_names
    }

    preview = {
        "models": model_names,
        "split_root": str(args.split_root),
        "source_manifest": str(args.source_manifest),
        "mambavision_checkpoint": str(args.mambavision_checkpoint.resolve())
        if args.mambavision_checkpoint
        else "",
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
        "model_specs": {
            name: {
                **{
                    key: value
                    for key, value in MODEL_SPECS[name].items()
                    if key not in {"checkpoint_url"}
                },
                "resolved_official_data_config": resolved_data_configs[name],
            }
            for name in model_names
        },
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

    run_name = args.run_name.strip() or f"visual_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_root = args.output_root / run_name
    run_root.mkdir(parents=True, exist_ok=False)
    run_config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
        **preview,
    }
    atomic_json_dump(run_root / "run_config.json", run_config)
    atomic_json_dump(run_root / "environment.json", environment_info(device))

    source_type_map = load_source_type_map(args.source_manifest)
    rows: list[dict[str, Any]] = []
    for model_name in model_names:
        print("=" * 88)
        print(f"Starting visual model: {model_name}")
        print("=" * 88)
        try:
            report = train_model(
                model_name=model_name,
                args=args,
                run_root=run_root,
                source_type_map=source_type_map,
                device=device,
            )
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
                write_comparison_outputs(run_root, run_config, rows, final=False)
                raise

        write_comparison_outputs(run_root, run_config, rows, final=False)

    write_comparison_outputs(run_root, run_config, rows, final=True)
    print(f"Visual comparison complete: {run_root}")
    print(f"Summary: {run_root / 'visual_model_comparison_summary.md'}")


if __name__ == "__main__":
    main()
