from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch  # type: ignore[import]
import torch.nn as nn  # type: ignore[import]
import torch.optim as optim  # type: ignore[import]
from torch.utils.data import DataLoader, TensorDataset  # type: ignore[import]

from common import ensure_directory, ensure_packages, resolve_relative_path
from models import SimpleCNN


TRAINING_EPOCHS = 5
TRAINING_BATCH_SIZE = 128
TRAINING_LEARNING_RATE = 1e-3
TRAINING_WEIGHT_DECAY = 1e-4
TRAINING_NUM_WORKERS = 2
MODEL_OUTPUT_PATH = Path("artifacts/models/model.pt")
HISTORY_OUTPUT_PATH = Path("artifacts/results/train_history.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small CNN on preprocessed data.")
    parser.add_argument("train_data", help="Path to artifacts/data/train.npz produced by preprocess.py")
    return parser.parse_args()


def set_seed() -> None:
    torch.manual_seed(42)


def load_dataset(path: Path) -> Dict[str, torch.Tensor]:
    import numpy as np

    npz_path = resolve_relative_path(path)
    with np.load(npz_path) as data:
        images = data["images"].astype("float32")
        labels = data["labels"].astype("int64")
        mean = data.get("mean")
        std = data.get("std")

    tensor_data = {
        "images": torch.from_numpy(images),
        "labels": torch.from_numpy(labels),
        "mean": mean.tolist() if hasattr(mean, "tolist") else None,
        "std": std.tolist() if hasattr(std, "tolist") else None,
    }
    return tensor_data


def build_dataloader(images: torch.Tensor, labels: torch.Tensor) -> DataLoader:
    dataset = TensorDataset(images, labels)
    return DataLoader(
        dataset,
        batch_size=TRAINING_BATCH_SIZE,
        shuffle=True,
        num_workers=TRAINING_NUM_WORKERS,
        pin_memory=False,
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return {"loss": epoch_loss, "accuracy": epoch_acc}


def main() -> None:
    args = parse_args()
    ensure_packages(["numpy"])

    set_seed()
    device = torch.device("cpu")

    dataset = load_dataset(Path(args.train_data))
    images = dataset["images"]
    labels = dataset["labels"]
    num_classes = int(labels.max().item()) + 1
    in_channels = int(images.shape[1]) if images.ndim >= 2 else 1

    dataloader = build_dataloader(images, labels)
    model = SimpleCNN(in_channels=in_channels, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=TRAINING_LEARNING_RATE,
        weight_decay=TRAINING_WEIGHT_DECAY,
    )

    history: Dict[str, List[float]] = {"loss": [], "accuracy": []}
    epoch_durations: List[float] = []

    for epoch in range(TRAINING_EPOCHS):
        start_time = time.time()
        metrics = train_one_epoch(model, dataloader, criterion, optimizer, device)
        duration = time.time() - start_time

        history["loss"].append(metrics["loss"])
        history["accuracy"].append(metrics["accuracy"])
        epoch_durations.append(duration)

        print(
            f"Epoch {epoch + 1}/{TRAINING_EPOCHS} | Loss: {metrics['loss']:.4f} | "
            f"Accuracy: {metrics['accuracy'] * 100:.2f}% | Duration: {duration:.1f}s"
        )

    metadata = {
        "epochs": TRAINING_EPOCHS,
        "batch_size": TRAINING_BATCH_SIZE,
        "learning_rate": TRAINING_LEARNING_RATE,
        "weight_decay": TRAINING_WEIGHT_DECAY,
        "num_classes": num_classes,
        "in_channels": in_channels,
        "history": history,
        "epoch_durations": epoch_durations,
        "mean": dataset.get("mean"),
        "std": dataset.get("std"),
        "device": str(device),
    }

    model_path = resolve_relative_path(MODEL_OUTPUT_PATH)
    model_dir = ensure_directory(model_path.parent)
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, model_path)
    print(f"Model saved to {model_path}")

    history_path = resolve_relative_path(HISTORY_OUTPUT_PATH)
    history_dir = ensure_directory(history_path.parent)
    history_payload = {**metadata, "model_path": str(model_path)}
    history_path.write_text(json.dumps(history_payload, indent=2))
    print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    main()
