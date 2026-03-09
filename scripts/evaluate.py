from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from common import ensure_directory, ensure_packages, resolve_relative_path
from models import SimpleCNN


EVALUATION_BATCH_SIZE = 256
METRICS_OUTPUT_PATH = Path("artifacts/results/metrics.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model on held-out data.")
    parser.add_argument("model_file", help="Path to artifacts/models/model.pt checkpoint")
    parser.add_argument("test_data", help="Path to artifacts/data/test.npz produced by preprocess.py")
    return parser.parse_args()


def load_checkpoint(path: Path) -> Dict:
    checkpoint_path = resolve_relative_path(path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "metadata" not in checkpoint:
        raise ValueError("Model checkpoint missing metadata required for evaluation")
    return checkpoint


def load_dataset(path: Path) -> Dict[str, torch.Tensor]:
    import numpy as np

    npz_path = resolve_relative_path(path)
    with np.load(npz_path) as data:
        images = data["images"].astype("float32")
        labels = data["labels"].astype("int64")
    return {"images": torch.from_numpy(images), "labels": torch.from_numpy(labels)}


def build_dataloader(tensors: Dict[str, torch.Tensor]) -> DataLoader:
    dataset = TensorDataset(tensors["images"], tensors["labels"])
    return DataLoader(dataset, batch_size=EVALUATION_BATCH_SIZE, shuffle=False)


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    class_correct: Dict[int, int] = defaultdict(int)
    class_totals: Dict[int, int] = defaultdict(int)

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            matches = predicted.eq(targets)
            correct += matches.sum().item()
            total_samples += targets.size(0)

            for label, match in zip(targets.cpu().tolist(), matches.cpu().tolist()):
                class_totals[label] += 1
                if match:
                    class_correct[label] += 1

    accuracy = correct / max(total_samples, 1)
    average_loss = total_loss / max(total_samples, 1)
    per_class_accuracy = {
        str(cls): class_correct.get(cls, 0) / max(class_totals.get(cls, 1), 1)
        for cls in class_totals
    }

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "num_samples": total_samples,
        "per_class_accuracy": per_class_accuracy,
    }


def main() -> None:
    args = parse_args()
    ensure_packages(["numpy"])
    device = torch.device("cpu")

    checkpoint = load_checkpoint(Path(args.model_file))
    metadata = checkpoint.get("metadata", {})
    tensors = load_dataset(Path(args.test_data))
    dataloader = build_dataloader(tensors)

    in_channels = metadata.get("in_channels")
    num_classes = metadata.get("num_classes")
    if in_channels is None or num_classes is None:
        raise ValueError("Model metadata must contain 'in_channels' and 'num_classes'")

    model = SimpleCNN(in_channels=int(in_channels), num_classes=int(num_classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    metrics = evaluate_model(model, dataloader, device)
    metrics.update(
        {
            "device": str(device),
            "history": metadata.get("history", {}),
            "epoch_durations": metadata.get("epoch_durations", []),
            "train_mean": metadata.get("mean"),
            "train_std": metadata.get("std"),
        }
    )

    metrics_path = resolve_relative_path(METRICS_OUTPUT_PATH)
    results_dir = ensure_directory(metrics_path.parent)
    metrics_path = results_dir / metrics_path.name
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Evaluation metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
