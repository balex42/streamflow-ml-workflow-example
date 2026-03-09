from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

from common import ensure_directory, ensure_packages, resolve_relative_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize dataset and create train/test splits.")
    parser.add_argument("raw_data", help="Path to raw_data.npz produced by download.py")
    return parser.parse_args()


def load_raw_arrays(path: Path) -> Tuple[Dict[str, object], Dict[str, object]]:
    import numpy as np

    with np.load(path, allow_pickle=True) as archive:
        data = {key: archive[key] for key in archive.files}

    metadata_path = path.parent / "raw_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing required metadata file: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())

    return data, metadata


def extract_splits(arrays: Dict[str, object]) -> Tuple[Dict[str, object], Dict[str, object]]:
    import numpy as np

    expected_keys = {"train_images", "train_labels", "test_images", "test_labels"}
    if not expected_keys.issubset(arrays):
        raise ValueError("raw_data.npz must contain train_images, train_labels, test_images, and test_labels")

    train = {
        "images": np.asarray(arrays["train_images"]),
        "labels": np.asarray(arrays["train_labels"]),
    }
    test = {
        "images": np.asarray(arrays["test_images"]),
        "labels": np.asarray(arrays["test_labels"]),
    }
    return train, test


def normalize_images(
    train: Dict[str, object],
    test: Dict[str, object],
) -> Tuple[Dict[str, object], Dict[str, object], Tuple[float, ...], Tuple[float, ...]]:
    import numpy as np

    train_images = np.asarray(train["images"], dtype="float32")
    test_images = np.asarray(test["images"], dtype="float32")

    if train_images.ndim != 4 or test_images.ndim != 4:
        raise ValueError("Expected image tensors with shape (N, C, H, W)")

    train_images /= 255.0
    test_images /= 255.0

    reduce_axes = (0, 2, 3)
    mean = train_images.mean(axis=reduce_axes, keepdims=True)
    std = train_images.std(axis=reduce_axes, keepdims=True)
    std[std < 1e-6] = 1.0

    train_images = (train_images - mean) / std
    test_images = (test_images - mean) / std

    train_out = {
        "images": train_images,
        "labels": np.asarray(train["labels"], dtype="int64"),
    }
    test_out = {
        "images": test_images,
        "labels": np.asarray(test["labels"], dtype="int64"),
    }
    channel_mean = tuple(float(x) for x in mean.reshape(-1))
    channel_std = tuple(float(x) for x in std.reshape(-1))
    return train_out, test_out, channel_mean, channel_std


def save_split(
    train: Dict[str, object],
    test: Dict[str, object],
    metadata: Dict[str, str],
    channel_mean: Tuple[float, ...],
    channel_std: Tuple[float, ...],
) -> None:
    import numpy as np

    data_dir = ensure_directory(resolve_relative_path("artifacts/data"))
    train_path = data_dir / "train.npz"
    test_path = data_dir / "test.npz"

    np.savez_compressed(train_path, images=train["images"], labels=train["labels"], mean=channel_mean, std=channel_std)
    np.savez_compressed(test_path, images=test["images"], labels=test["labels"], mean=channel_mean, std=channel_std)

    metadata_out = {
        **metadata,
        "channel_mean": channel_mean,
        "channel_std": channel_std,
        "train_samples": int(train["images"].shape[0]),
        "test_samples": int(test["images"].shape[0]),
        "normalization": "standardize_per_channel",
    }
    metadata_path = data_dir / "preprocess_metadata.json"
    metadata_path.write_text(json.dumps(metadata_out, indent=2))


def main() -> None:
    args = parse_args()
    ensure_packages(["numpy"])

    raw_path = resolve_relative_path(args.raw_data)
    arrays, metadata = load_raw_arrays(raw_path)
    train_split, test_split = extract_splits(arrays)
    train_norm, test_norm, channel_mean, channel_std = normalize_images(train_split, test_split)
    save_split(train_norm, test_norm, metadata, channel_mean, channel_std)
    print("Saved normalized datasets to artifacts/data/train.npz and artifacts/data/test.npz")


if __name__ == "__main__":
    main()
