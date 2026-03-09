from __future__ import annotations

import argparse
import pickle
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Dict, Tuple

from common import ensure_directory, ensure_packages, resolve_relative_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a CIFAR-10 archive and store it as NPZ.")
    parser.add_argument(
        "dataset_url",
        help="Direct URL to the CIFAR-10 tar.gz archive.",
    )
    return parser.parse_args()


def download_url(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        data = response.read()
    destination.write_bytes(data)


def load_pickle(path: Path) -> Dict:
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def load_cifar10(source_url: str) -> Tuple[Dict[str, object], Dict[str, object]]:
    import numpy as np

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        archive_path = tmpdir_path / "cifar10.tar.gz"
        download_url(source_url, archive_path)
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(tmpdir_path)

        dataset_dirs = list(tmpdir_path.glob("**/cifar-10-batches-py"))
        if not dataset_dirs:
            raise FileNotFoundError("Extracted CIFAR-10 archive but could not locate batches directory")
        dataset_dir = dataset_dirs[0]
        train_images, train_labels = [], []
        for i in range(1, 6):
            batch = load_pickle(dataset_dir / f"data_batch_{i}")
            train_images.append(batch["data"])
            train_labels.extend(batch["labels"])

        test_batch = load_pickle(dataset_dir / "test_batch")
        test_images = test_batch["data"]
        test_labels = test_batch["labels"]

        train_images_np = (
            np.concatenate(train_images, axis=0).reshape(-1, 3, 32, 32).astype("uint8")
        )
        train_labels_np = np.asarray(train_labels, dtype="int64")
        test_images_np = np.asarray(test_images, dtype="uint8").reshape(-1, 3, 32, 32)
        test_labels_np = np.asarray(test_labels, dtype="int64")

        meta = load_pickle(dataset_dir / "batches.meta")
        meta_labels = meta.get("label_names")
        if meta_labels is None:
            meta_labels = meta.get(b"label_names", [])
        class_names = [name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in meta_labels]

    arrays = {
        "train_images": train_images_np,
        "train_labels": train_labels_np,
        "test_images": test_images_np,
        "test_labels": test_labels_np,
    }
    metadata = {"dataset": "cifar10", "num_classes": len(class_names), "class_names": class_names}
    return arrays, metadata


def save_outputs(arrays: Dict[str, object], metadata: Dict[str, object]) -> None:
    import json
    import numpy as np

    data_dir = ensure_directory(resolve_relative_path("artifacts/data"))
    output_path = data_dir / "raw_data.npz"
    np.savez_compressed(output_path, **arrays)

    metadata_path = data_dir / "raw_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))


def main() -> None:
    args = parse_args()
    ensure_packages(["numpy"])

    arrays, metadata = load_cifar10(args.dataset_url)

    save_outputs(arrays, metadata)
    output_npz = resolve_relative_path("artifacts/data/raw_data.npz")
    print(f"Saved dataset to {output_npz}")


if __name__ == "__main__":
    main()
