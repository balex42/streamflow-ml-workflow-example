"""Shared utilities for the ML workflow scripts."""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from typing import Iterable

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def ensure_packages(packages: Iterable[str]) -> None:
    """Install required Python packages into the runtime environment when missing."""

    for package in packages:
        try:
            importlib.import_module(package)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def ensure_directory(path: Path | str) -> Path:
    """Create ``path`` (and parents) when it does not exist and return it."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def resolve_relative_path(path: Path | str) -> Path:
    """Resolve ``path`` relative to the project root."""
    return (_PROJECT_ROOT / path).resolve()
