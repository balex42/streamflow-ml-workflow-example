from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from common import ensure_packages, resolve_relative_path


CURVES_OUTPUT_PATH = Path("artifacts/results/training_curves.png")
REPORT_OUTPUT_PATH = Path("artifacts/results/report.pdf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate training report PDF with curves and metrics.")
    parser.add_argument("metrics", help="Path to artifacts/results/metrics.json produced by evaluate.py")
    return parser.parse_args()


def load_metrics(path: Path) -> Dict:
    metrics_path = resolve_relative_path(path)
    return json.loads(metrics_path.read_text())


def ensure_plotting_packages() -> None:
    ensure_packages(["matplotlib", "numpy"])


def plot_history(history: Dict[str, List[float]], curves_path: Path) -> Dict[str, Path]:
    ensure_plotting_packages()
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(history["loss"]) + 1))
    figure_paths: Dict[str, Path] = {}

    if epochs:
        curves_fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes = axes.reshape(1, -1)[0]
        axes[0].plot(epochs, history["loss"], marker="o", color="#1f77b4")
        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, linestyle="--", alpha=0.5)

        axes[1].plot(epochs, [acc * 100 for acc in history["accuracy"]], marker="o", color="#ff7f0e")
        axes[1].set_title("Training Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_ylim(0, 100)
        axes[1].grid(True, linestyle="--", alpha=0.5)

        curves_fig.tight_layout()
        curves_path.parent.mkdir(parents=True, exist_ok=True)
        curves_fig.savefig(curves_path, dpi=200)
        figure_paths["curves"] = curves_path
        plt.close(curves_fig)
    else:
        figure_paths["curves"] = curves_path

    return figure_paths


def build_pdf_report(metrics: Dict, output_path: Path, curves_image: Path | None) -> None:
    ensure_plotting_packages()
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        # Page 1: Summary
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        title = "Model Evaluation Report"
        ax.set_title(title, fontsize=18, fontweight="bold", pad=20)

        summary_lines = [
            f"Overall Accuracy: {metrics['accuracy'] * 100:.2f}%",
            f"Average Loss: {metrics['loss']:.4f}",
            f"Number of Samples: {metrics['num_samples']}",
            f"Evaluation Device: {metrics['device']}",
        ]

        history = metrics["history"]
        if history:
            summary_lines.append(f"Training Epochs: {len(history['loss'])}")
        durations = metrics["epoch_durations"]
        if durations:
            summary_lines.append(
                f"Average Epoch Duration: {sum(durations) / max(len(durations), 1):.1f}s"
            )

        text_y = 0.85
        for line in summary_lines:
            ax.text(0.02, text_y, line, fontsize=12, transform=ax.transAxes)
            text_y -= 0.05

        per_class = metrics["per_class_accuracy"]
        if per_class:
            ax.text(0.02, text_y - 0.02, "Per-Class Accuracy:", fontsize=12, fontweight="bold", transform=ax.transAxes)
            text_y -= 0.08
            for cls, acc in sorted(per_class.items(), key=lambda x: x[0]):
                ax.text(0.04, text_y, f"Class {cls}: {acc * 100:.2f}%", fontsize=11, transform=ax.transAxes)
                text_y -= 0.04

        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Curves
        if curves_image is not None and curves_image.exists():
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            img = plt.imread(curves_image)
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)

        # Page 3: Raw data table
        if history:
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.axis("off")
            table_data = [
                ["Epoch", "Loss", "Accuracy (%)", "Duration (s)"]
            ]
            durations = metrics["epoch_durations"]
            for idx, (loss, acc) in enumerate(zip(history["loss"], history["accuracy"]), start=1):
                duration = durations[idx - 1] if idx - 1 < len(durations) else None
                table_data.append([idx, f"{loss:.4f}", f"{acc * 100:.2f}", f"{duration:.1f}" if duration else "-"])

            table = ax.table(cellText=table_data, loc="center", cellLoc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.2)
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    args = parse_args()
    metrics = load_metrics(Path(args.metrics))

    curves_path = resolve_relative_path(CURVES_OUTPUT_PATH)
    history = metrics["history"]
    curves_image = None
    if history and history["loss"]:
        generated = plot_history(history, curves_path)
        curves_image = generated.get("curves")

    report_output = resolve_relative_path(REPORT_OUTPUT_PATH)
    build_pdf_report(metrics, report_output, curves_image)
    print(f"Report generated at {report_output}")


if __name__ == "__main__":
    main()
