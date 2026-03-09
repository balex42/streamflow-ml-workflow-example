# ML Workflow with StreamFlow and CWL

This repository packages a complete ML pipeline (download → preprocess → train → evaluate → report) using
[StreamFlow](https://streamflow.di.unito.it/) to orchestrate CWL tools inside containers. The default example trains a
lightweight CNN on CIFAR-10 using PyTorch.

## Repository layout

| Path | Description |
| --- | --- |
| `cwl/` | Individual CWL `CommandLineTool` definitions and the top-level workflow. |
| `scripts/` | Python entry points for the five pipeline stages plus shared utilities and model definition. |
| `inputs/input.yaml` | Workflow inputs (dataset URL, etc.). |
| `streamflow_local.yml` | StreamFlow configuration for running locally with Docker/Podman. |

Generated artifacts (datasets, models, reports) land under `artifacts/`.

Local runtime state (virtualenv, StreamFlow state DB, caches) is ignored via `.gitignore`.

## Prerequisites

- Linux host with Docker.
- Python 3.8+ with StreamFlow installed (`pip install streamflow`).
- Network access to download CIFAR-10 and Python packages inside containers.

This workflow was tested on Ubuntu 24.04 with Docker from the Ubuntu repository and Python 3.11 in a virtual environment created with `uv`.

## Quick start

1. Create and activate a Python virtual environment (optional but recommended).
2. Install StreamFlow:
   ```bash
   pip install streamflow
   ```
   3. Kick off the workflow:
      ```bash
      streamflow run --outdir artifacts/final streamflow_local.yml
      ```
      or optionally include logs:
      ```bash
      set -o pipefail
      streamflow run --outdir artifacts/final streamflow_local.yml 2>&1 | tee streamflow-run.log
      ```
4. Results are written to:
   - `artifacts/models/model.pt` – trained PyTorch checkpoint.
   - `artifacts/results/metrics.json` – evaluation metrics.
   - `artifacts/results/report.pdf` – generated PDF report with plots.

   StreamFlow also copies the declared CWL outputs into `artifacts/final/`.

## GPU notes

- The local config is **CPU-default** to work on machines without NVIDIA Container Toolkit.
- To enable GPU execution, you need NVIDIA Container Toolkit configured for your runtime (Docker or Podman), then add back a `gpus: all` block (and optional `NVIDIA_*` env vars) under the `docker-pytorch` model in `streamflow_local.yml`.

