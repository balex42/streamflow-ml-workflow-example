cwlVersion: v1.2
class: CommandLineTool
baseCommand: ["python", "/workspace/ml_workflow/scripts/download.py"]
requirements:
  DockerRequirement:
    dockerPull: python:3.11
    dockerOutputDirectory: /workspace/ml_workflow
inputs:
  dataset_url:
    type: string
    inputBinding:
      position: 1
outputs:
  raw_data:
    type: File
    outputBinding:
      glob: "artifacts/data/raw_data.npz"

