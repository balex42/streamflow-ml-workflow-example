cwlVersion: v1.2
class: CommandLineTool
baseCommand: ["python", "/workspace/ml_workflow/scripts/train.py"]
requirements:
  DockerRequirement:
    dockerPull: pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel
    dockerOutputDirectory: /workspace/ml_workflow
inputs:
  train_data:
    type: File
    inputBinding:
      position: 1
outputs:
  model_file:
    type: File
    outputBinding:
      glob: "artifacts/models/model.pt"

