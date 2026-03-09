cwlVersion: v1.2
class: CommandLineTool
baseCommand: ["python", "/workspace/ml_workflow/scripts/preprocess.py"]
requirements:
  DockerRequirement:
    dockerPull: python:3.11
    dockerOutputDirectory: /workspace/ml_workflow
inputs:
  raw_data:
    type: File
    inputBinding:
      position: 1
outputs:
  train_data:
    type: File
    outputBinding:
      glob: "artifacts/data/train.npz"
  test_data:
    type: File
    outputBinding:
      glob: "artifacts/data/test.npz"

