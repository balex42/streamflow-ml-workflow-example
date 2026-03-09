cwlVersion: v1.2
class: CommandLineTool
baseCommand: ["python", "/workspace/ml_workflow/scripts/evaluate.py"]
requirements:
  DockerRequirement:
    dockerPull: pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel
    dockerOutputDirectory: /workspace/ml_workflow
inputs:
  model_file:
    type: File
    inputBinding:
      position: 1
  test_data:
    type: File
    inputBinding:
      position: 2
outputs:
  metrics:
    type: File
    outputBinding:
      glob: "artifacts/results/metrics.json"

