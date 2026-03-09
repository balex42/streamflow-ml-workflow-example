cwlVersion: v1.2
class: CommandLineTool
baseCommand: ["python", "/workspace/ml_workflow/scripts/report.py"]
requirements:
  DockerRequirement:
    dockerPull: python:3.11
    dockerOutputDirectory: /workspace/ml_workflow
inputs:
  metrics:
    type: File
    inputBinding:
      position: 1
outputs:
  report:
    type: File
    outputBinding:
      glob: "artifacts/results/report.pdf"

