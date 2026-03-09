cwlVersion: v1.2
class: Workflow

inputs:
  dataset_url: string

steps:
  download:
    run: download.cwl
    in:
      dataset_url: dataset_url
    out: [raw_data]

  preprocess:
    run: preprocess.cwl
    in:
      raw_data: download/raw_data
    out: [train_data, test_data]

  train:
    run: train.cwl
    in:
      train_data: preprocess/train_data
    out: [model_file]

  evaluate:
    run: evaluate.cwl
    in:
      model_file: train/model_file
      test_data: preprocess/test_data
    out: [metrics]

  report:
    run: report.cwl
    in:
      metrics: evaluate/metrics
    out: [report]

outputs:
  final_model:
    type: File
    outputSource: train/model_file
  final_report:
    type: File
    outputSource: report/report

