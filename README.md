# ThinkCXR

ThinkCXR is a reproducible training and evaluation package for structured chest X-Ray report generation with an explicit reasoning stage. The model receives an X-Ray image and generates a response with the following structure:

```text
<think>
[image-grounded reasoning]
</think>
<answer>
### FINDINGS:
[objective observations]
### IMPRESSIONS:
[summary]
</answer>
```

This repository contains the data-format conversion, LoRA supervised fine-tuning entry point, batch inference command, and field-level evaluation utilities. It does not redistribute clinical data, model weights, generated predictions, or private annotations.

## Repository Layout

```text
ThinkCXR/
├── README.md
├── requirements.txt
├── scripts/
│   ├── prepare_dataset.py
│   ├── train_lora.sh
│   ├── run_inference.sh
│   └── evaluate_reports.py
└── examples/
    └── sample.csv
```

## Data Format

The conversion script expects a CSV with these columns:

```text
image_path,CoT,Finding,Impression
```

`image_path` may be absolute or relative to `--dataset-root`. The image must be readable by the training framework. Findings and Impression must be non-empty. `CoT` contains a quality-controlled reasoning annotation for SFT. The converter writes a ShareGPT-style JSON file and a dataset registry file.

The training and inference prompt is intentionally fixed across both stages. Use `--require-cot` when preparing training data. For external evaluation data without verified reasoning annotations, omit `--require-cot`; the converter retains an empty `<think>` block and generated CoT is not scored.

## Installation

Create an environment containing the model-training toolkit and the packages listed in `requirements.txt`. The exact CUDA and PyTorch versions should be selected for the target GPU and model implementation.

```bash
python -m pip install -r requirements.txt
```

The public repository provides the complete SFT invocation in `scripts/train_lora.sh`; install a compatible training environment before running it. The launcher path can be overridden through `TRAINER_BIN`.

## Prepare A Dataset

```bash
python scripts/prepare_dataset.py \
  --csv /path/to/metadata.csv \
  --dataset-root /path/to/dataset \
  --output-root /path/to/dataset \
  --dataset-name thinkcxr_train \
  --require-cot
```

The command creates:

```text
/path/to/dataset/thinkcxr_train.json
/path/to/dataset/dataset_info.json
```

For a held-out evaluation set, use the same conversion script with its own CSV and dataset name. Keep training, validation, and test cases disjoint at the patient or study level.

## LoRA SFT

The public training entry point is a shell script with configurable variables:

```bash
MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct \
DATASET_DIR=/path/to/dataset \
DATASET_NAME=thinkcxr_train \
OUTPUT_DIR=/path/to/output \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/train_lora.sh
```


## Batch Inference

```bash
MODEL_PATH=/path/to/finetuned/model \
DATASET_DIR=/path/to/test_dataset \
DATASET_NAME=thinkcxr_test \
OUTPUT_DIR=/path/to/predictions \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_inference.sh
```

The test input is the X-Ray image only. Findings and Impression are reference targets, not inference inputs. The output should contain `<think>` followed by `<answer>` with the two required report sections.

## Evaluation

```bash
python scripts/evaluate_reports.py \
  --prediction /path/to/predictions/generated_predictions.jsonl
```

The prediction JSONL must contain `label` and `predict` fields. The evaluator reports ROUGE-1, ROUGE-2, ROUGE-L, BLEU-1, BLEU-2, and BLEU-4 separately for Findings and Impression.


## License And Data Access

This code is released for research reproducibility. Users must obtain and use the underlying medical datasets according to their original licenses, access controls, de-identification requirements, and institutional policies. No patient data are included in this repository.
