#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the base or fine-tuned model}"
: "${DATASET_DIR:?Set DATASET_DIR to the evaluation dataset}"
: "${DATASET_NAME:?Set DATASET_NAME to a registered evaluation dataset}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for predictions}"
TRAINER_BIN="${TRAINER_BIN:-llamafactory-cli}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

"${TRAINER_BIN}" train \
  --stage sft \
  --model_name_or_path "${MODEL_PATH}" \
  --finetuning_type "${FINETUNING_TYPE:-lora}" \
  --template "${TEMPLATE:-qwen2_vl}" \
  --dataset_dir "${DATASET_DIR}" \
  --eval_dataset "${DATASET_NAME}" \
  --cutoff_len "${CUTOFF_LEN:-8192}" \
  --max_samples "${MAX_SAMPLES:-100000}" \
  --per_device_eval_batch_size "${EVAL_BATCH_SIZE:-2}" \
  --predict_with_generate True \
  --max_new_tokens "${MAX_NEW_TOKENS:-4096}" \
  --do_predict True \
  --report_to none \
  --output_dir "${OUTPUT_DIR}" \
  --trust_remote_code True \
  --seed "${SEED:-42}"
