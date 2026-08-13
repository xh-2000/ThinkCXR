#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the base vision-language model}"
: "${DATASET_DIR:?Set DATASET_DIR to the directory containing dataset_info.json}"
: "${DATASET_NAME:?Set DATASET_NAME to a registered dataset name}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to the LoRA output directory}"

TRAINER_BIN="${TRAINER_BIN:-swift}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

"${TRAINER_BIN}" sft \
  --model "${MODEL_PATH}" \
  --dataset "${DATASET_DIR}/${DATASET_NAME}.json" \
  --train_type lora \
  --torch_dtype bfloat16 \
  --max_length "${CUTOFF_LEN:-8192}" \
  --dataset_num_proc "${PREPROCESSING_WORKERS:-16}" \
  --per_device_train_batch_size "${TRAIN_BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRAD_ACC_STEPS:-8}" \
  --learning_rate "${LEARNING_RATE:-5.0e-5}" \
  --num_train_epochs "${EPOCHS:-3.0}" \
  --lr_scheduler_type cosine \
  --target_modules all-linear \
  --freeze_vit true \
  --freeze_aligner true \
  --lora_rank "${LORA_RANK:-8}" \
  --lora_alpha "${LORA_ALPHA:-16}" \
  --lora_dropout "${LORA_DROPOUT:-0}" \
  --logging_steps "${LOGGING_STEPS:-5}" \
  --save_steps "${SAVE_STEPS:-100}" \
  --output_dir "${OUTPUT_DIR}" \
  --seed "${SEED:-42}" \
  --report_to none
