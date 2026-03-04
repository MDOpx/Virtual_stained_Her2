#!/usr/bin/env bash
# =============================================================================
# Classification만 수행. config 없이 Inference/classification + ckpt/classification 사용.
# - 이미 recon을 돌린 경우: results/<RUN_ID>/recon_dataset (valA+predB) 사용
# - 아니면 자동 탐색한 데이터 루트의 valA/valB 사용
# =============================================================================
# 사용법: ./run_cls_only.sh [RUN_ID]
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_ID="${1:-A}"
OUTPUT_BASE="$SCRIPT_DIR/results/$RUN_ID"
CLS_OUTPUT="$OUTPUT_BASE/classification"
RECON_DATASET="$OUTPUT_BASE/recon_dataset"

if [[ -d "$RECON_DATASET/valA" && -d "$RECON_DATASET/predB" ]]; then
  DATA_ROOT="$RECON_DATASET"
  USE_PRED="--is_pred"
  echo "Using recon_dataset: $DATA_ROOT (valA + predB)"
else
  if [[ -d "$SCRIPT_DIR/../datasets/BCI_HER2" ]]; then
    DATA_ROOT="$(cd "$SCRIPT_DIR/../datasets/BCI_HER2" && pwd)"
  elif [[ -d "$SCRIPT_DIR/datasets/BCI_HER2" ]]; then
    DATA_ROOT="$(cd "$SCRIPT_DIR/datasets/BCI_HER2" && pwd)"
  else
    echo "Error: No data found. Run recon first or put datasets/BCI_HER2."
    exit 1
  fi
  USE_PRED=""
  echo "Using data_root: $DATA_ROOT"
fi

mkdir -p "$CLS_OUTPUT"
python classification/test.py \
  --ckpt_dir "$SCRIPT_DIR/ckpt/classification" \
  --output_dir "$CLS_OUTPUT" \
  --data_root "$DATA_ROOT" \
  --mode AB \
  $USE_PRED \
  --fold all

echo "Classification results: $CLS_OUTPUT"
