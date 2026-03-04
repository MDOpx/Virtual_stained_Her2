#!/usr/bin/env bash
# =============================================================================
# Inference 전용: Recon -> Classification. config 없이 Inference 폴더만으로 동작.
# - Recon: Inference/recon, 체크포인트 Inference/ckpt
# - Classification: Inference/classification, 체크포인트 Inference/ckpt/classification
# - 결과: Inference/results/<RUN_ID>/
# =============================================================================
# 사용법: ./run_inference.sh [RUN_ID]
#   RUN_ID 생략 시 A. 데이터는 자동 탐색: ../datasets/BCI_HER2 또는 ./datasets/BCI_HER2
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_ID="${1:-A}"
RESULTS_BASE="$SCRIPT_DIR/results"
OUTPUT_BASE="$RESULTS_BASE/$RUN_ID"
RECON_OUTPUT="$OUTPUT_BASE/recon"
RECON_FAKE_B="$RECON_OUTPUT"
RECON_DATASET="$OUTPUT_BASE/recon_dataset"
CLS_OUTPUT="$OUTPUT_BASE/classification"

# 데이터 경로 자동 탐색 (입력 받지 않음)
if [[ -d "$SCRIPT_DIR/../datasets/BCI_HER2" ]]; then
  INPUT_DATAROOT="$(cd "$SCRIPT_DIR/../datasets/BCI_HER2" && pwd)"
elif [[ -d "$SCRIPT_DIR/datasets/BCI_HER2" ]]; then
  INPUT_DATAROOT="$(cd "$SCRIPT_DIR/datasets/BCI_HER2" && pwd)"
else
  echo "Error: datasets/BCI_HER2 not found. Put data in Inference/../datasets/BCI_HER2 or Inference/datasets/BCI_HER2"
  exit 1
fi
echo "Using dataroot: $INPUT_DATAROOT"

# -----------------------------------------------------------------------------
# 1. Reconstruction (Inference/recon)
# -----------------------------------------------------------------------------
echo "[1/4] Running reconstruction..."
mkdir -p "$RECON_OUTPUT"
python recon/test.py \
  --dataroot "$INPUT_DATAROOT" \
  --name reconstruction \
  --checkpoints_dir "$SCRIPT_DIR/ckpt" \
  --results_dir "$RECON_OUTPUT" \
  --model cpt \
  --CUT_mode CUT \
  --gpu_ids 0 \
  --netD n_layers \
  --ndf 32 \
  --netG resnet_9blocks \
  --n_layers_D 5 \
  --normG instance \
  --normD instance \
  --weight_norm spectral \
  --lambda_GAN 1.0 \
  --lambda_NCE 10.0 \
  --nce_layers 0,4,8,12,16 \
  --nce_T 0.07 \
  --num_patches 256 \
  --lambda_style 100.0 \
  --lambda_content 1.0 \
  --lambda_gp 10.0 \
  --gp_weights '[0.015625,0.03125,0.0625,0.125,0.25,1.0]' \
  --lambda_asp 10.0 \
  --asp_loss_mode lambda_linear \
  --use_simsiam True \
  --use_clsA 0 \
  --use_clsB 1 \
  --use_clsfB 1 \
  --lambda_cls 10.0 \
  --cls_content False \
  --lambda_discls 0.1 \
  --dataset_mode aligned \
  --direction AtoB \
  --num_threads 2 \
  --batch_size 1 \
  --load_size 1024 \
  --crop_size 1024 \
  --preprocess crop \
  --flip_equivariance False \
  --display_winsize 512 \
  --phase val \
  --num_test 10000 \
  --epoch latest

# fake_B PNG들이 results_dir 직하위에 저장됨
if ! ls "$RECON_FAKE_B"/*.png 1>/dev/null 2>&1; then
  echo "Error: No reconstruction output PNGs in $RECON_FAKE_B"
  exit 1
fi

# -----------------------------------------------------------------------------
# 2. Prepare classification dataset (valA + predB)
# -----------------------------------------------------------------------------
echo "[2/4] Preparing classification dataset (valA + predB)..."
mkdir -p "$RECON_DATASET/valA" "$RECON_DATASET/predB"
VALA_SRC="$INPUT_DATAROOT/valA"
[[ ! -d "$VALA_SRC" ]] && VALA_SRC="$INPUT_DATAROOT/trainA"
if [[ ! -d "$VALA_SRC" ]]; then
  echo "Error: valA/trainA not found under $INPUT_DATAROOT"
  exit 1
fi
for f in "$VALA_SRC"/*; do
  [[ -e "$f" ]] || continue
  ln -sf "$(realpath "$f")" "$RECON_DATASET/valA/$(basename "$f")" 2>/dev/null || cp -n "$f" "$RECON_DATASET/valA/$(basename "$f")"
done
for f in "$RECON_FAKE_B"/*.png; do
  [[ -e "$f" ]] || continue
  ln -sf "$(realpath "$f")" "$RECON_DATASET/predB/$(basename "$f")" 2>/dev/null || cp -n "$f" "$RECON_DATASET/predB/$(basename "$f")"
done

# -----------------------------------------------------------------------------
# 3. Classification (Inference/classification)
# -----------------------------------------------------------------------------
echo "[3/4] Running classification..."
mkdir -p "$CLS_OUTPUT"
python classification/test.py \
  --ckpt_dir "$SCRIPT_DIR/ckpt/classification" \
  --output_dir "$CLS_OUTPUT" \
  --data_root "$RECON_DATASET" \
  --mode AB \
  --is_pred \
  --fold all

echo "[4/4] Done."
echo "  Recon (fake_B): $RECON_OUTPUT"
echo "  Classification: $CLS_OUTPUT"
