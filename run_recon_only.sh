#!/usr/bin/env bash
# =============================================================================
# Reconstruction만 수행. config 없이 Inference/recon + Inference/ckpt 사용.
# 결과: Inference/results/<RUN_ID>/recon
# =============================================================================
# 사용법: ./run_recon_only.sh [RUN_ID]
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_ID="${1:-A}"
RECON_OUTPUT="$SCRIPT_DIR/results/$RUN_ID/recon"

if [[ -d "$SCRIPT_DIR/../datasets/BCI_HER2" ]]; then
  INPUT_DATAROOT="$(cd "$SCRIPT_DIR/../datasets/BCI_HER2" && pwd)"
elif [[ -d "$SCRIPT_DIR/datasets/BCI_HER2" ]]; then
  INPUT_DATAROOT="$(cd "$SCRIPT_DIR/datasets/BCI_HER2" && pwd)"
else
  echo "Error: datasets/BCI_HER2 not found."
  exit 1
fi

echo "Reconstruction only (run_id=$RUN_ID) -> $RECON_OUTPUT"
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

echo "Recon results (fake_B): $RECON_OUTPUT"
