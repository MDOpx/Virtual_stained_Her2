# Inference Pipeline (Reconstruction + Classification)

Run reconstruction and classification. 

---

## 0. Environment setup

### Docker (recommended)

Use a Python image with CUDA support (e.g. PyTorch base). Example:

```bash
# Example: mount project and datasets, then run inside container
docker run -it --gpus all -v /path/to/workspace:/workspace pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime bash
cd /workspace/Inference
pip install -r requirements.txt
```

### Local / pip

From the `workspace` directory:

```bash
cd workspace
pip install -r requirements.txt
```
## 1. Data Setup

Download the **BCI dataset** and place test images under the expected layout.

- **Dataset**: [BCI – Breast Cancer Immunohistochemical Image Generation Challenge](https://bupt-ai-cz.github.io/BCI/)
- After downloading, organize test data so that:
  - **Test images** are in two folders:
    - **valA**: source domain (data from HE/test)
    - **valB**: target domain (data from IHC/test)
- Recommended layout under the project:

  ```
  workspace/
  └── datasets/
      └── BCI_HER2/
          ├── valA/   # source images for inference
          └── valB/   # target images (optional for recon-only)
  ```

  Use `--dataroot datasets/BCI_HER2` (or `../datasets/BCI_HER2` if running from a different cwd) when running the scripts.

---

## 2. Checkpoints

Pre-trained weights are provided separately.

- **Download**: [Checkpoints](#)
- **Steps**:
  1. Download the checkpoint archive.
  2. Unzip it.
  3. Place the extracted **`ckpt`** folder inside the main workspace directory.

- **Expected layout**:

  ```
  workspace/
  └── ckpt/
      ├── reconstruction/   
      └── classification/ 
  ```

**Required for inference**

| Step | Required files |
|------|-----------------|
| **Reconstruction** | `ckpt/reconstruction/latest_net_G.pth` |
| **Classification** | `ckpt/classification/fold_<i>/checkpoints/best.pth` |

---

## 3. Reconstruction and Classification

### Reconstruction

- **Role**: Image-to-image translation (e.g. HE → IHC) using a CPT (Contrastive Paired Translation) model.
- **Input**: Pairs from `valA` and `valB` under the dataroot.
- **Output**: Generated images (fake_B) only, written as PNGs under the directory given by `--results_dir`.
- **Run** (from `workspace`):

  ```bash
  python recon/test.py --dataroot datasets/BCI_HER2 --name reconstruction --checkpoints_dir ckpt --results_dir results/reconstruction --model cpt --CUT_mode CUT --gpu_ids 0 --netD n_layers --ndf 32 --netG resnet_9blocks --n_layers_D 5 --normG instance --normD instance --weight_norm spectral --lambda_GAN 1.0 --lambda_NCE 10.0 --nce_layers 0,4,8,12,16 --nce_T 0.07 --num_patches 256 --lambda_style 100.0 --lambda_content 1.0 --lambda_gp 10.0 --gp_weights '[0.015625,0.03125,0.0625,0.125,0.25,1.0]' --lambda_asp 10.0 --asp_loss_mode lambda_linear --use_simsiam True --use_clsA 0 --use_clsB 1 --use_clsfB 1 --lambda_cls 10.0 --cls_content False --lambda_discls 0.1 --dataset_mode aligned --direction AtoB --num_threads 2 --batch_size 1 --load_size 1024 --crop_size 1024 --preprocess crop --flip_equivariance False --display_winsize 512 --phase val --num_test 10000 --epoch latest
  ```

### Classification

- **Role**: Classify images (e.g. by HER2 score) using a pre-trained classifier.
- **Input**: Either a single data root with `valA`/`predB`-style subfolders, or separate roots for A and B via `--data_root_A` and `--data_root_B`.
- **Output**: Predictions and metrics under `--output_dir` (e.g. `predictions.json`, `metrics.json`, confusion matrix plot).
- **Run** (from `workspace`), using recon output as B:

  ```bash
  python classification/test.py --ckpt_dir ckpt/classification --output_dir results/classification --data_root_A datasets/BCI_HER2/valA --data_root_B results/reconstruction --mode AB --fold all
  ```

---

## 4. Results

All outputs are written under the paths you pass to the scripts (defaults below).

| Step            | Output directory (example)   | Contents |
|----------------|-----------------------------|----------|
| Reconstruction | `results/reconstruction/`   | `*.png` (fake_B images only), one per input. |
| Classification | `results/classification/`  | `predictions.json`, `metrics.json`, `confusion_matrix.png`, and per-sample visualizations (e.g. `*_test_*.png`). |

- Use a different run name by changing the result folder (e.g. `results/A/recon`, `results/B/recon`) to avoid overwriting previous runs.
- Reconstruction writes **only** the generated images (fake_B) into the given `--results_dir`, with no extra subfolders.

---

## Directory layout (reference)

```
workspace/
├── recon/                  # Reconstruction (CPT) code
├── classification/         # Classification code
├── ckpt/
│   ├── reconstruction/     # Recon weights (e.g. latest_net_G.pth)
│   └── classification/     # Per-fold checkpoints and args (e.g. fold_0/...)
├── prepare_recon_dataset.py
├── README.md
└── results/                # Created at run time
    ├── reconstruction/     # Recon output (fake_B *.png)
    └── classification/     # Classification outputs
```

---

## Repository (Virtual_stained_Her2)

This inference code is intended to be synced with [**Virtual_stained_Her2**](https://github.com/MDOpx/Virtual_stained_Her2).

- **Upload only the contents inside this folder** — do not create an `Inference/` directory in the repo. The repo root should be `recon/`, `classification/`, `README.md`, etc. directly.
- **Included**: `recon/`, `classification/`, `prepare_recon_dataset.py`, `requirements.txt`, shell scripts, `.gitignore`, and this README.
- **Excluded** (see `.gitignore`): `ckpt/`, `datasets/`, `models_train/`, `results/`, `__pycache__/`.

To publish (from a machine that has this Inference folder):

```bash
git clone https://github.com/MDOpx/Virtual_stained_Her2.git
cd Virtual_stained_Her2

# Copy only the contents of Inference into repo root (no Inference folder)
rsync -av --exclude='ckpt' --exclude='datasets' --exclude='models_train' --exclude='results' --exclude='__pycache__' \
  /path/to/Inference/ .

git add .
git commit -m "Inference code"
git push
```

Replace `/path/to/Inference` with the actual path. Then add `ckpt` and `datasets` locally (see sections 1–2 above); they are not pushed.
