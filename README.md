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

### Sample data (try without full download)

The repo includes a **sample dataset** and **sample reconstruction results** so you can run the pipeline without downloading the full BCI dataset.

- **`datasets/BCI_HER2_sample/`** — small sample with `valA` and `valB` (HE and IHC pairs). Use this to run reconstruction and classification locally.
- **`results/reconstruction/`** — pre-run reconstruction outputs (fake_B images) on the sample. With this and the sample data, you can run **classification only** without running reconstruction.

You can try the full pipeline (recon → classification) on the sample, or run classification directly using the provided sample + sample reconstruction results.

### Full dataset (paper reproduction)

For **reproducing the paper results**, download the full **BCI dataset** and place test images as below.

- **Dataset**: [BCI – Breast Cancer Immunohistochemical Image Generation Challenge](https://bupt-ai-cz.github.io/BCI/)
- Organize so that:
  - **valA**: source domain (HE)
  - **valB**: target domain (IHC)
- Layout:

  ```
  workspace/
  └── datasets/
      ├── BCI_HER2_sample/   # included: sample valA, valB
      └── BCI_HER2/          # full data (download): valA, valB
  ```

Use `--dataroot datasets/BCI_HER2_sample` for the sample, or `--dataroot datasets/BCI_HER2` for the full data.

---

## 2. Checkpoints

Pre-trained weights are provided separately.

- **Download**: [Checkpoints](https://1drv.ms/u/c/de011cb09ae2716d/IQBjTzKSq7_6QYqOKT-3Qm0BAZk3tV5gbgPV9dCA7qE2ZPQ?e=c372wF)
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
- **Run** (from `workspace`). For the sample dataset use `--dataroot datasets/BCI_HER2_sample`; for the full dataset use `--dataroot datasets/BCI_HER2`:

  ```bash
  python recon/test.py --dataroot datasets/BCI_HER2_sample --name reconstruction --checkpoints_dir ckpt --results_dir results/reconstruction --model cpt --CUT_mode CUT --gpu_ids 0 --netD n_layers --ndf 32 --netG resnet_9blocks --n_layers_D 5 --normG instance --normD instance --weight_norm spectral --lambda_GAN 1.0 --lambda_NCE 10.0 --nce_layers 0,4,8,12,16 --nce_T 0.07 --num_patches 256 --lambda_style 100.0 --lambda_content 1.0 --lambda_gp 10.0 --gp_weights '[0.015625,0.03125,0.0625,0.125,0.25,1.0]' --lambda_asp 10.0 --asp_loss_mode lambda_linear --use_simsiam True --use_clsA 0 --use_clsB 1 --use_clsfB 1 --lambda_cls 10.0 --cls_content False --lambda_discls 0.1 --dataset_mode aligned --direction AtoB --num_threads 2 --batch_size 1 --load_size 1024 --crop_size 1024 --preprocess crop --flip_equivariance False --display_winsize 512 --phase val --num_test 10000 --epoch latest
  ```

### Classification

- **Role**: Classify images (e.g. by HER2 score) using a pre-trained classifier.
- **Input**: Either a single data root with `valA`/`predB`-style subfolders, or separate roots for A and B via `--data_root_A` and `--data_root_B`.
- **Output**: Predictions and metrics under `--output_dir` (e.g. `predictions.json`, `metrics.json`, confusion matrix plot).

**Using sample data (no download):** With the included `datasets/BCI_HER2_sample` and `results/reconstruction`, you can run classification without downloading the full dataset:

  ```bash
  python classification/test.py --ckpt_dir ckpt/classification --output_dir results/classification --data_root_A datasets/BCI_HER2_sample/valA --data_root_B results/reconstruction --mode AB --fold all
  ```

**Using full data (paper reproduction):** For numbers reported in the paper, use the full test set:

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

- **Included in the repo:** `results/reconstruction/` contains sample reconstruction results on `BCI_HER2_sample`, so you can run classification without running reconstruction first.
- Use a different run name by changing the result folder (e.g. `results/A/recon`) to avoid overwriting previous runs.
- Reconstruction writes **only** the generated images (fake_B) into the given `--results_dir`, with no extra subfolders.

**Reproducibility:** Results in the paper are reported on the full test set. To reproduce them, download the full BCI test data and run reconstruction and classification on `datasets/BCI_HER2` (see §1).