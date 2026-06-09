# CellShape3D-VAE
### Unsupervised 3D Cell Shape Representation Learning via Variational Autoencoder

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Saggy7276/cellshape3d-vae/blob/main/CellShape3D_VAE.ipynb)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

An end-to-end pipeline for **3D microscopy cell shape analysis** combining classical morphological descriptors with a deep **3D Variational Autoencoder (VAE)** for unsupervised representation learning.

Morphological changes in cell nuclei are early biomarkers for diseases including cancer and cardiovascular conditions. This project demonstrates how unsupervised shape representation learning can reveal latent morphological structure in 3D fluorescence microscopy data — **without requiring any manual labels**.

---

## Pipeline

```
Raw 3D fluorescence volumes (.tif)
         │
         ▼
Preprocessing & instance extraction
(bounding box crop → resize to 32×64×64)
         │
         ▼
3D Shape Descriptors                     3D-VAE Training
- Volume, surface area                   - Conv3D encoder
- Sphericity, elongation, flatness       - 32-dim latent space
- Solidity, bounding box ratio           - Conv3DTranspose decoder
         │                               - β-VAE loss (BCE + β·KL)
         └──────────────┬────────────────┘
                        ▼
         Latent Space Analysis
         - UMAP 2D embedding
         - K-Means phenotyping (unsupervised)
         - Shape descriptor correlation
         - Latent space traversal (shape interpolation)
```

---

## Dataset

**Cell Tracking Challenge — `Fluo-N3DH-CHO`**
- 3D confocal fluorescence microscopy of Chinese Hamster Ovary (CHO) cell nuclei
- Gold-standard instance segmentation ground truth masks
- Download: http://celltrackingchallenge.net/3d-cell-tracking/

A **synthetic fallback** (no download needed) is included in the notebook for immediate end-to-end execution on Colab free tier.

---

## Results

| Metric | Value |
|---|---|
| VAE latent dimensions | 32 |
| Reconstruction (BCE) | converges in ~80 epochs |
| Clustering ARI (unsupervised) | ≥ 0.65 on synthetic, dataset-dependent on real CTC |
| Silhouette score | > 0.35 |

**Key outputs:**

| Figure | Description |
|---|---|
| `fig1_sample_crops.png` | Max-intensity projections of input 3D cell volumes |
| `fig2_shape_descriptors.png` | Classical shape descriptor scatter + distributions |
| `fig3_training_curves.png` | VAE total loss, BCE, KL divergence over epochs |
| `fig4_latent_space.png` | UMAP of latent space, coloured by shape class + descriptors |
| `fig5_reconstructions.png` | Input vs VAE reconstruction side-by-side |
| `fig6_latent_traversal.png` | Smooth shape interpolation through latent space |
| `fig7_clustering.png` | K-Means phenotyping vs ground truth labels |
| `fig8_cluster_profiles.png` | Morphological descriptor heatmap per cluster |

---

## Architecture

```
Encoder3D
  Conv3D(1→16, stride=2)  + GroupNorm + LeakyReLU   # 32×64×64 → 16×32×32
  Conv3D(16→32, stride=2) + GroupNorm + LeakyReLU   # → 32×16×16
  Conv3D(32→64, stride=2) + GroupNorm + LeakyReLU   # → 64×8×8
  Conv3D(64→128,stride=2) + GroupNorm + LeakyReLU   # → 128×4×4
  Linear → μ (32-dim), log σ² (32-dim)

Decoder3D  (symmetric, ConvTranspose3D)
  Linear(32) → reshape(128, 2, 4, 4)
  ConvTranspose3D ×4  → (1, 32, 64, 64)
  Sigmoid output

Loss: L = BCE(recon, x) + β × KL(N(μ,σ²) || N(0,I))
      β = 0.5   (β-VAE for mild disentanglement)
```

---

## Quickstart

**Option A — Google Colab (recommended, no setup)**

Click the badge above or open:
```
https://colab.research.google.com/drive/104sCtTZ3gL_Z1zd7W586SYcWcT0hy_Ml#scrollTo=JdCZAXHaZgu4
```
Set runtime to **T4 GPU** → Run All. The synthetic data path runs without any download.

**Option B — Local**

```bash
git clone https://github.com/Saggy7276/cellshape3d-vae
cd cellshape3d-vae
pip install -r requirements.txt
jupyter notebook CellShape3D_VAE.ipynb
```

---

## Requirements

```
torch>=2.0
torchvision
tifffile
scikit-image
scipy
umap-learn
matplotlib
seaborn
pandas
scikit-learn
tqdm
```

---

## Relevance to Biomedical Image Analysis

This project directly addresses core challenges in precision medicine imaging:

- **Scalable 3D pipelines** — handles full volumetric stacks from confocal and light-sheet microscopy
- **Label-free phenotyping** — VAE discovers morphological groups without annotation cost
- **Quantitative shape modeling** — classical descriptors (sphericity, elongation, solidity) provide interpretable biological features alongside the deep representation
- **Latent space biology** — smooth interpolation between cell shapes enables in-silico exploration of morphological variation

---

## Author

**Sagar** | M.Sc. Smart Systems, Hochschule Furtwangen University (2024)  
Computer Vision · Deep Learning · Biomedical Image Analysis  
[github.com/Saggy7276](https://github.com/Saggy7276)

---

## License

MIT License — see [LICENSE](LICENSE) for details.# cellshape3d-vae
