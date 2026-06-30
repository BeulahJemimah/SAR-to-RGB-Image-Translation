# SAR-to-RGB Image Translation using Pix2Pix GAN

## Overview

This project implements a Pix2Pix Generative Adversarial Network (GAN) to translate Sentinel-1 Synthetic Aperture Radar (SAR) images into Sentinel-2 RGB optical images using the SEN1-2 dataset.

The objective is to generate realistic RGB images from SAR inputs, enabling improved visualization and interpretation of remote sensing imagery where optical images may be unavailable due to cloud cover or adverse weather conditions.

This project was developed as part of the GalaxEye Space Solutions technical assessment.

---

## Dataset

**Dataset:** SEN1-2 Dataset

- Source: https://mediatum.ub.tum.de/1436631
- Season Used: Summer
- Image Size: 256 × 256 pixels
- Input: Sentinel-1 SAR Images (Grayscale)
- Target: Sentinel-2 RGB Images

---

## Model Architecture

The project uses the Pix2Pix conditional GAN architecture consisting of:

- U-Net based Generator
- PatchGAN Discriminator

### Generator
- Encoder-Decoder architecture with skip connections.
- Generates RGB images from SAR inputs.

### Discriminator
- PatchGAN classifier.
- Distinguishes between real and generated SAR-RGB image pairs.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Framework | PyTorch |
| Image Size | 256 × 256 |
| Batch Size | 16 |
| Epochs | 50 |
| Learning Rate | 0.0002 |
| Optimizer | Adam |
| Loss Function | GAN Loss + L1 Loss |
| Lambda L1 | 100 |

---

## Repository Structure

```
SAR-to-RGB-Pix2Pix-GalaxEye/

├── notebook/
│   └── assessment1-galaxyeye.ipynb
│
├── configs/
│   └── config.json
│
├── results/
│
├── checkpoints/
│
├── infer.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/SAR-to-RGB-Pix2Pix-GalaxEye.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

Run the Jupyter notebook to:

- Load the SEN1-2 dataset
- Preprocess SAR-RGB image pairs
- Train the Pix2Pix GAN
- Save the trained model

---

## Inference

The trained Generator can be used to convert unseen SAR images into RGB images.

Example:

```python
python infer.py
```

---

## Results

The model successfully learns to translate Sentinel-1 SAR images into corresponding Sentinel-2 RGB images using the Pix2Pix framework.

Example outputs include:

- Input SAR Image
- Generated RGB Image
- Ground Truth RGB Image

---

## Model Weights

The trained model weights can be downloaded from:

**Google Drive:** *(Add your Google Drive link here)*

---

## Requirements

- Python 3.12
- PyTorch
- Torchvision
- NumPy
- Pillow
- Matplotlib
- Scikit-learn
- tqdm

---

## Future Improvements

- Evaluate using PSNR, SSIM, LPIPS, and FID metrics.
- Train on all seasons of the SEN1-2 dataset.
- Experiment with attention-based GAN architectures.
- Improve inference speed and image quality.

---

## Author

**Beulah Jemimah**

GalaxEye Space Solutions – Technical Assessment


