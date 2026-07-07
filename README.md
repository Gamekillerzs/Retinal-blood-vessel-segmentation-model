# Retinal Blood Vessel Segmentation — MSFAUMobileNet

A deep learning model for automated segmentation of blood vessels in retinal fundus images, built around a proposed **MSFAUMobileNet** architecture (Multi-Scale Feature Attention U-Net with a MobileNet-based backbone).

## Overview

Accurate segmentation of retinal blood vessels is a key step in the early diagnosis of diseases such as diabetic retinopathy, hypertension, and glaucoma. This project implements a lightweight, attention-augmented encoder-decoder network built on top of a pretrained **MobileNetV2** backbone.

### Architecture

- **Encoder:** A frozen (non-trainable) `MobileNetV2` pretrained on ImageNet, used as a feature extractor. Feature maps are tapped at four resolutions (`block_1_expand_relu`, `block_3_expand_relu`, `block_6_expand_relu`, `block_13_expand_relu`), plus the final bottleneck (`block_16_project`).
- **MSFA (Multi-Scale Feature Aggregation) block:** Applies four parallel convolutions (1×1, 3×3, 5×5, 7×7) to the input, then concatenates them so the combined output matches the target channel count. Used both at the bottleneck and on each skip connection before merging with the decoder.
- **Channel Attention block:** Combines global average pooling and global max pooling, concatenates them, passes through a 1×1 convolution and sigmoid, and uses the result to rescale the feature map channel-wise — helping the network emphasize thin, low-contrast vessels.
- **Decoder block:** Upsamples via `Conv2DTranspose`, applies channel attention, concatenates with the MSFA-processed skip connection, then refines with two conv-BN-ReLU layers plus a residual (projection) connection.
- **Output head:** A final `Conv2DTranspose` upsample to the input resolution, followed by a 1×1 convolution with sigmoid activation for binary vessel/background segmentation.
- **Input size:** 256×256×3
- **Loss / metrics:** Binary cross-entropy loss, tracked with Dice coefficient, IoU (Jaccard) coefficient, and accuracy.

## Dataset

- **DRIVE** (Digital Retinal Images for Vessel Extraction) — 40 color fundus images (768×584) with corresponding vessel masks.

> Note: The DRIVE dataset is not included in this repository due to licensing. It's publicly available on [Kaggle](https://www.kaggle.com/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction) or the [official DRIVE website](https://drive.grand-challenge.org/). Place it in a `data/` directory before running the notebook.

Preprocessing applied: normalization ([0, 255] → [0, 1] for images, binary 0/1 for masks), resizing to 256×256, and data augmentation (horizontal/vertical flip, rotation, scaling) — expanding the dataset to 240 images, split 80% train / 10% test / 10% validation. Training uses Focal Loss (γ=2.0, α=0.25) to address the class imbalance between vessel and background pixels.

## Repository Contents

| File | Description |
|---|---|
| `Proposed_MSFAUMobileNet_Model.ipynb` | End-to-end notebook: data loading/preprocessing, model architecture, training, and evaluation |

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook / JupyterLab
- TensorFlow (Keras) with `MobileNetV2` pretrained weights
- Common scientific libraries: `numpy`, `opencv-python`, `matplotlib`, `scikit-learn`, `scikit-image`

### Installation

```bash
git clone https://github.com/Gamekillerzs/Retinal-blood-vessel-segmentation-model.git
cd Retinal-blood-vessel-segmentation-model
pip install -r requirements.txt   # add a requirements.txt if not already present
```

### Usage

1. Download and place the DRIVE dataset in the expected data directory.
2. Open `Proposed_MSFAUMobileNet_Model.ipynb` in Jupyter.
3. Run the cells sequentially to preprocess the data, train the model, and evaluate segmentation performance.

## Evaluation Metrics

The notebook tracks the following during training/evaluation:
- **Dice coefficient**
- **IoU (Jaccard) coefficient**
- **Accuracy**
- **Segmentation loss** — the notebook's example usage compiles with binary cross-entropy, while the published results use **Focal Loss** (γ=2.0, α=0.25) to counter the strong vessel/background class imbalance in DRIVE

Reported results from the published paper (trained on DRIVE with Adam, 200 epochs, batch size 3, Focal Loss for class imbalance):

| Metric | Training | Validation |
|---|---|---|
| Accuracy | 99.99% | 99.99% |
| Precision | 99.92% | 99.94% |
| Recall | 99.99% | 99.99% |
| F1-score | 99.95% | 99.97% |
| IoU (Jaccard) | 99.91% | 99.94% |
| Dice coefficient | 99.95% | 99.97% |
| Loss | 0.0012 | 0.0041 |

These figures substantially outperform prior state-of-the-art methods evaluated on DRIVE (e.g., MRU-Net: 98.37% accuracy / 72.91% IoU; CE-Net: 95.23% accuracy / 81.00% IoU) — see the paper's State-of-the-Art analysis for the full comparison.

## Citation

If you use this work in your research, please cite:
