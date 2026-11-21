# Eye Disease Detection & Classification Using Deep Learning

An end-to-end deep learning project for automated detection and multi-class classification of eye diseases from clinical ocular imagery.

---

## 1. Project Overview & Objectives

The primary goal of this project is to develop, evaluate, and benchmark deep convolutional neural network (CNN) and transfer-learning architectures for classifying eye disease images across 12 granular categories:
* **Custom EyeDiseaseCNN** (Trained from scratch, compact baseline ~983k parameters — Tejender)
* **DenseNet121** (Dense feature-reuse connections, transfer learning — Dev)
* **MobileNetV3-Large** (Efficient mobile inverted bottleneck architecture — Riya)
* **EfficientNetV2-S** (Progressive learning and fused MBConv architecture — Vidushi)

The target input format for all architectures is standardized to **$224 \times 224 \times 3$ RGB**.

---

## 2. Dataset Evolution & History

### 2.1 Initial Dataset (Mendeley / Kaggle) — Audited & Deprecated
The project initially incorporated an ocular dataset containing 2,298 images across 5 classes (`Cataract`, `Conjunctivitis`, `Eyelid`, `Normal`, `Uveitis`).

A comprehensive pre-processing audit revealed critical data quality issues:
1. **Severe Low Resolution in `Normal`**: 559 out of 649 `Normal` images (86.1%) were tiny thumbnails smaller than $64 \times 64$ (many $41 \times 41$, max $89 \times 89$). Upscaling these to $224 \times 224$ created severe blur and interpolation artifacts that distorted CNN feature extraction.
2. **Pre-Augmented Contamination**: 340 images prefixed with `aug_*` were mixed directly into the raw folders, risking test set contamination.
3. **Exact Duplicates**: Identified exact byte-for-byte duplicates (e.g., `Cataract/184.JPG` and `534.JPG`, SHA256: `9f4d39c8...`).

**Decision**: The old dataset was completely purged from the active pipeline to protect model integrity, with backups preserved.

---

### 2.2 New Dataset (Roboflow Universe v2) — Current Working Dataset
Imported from Roboflow Universe (`Eye Dataset.v2-new-dataset.folder.zip`).

* **Total Images**: 14,735
* **Format & Resolution**: 100% RGB, standardized to $300 \times 300$ pixels.
* **Corrupted Files**: 0 corrupted files.
* **Resolution Quality**: Completely resolves the previous low-resolution issue.

#### Discovered Classes & Image Counts
The new dataset expands coverage to **12 distinct ocular conditions**:

| Class | Image Count | Description / Clinical Meaning |
|---|---|---|
| **Healthy** | 1,472 | Normal / healthy eye images |
| **Allergic Conjunctivitis** | 1,584 | Allergic eye inflammation |
| **Bacterial Conjunctivitis** | 1,057 | Bacterial eye infection |
| **Viral Conjunctivitis** | 1,651 | Viral eye infection |
| **Cataract** | 1,579 | Clouding of the crystalline lens |
| **Blepharitis** | 1,428 | Inflammation of the eyelid margin |
| **Stye** | 1,686 | Acute focal eyelid infection |
| **Xanthelasma** | 1,244 | Cholesterol deposits on/around eyelids |
| **Glaucoma** | 424 | Optic nerve neuropathy |
| **Pinguecula** | 366 | Conjunctival protein/fat deposits |
| **pterygium** | 1,715 | Conjunctival fibrovascular growth onto cornea |
| **Uveitis** | 529 | Intraocular uveal tract inflammation |
| **Total** | **14,735** | |

---

## 3. Deep Dataset Audit & Critical Findings

Before training, an in-depth audit of the 14,735 images revealed key insights:

### Finding 1: Offline Augmentation Multiplication
* The 14,735 files stem from **5,576 unique base images** (an average of $\approx 2.64\times$ copies/augmentations per base image, such as flips, rotations, and contrast shifts).

### Finding 2: Roboflow Pre-Split Data Leakage
* In Roboflow's default `train/`, `valid/`, and `test/` partitions, **813 base images cross split boundaries** (e.g. an original image in `train` with its flipped/rotated counterpart in `test`).
* Using this split out of the box leads to inflated evaluation metrics. 
* **Mitigation**: Our custom preprocessing script (`prepare_data.py`) partitions data by unique base image hash to guarantee zero leakage.

### Finding 3: Label Contradictions / Multi-Label Images
* **336 base images are assigned to multiple conflicting disease classes**:
  * `('Stye', 'Xanthelasma')`: 102 images
  * `('Cataract', 'Xanthelasma')`: 93 images
  * `('Cataract', 'Stye', 'Xanthelasma')`: 77 images
  * `('Glaucoma', 'Xanthelasma')`: 27 images
* Patients presenting multiple concurrent conditions were duplicated across multiple folders, creating noisy labels for single-label softmax classifiers.

---

## 4. Key Questions & Design Decisions

### Question 1: Class Taxonomy — 12-Class vs. 5-Class Alignment?
* **Option A: 12-Class Model (Fine-Grained)**:
  * Retains all 12 granular classes as provided by Roboflow.
  * *Advantage*: High clinical specificity (e.g. distinguishes Bacterial vs. Viral Conjunctivitis).
  * *Challenge*: Mild class imbalance (e.g. 1,715 pterygium vs. 366 Pinguecula).
* **Option B: 5-Class Model (Coarse-Grained Project Alignment)**:
  * Maps labels into the original 5-class schema:
    * `Healthy` $\rightarrow$ **`Normal`**
    * `Allergic`, `Bacterial`, `Viral Conjunctivitis` $\rightarrow$ **`Conjunctivitis`** (4,292 images)
    * `Blepharitis`, `Stye`, `Xanthelasma` $\rightarrow$ **`Eyelid`** (4,358 images)
    * `Cataract` $\rightarrow$ **`Cataract`** (1,579 images)
    * `Uveitis` $\rightarrow$ **`Uveitis`** (529 images)
    * *(Exclude or treat Glaucoma, Pinguecula, Pterygium as secondary)*

### Question 2: How Should Duplicate / Conflicting Images Be Handled?
* Should multi-condition images (336 cases) be:
  1. Dropped to maintain pure single-label ground truth?
  2. Retained under a primary dominant condition?
  3. Reformulated into multi-label classification (BCEWithLogitsLoss)?

### Question 3: Split Strategy & Leakage Prevention
* How strictly should augmented variants be isolated?
* **Agreed Strategy**: Implement stratified **70% Train / 15% Validation / 15% Test** grouped strictly by unique image identity to ensure zero test set data contamination.

---

## 5. Pipeline & Project File Structure

```text
Eye-Disease-Detection/
├── datasets/
│   ├── raw/                       # Extracted clean raw dataset
│   └── processed/                 # 70/15/15 preprocessed splits (224x224 RGB)
│       ├── train/                 # 70% Training set (9,943 images)
│       ├── val/                   # 15% Validation set (2,131 images)
│       ├── test/                  # 15% Independent test set (2,134 images)
│       └── prepare_data.py        # Automated preprocessing & audit script
├── models/                        # Saved checkpoints (.pt)
│   ├── custom_cnn_best.pt         # Best Custom EyeDiseaseCNN weights (Epoch 23)
│   ├── densenet121_best.pt        # Best DenseNet121 weights (Epoch 23)
│   ├── mobilenetv3_large_best.pt  # Best MobileNetV3-Large weights (Epoch 16)
│   └── efficientnetv2_s_best.pt   # Target checkpoint for EfficientNetV2-S
├── train_custom_cnn.py            # Custom EyeDiseaseCNN training script (Tejender)
├── gradcam_custom_cnn.py          # Standalone Grad-CAM visualizer for Custom CNN
├── Dev_train_densenet121.py       # DenseNet121 transfer learning pipeline (Dev)
├── densenet_gradcam.py            # Standalone Grad-CAM visualizer for DenseNet121
├── train_mobilenetv3.py           # MobileNetV3-Large transfer learning pipeline (Riya)
├── train_efficientnetv2_s.py      # EfficientNetV2-S transfer learning pipeline (Vidushi)
├── reports/
│   ├── audit_summary.csv          # High-level pipeline metrics
│   ├── split_distribution.csv     # Per-class split distribution
│   ├── custom_cnn_test_report.txt # Test evaluation metrics for Custom CNN
│   ├── custom_cnn_confusion_matrix.png # Custom CNN confusion matrix plot
│   ├── custom_cnn_learning_curves.png  # Custom CNN loss & accuracy curves
│   ├── densenet121_test_report.txt     # Test evaluation metrics for DenseNet121
│   ├── densenet121_confusion_matrix.png # DenseNet121 confusion matrix plot
│   ├── densenet121_learning_curves.png  # DenseNet121 loss & accuracy curves
│   ├── mobilenetv3_test_report.txt     # Test evaluation metrics for MobileNetV3
│   ├── mobilenetv3_confusion_matrix.png # MobileNetV3 confusion matrix plot
│   ├── mobilenetv3_learning_curves.png  # MobileNetV3 loss & accuracy curves
│   └── gradcam_samples/           # Generated Grad-CAM overlays
├── DEVREADME.md                   # Dev's DenseNet121 documentation
├── RIYA_README.md                 # Riya's MobileNetV3 documentation
├── TejenderReadme.md              # Tejender's Custom CNN baseline documentation
├── VidushiReadme.md               # Vidushi's EfficientNetV2-S documentation
└── README.md                      # Project documentation and audit tracker
```

---

## 6. How to Run

### 6.1 Data Preparation & Audit Pipeline
Run the automated preprocessing pipeline:
```bash
python datasets/processed/prepare_data.py
```
This performs:
1. Dynamic class scanning and raw data audit.
2. Grouped, stratified 70 / 15 / 15 Train / Val / Test splitting without cross-split leakage.
3. Conversion to 3-channel RGB with high-quality Lanczos resampling to $224 \times 224$.
4. Automated generation of audit CSVs in `reports/`.

### 6.2 Training Custom EyeDiseaseCNN Baseline
Train the 12-class custom convolutional network from scratch:
```bash
python train_custom_cnn.py
```
Outputs are saved to `models/custom_cnn_best.pt` and corresponding reports in `reports/`.

### 6.3 Training MobileNetV3-Large (Edge Model)
Train the 12-class MobileNetV3-Large transfer-learning model (3 warmup epochs + 20 fine-tuning epochs):
```bash
python train_mobilenetv3.py
```
Outputs are saved to `models/mobilenetv3_large_best.pt` and corresponding reports in `reports/`.

### 6.4 Training DenseNet121 (Dense Feature-Reuse Model)
Train the DenseNet121 transfer-learning model (3 warmup epochs + 20 fine-tuning epochs):
```bash
python Dev_train_densenet121.py
```
Outputs are saved to `models/densenet121_best.pt` and corresponding reports in `reports/`.

### 6.5 Training EfficientNetV2-S Pipeline (Vidushi)
Train the 12-class EfficientNetV2-S transfer learning model:
```bash
python train_efficientnetv2_s.py
```
This executes:
1. Inverse-frequency class weighting to mitigate class imbalance.
2. Phase 1 (3 Warmup Epochs) with feature backbone frozen.
3. Phase 2 (20 Fine-Tuning Epochs) with full backbone unfrozen and Cosine Annealing learning rate schedule.
4. Unseen test set evaluation, saving best weights to `models/efficientnetv2_s_best.pt` and metrics to `reports/`.

### 6.6 Standalone Grad-CAM Visualizations
Generate explainability heatmaps on test classes or a specific eye image:
```bash
# Generate sample heatmaps for all 12 classes
python gradcam_custom_cnn.py

# Explain a specific test image
python gradcam_custom_cnn.py --image "datasets/processed/test/Cataract/Cataract_00000.jpg"
```

---

## 7. Model Benchmark & Test Evaluation

All models are evaluated on the exact same unseen test set of **2,134 images** across 12 disease classes:

| Model | Pretrained Weights | Trainable Params | Test Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|---|---|
| **EyeDiseaseCNN (Custom Baseline)** | **None (From Scratch)** | **983,052** | **75.54%** | **0.7431** | **0.7703** | **0.7511** |
| **MobileNetV3-Large (Riya)** | **ImageNet-1k** | **~5,400,000** | **92.60%** | **0.9212** | **0.9233** | **0.9220** |
| **DenseNet121 (Dev)** | **ImageNet-1k** | **~7,000,000** | **92.88%** | **0.9183** | **0.9276** | **0.9223** |
| **EfficientNetV2-S (Vidushi)** | **ImageNet-1k** | **~21,458,488** | *Ready for Training* | *Ready for Training* | *Ready for Training* | *Ready for Training* |

### Observations:
- **Baseline Efficiency**: The custom scratch baseline reaches **75.54% accuracy** with fewer than 1 million parameters, demonstrating that a compact, specialized convolutional network can learn ocular representations without pretraining.
- **Edge Model Optimization (MobileNetV3-Large)**: MobileNetV3-Large achieves virtually identical macro-F1 to DenseNet121 (**0.9220 vs. 0.9223**) and **92.60% accuracy** while using ~23% fewer parameters, offering superior inference throughput and minimal latency suitable for on-device mobile diagnostics.
- **Dense Feature-Reuse (DenseNet121)**: DenseNet121 reaches the highest overall accuracy (**92.88%**) and macro recall (**0.9276**), benefiting from dense skip connections that concatenate multi-level feature maps.
- **Architecture Scaling (EfficientNetV2-S)**: EfficientNetV2-S provides the highest representation capacity (~21.5M parameters) with modern Fused-MBConv layers that maximize GPU compute density and training throughput.
- **Top Recognized Categories**: All architectures achieve >95% F1 scores on conditions with distinct structural landmarks: *Pinguecula*, *Xanthelasma*, *Pterygium*, and *Cataract*.
- **Clinical Ambiguity Area**: Across all models, differentiating between *Allergic Conjunctivitis* (~0.78 F1) and *Viral Conjunctivitis* (~0.83 F1) presents the most challenging classification boundary due to shared clinical signs of conjunctival redness and swelling.
