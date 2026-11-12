# Eye Disease Detection & Classification Using Deep Learning

An end-to-end deep learning project for automated detection and multi-class classification of eye diseases from clinical ocular imagery.

---

## 1. Project Overview & Objectives

The primary goal of this project is to develop, evaluate, and benchmark deep convolutional neural network (CNN) and transformer architectures for classifying eye disease images:
* **Custom CNN Baseline** (Trained from scratch)
* **ResNet50** (Residual connections, transfer learning)
* **DenseNet121** (Dense feature-reuse connections)
* **Vision Transformer (ViT)** (Self-attention patch processing, optional/exploratory)

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
│       ├── train/                 # 70% Training set
│       ├── val/                   # 15% Validation set
│       ├── test/                  # 15% Independent test set
│       └── prepare_data.py        # Automated preprocessing & audit script
├── models/                        # Model architectures and saved checkpoints
├── reports/
│   ├── audit_summary.csv          # High-level pipeline metrics
│   ├── split_distribution.csv     # Per-class split distribution
│   ├── exact_duplicates.csv       # Pre-split duplicate audit report
│   ├── corrupted_images.csv       # Verification of file read integrity
│   └── image_dimensions.csv       # Dimensional metadata
├── Eye Dataset.v2-new-dataset.folder.zip # New raw dataset archive
├── DEVREADME.md                   # Developer contributions and architecture plan
└── README.md                      # Project documentation and audit tracker
```

---

## 6. How to Run the Preprocessing Pipeline

Run the automated data preparation and audit pipeline:
```bash
python datasets/processed/prepare_data.py
```

This performs:
1. Dynamic class scanning and raw data audit.
2. Grouped, stratified 70 / 15 / 15 Train / Val / Test splitting.
3. Conversion to 3-channel RGB with high-quality Lanczos resampling to $224 \times 224$.
4. Post-processing cross-split leakage verification.
5. Automated generation of audit CSVs in `reports/`.
