# Riya's Contribution — Eye Disease Classification & Edge Detection Using Deep Learning

## 1. Introduction

This document details my individual contribution and engineering role in the **Eye Disease Detection & Classification** project.

My responsibilities encompass two vital pillars of the project:
1. **Data Engineering & Dataset Sanitization**: Leading the deep auditing pipeline, perceptual de-duplication, resolving cross-class contamination, and establishing zero-leakage dataset splits.
2. **Model Development & Edge Deployment**: Designing, implementing, training, and benchmarking **MobileNetV3-Large** (~5.4M parameters) using a 2-phase transfer learning regimen, class-weighted loss, and Grad-CAM explainability heatmaps.

### Assigned Model Specifications
- **Model Name**: MobileNetV3-Large
- **Category**: Light Model (Edge & Mobile Optimized)
- **Parameter Count**: $\sim 5.4\text{ Million}$
- **Training Schedule**: 20 – 25 Epochs (3 Warmup Epochs + $\sim 20$ Fine-Tuning Epochs)
- **Loss Strategy**: Class-Weighted Cross-Entropy Loss
- **Complexity**: Highly efficient, mobile-ready architecture

---

## 2. Project Alignment with Clinical Synopsis

Our overarching problem statement is:
> *"You wake up with a red, itchy, watery eye... Is it just dust and screen fatigue, or the start of something that could threaten your vision? Our project turns the camera everyone already carries into an early-warning eye check... Deploy the final model as a simple web or mobile application for easy use."*

My work directly fulfills this goal:
- High-capacity models (e.g. ResNet50 at 25.6M parameters or Vision Transformers at 86M parameters) are too computationally intensive to execute smoothly on everyday smartphones.
- **MobileNetV3-Large provides the low-latency, compact foundation** necessary to bring clinical diagnostic intelligence directly onto mobile devices and web browsers.

---

## 3. Comprehensive Data Audit: What We Got After Auditing

Before training, I conducted an exhaustive audit of the 14,664 raw images across all 12 classes (`datasets/raw/`). 

### 3.1 Raw Baseline Findings
- **Total Scanned Images**: 14,664
- **Format & Resolution**: 100% 3-channel RGB, standardized at $300 \times 300$ px.
- **Corrupted / Damaged Files**: 0 unreadable files (all image headers verified with PIL).
- **Exact Byte Duplicates**: 0 exact SHA-256 byte duplicates.

### 3.2 The Critical Discovery: Cross-Class Conflicting Images
While byte-level hashes showed no duplicates, a **perceptual difference hash (`dHash`)** audit uncovered a severe hidden flaw:
- **456 images across 167 visual duplicate groups appeared under conflicting disease labels.**
- Multiple online contributors and automated scrapers had uploaded the same patient's eye photo (or augmented variants of it) under different clinical categories.

#### Top Conflicting Pairs Discovered:
| Conflicting Disease Pair | Overlapping Images | Severity / Clinical Consequence |
| :--- | :---: | :--- |
| **Allergic Conjunctivitis $\leftrightarrow$ Viral Conjunctivitis** | **232** | High confusion between allergic inflammation and contagious viral eye flu |
| **Allergic Conjunctivitis $\leftrightarrow$ Bacterial Conjunctivitis** | **77** | Cross-labeled pink eye images |
| **Uveitis $\leftrightarrow$ Viral Conjunctivitis** | **45** | Red eye imagery cross-indexed under both diagnoses |
| **Bacterial Conjunctivitis $\leftrightarrow$ Viral Conjunctivitis** | **37** | Contagious infection ambiguity |
| **Healthy $\leftrightarrow$ Viral Conjunctivitis** | **29** | ⚠️ **Critical label noise**: Normal eyes tagged as infected |
| **Allergic Conjunctivitis $\leftrightarrow$ Healthy** | **19** | ⚠️ Normal eyes tagged as diseased |
| **Allergic Conjunctivitis $\leftrightarrow$ Uveitis** | **18** | Inflamed eye photos duplicated |
| **Allergic Conjunctivitis $\leftrightarrow$ Stye** | **12** | Eyelid / conjunctiva overlap |
| **Cataract $\leftrightarrow$ Glaucoma** | **6** | Anterior segment duplication |

---

## 4. How We Tackle Duplicate Images (And Why It Matters)

### 4.1 What Happens If We Do NOT Tackle Duplicate Images?
If these 456 conflicting images had been left in the dataset, the project would suffer four fatal consequences:
1. **Conflicting Gradients & Optimization Failure**: During backpropagation, the loss function computes weight updates $\nabla_\theta \mathcal{L}$. If image $X$ is labeled *Viral Conjunctivitis* in batch 1, the model is pushed to activate the Viral node. If the exact same image $X$ appears in batch 2 as *Allergic Conjunctivitis* or *Healthy*, the gradients pull in opposite directions. The optimizer oscillates and cannot converge.
2. **Artificial Accuracy Ceiling**: The neural network is penalized for recognizing real features, placing a hard cap on training accuracy (often plateauing at 65–75%).
3. **Cross-Split Data Leakage**: If image $X$ is placed in `train` as *Viral* and its duplicate lands in `test` as *Allergic*, the model will be evaluated as "wrong" even if it correctly recognized the visual features it was trained on.
4. **Clinical Disaster**: In an early-warning diagnostic tool, misclassifying a normal eye as contagious viral infection (or a diseased eye as healthy) destroys all medical credibility.

### 4.2 Our Solution & Sanitized Baseline
I upgraded [`prepare_data.py`](file:///c:/Users/RIYA/OneDrive/Apps/Desktop/projects/eye_disease/Eye-Disease-Detection/datasets/processed/prepare_data.py) to automatically:
1. Compute perceptual `dHash` for every image during the raw audit.
2. Flag and isolate all 456 conflicting images into [`reports/cross_class_conflicts.csv`](file:///c:/Users/RIYA/OneDrive/Apps/Desktop/projects/eye_disease/Eye-Disease-Detection/reports/cross_class_conflicts.csv).
3. Filter out these images entirely before creating training, validation, and test splits.
4. Standardize the remaining **14,208 clean images** to $224 \times 224$ RGB using Lanczos resampling.

### Clean Split Distribution (70% Train / 15% Val / 15% Test — Zero Leakage)
| Disease Class | Raw Count | Excluded Conflicts | Clean Total | Train (70%) | Val (15%) | Test (15%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Allergic Conjunctivitis** | 1,561 | -161 | **1,400** | 980 | 210 | 210 |
| **Bacterial Conjunctivitis** | 1,054 | -48 | **1,006** | 704 | 151 | 151 |
| **Blepharitis** | 1,428 | -6 | **1,422** | 995 | 213 | 214 |
| **Cataract** | 1,579 | -2 | **1,577** | 1,104 | 237 | 236 |
| **Glaucoma** | 424 | -2 | **422** | 295 | 63 | 64 |
| **Healthy** | 1,459 | -36 | **1,423** | 996 | 213 | 214 |
| **Pinguecula** | 366 | 0 | **366** | 256 | 55 | 55 |
| **Pterygium** | 1,715 | 0 | **1,715** | 1,200 | 257 | 258 |
| **Stye** | 1,686 | -13 | **1,673** | 1,171 | 251 | 251 |
| **Uveitis** | 529 | -14 | **515** | 360 | 77 | 78 |
| **Viral Conjunctivitis** | 1,619 | -174 | **1,445** | 1,011 | 217 | 217 |
| **Xanthelasma** | 1,244 | 0 | **1,244** | 871 | 187 | 186 |
| **Total** | **14,664** | **-456** | **14,208** | **9,943** | **2,131** | **2,134** |

---

## 5. How We Balance the Data: Class-Weighted Loss

### 5.1 The Imbalance Challenge
Our clean training split contains a significant frequency gap:
- **Majority Classes**: `Pterygium` (1,200 images), `Stye` (1,171 images), `Cataract` (1,104 images).
- **Minority Classes**: `Pinguecula` (256 images), `Glaucoma` (295 images), `Uveitis` (360 images).
- **Imbalance Ratio**: $\approx \mathbf{4.7\times}$.

If trained with standard unweighted Cross-Entropy loss:
- The gradients of majority classes dominate the updates ($\approx 80\%$ of gradient signal).
- The model exhibits lazy majority bias, achieving $\approx 85\%$ accuracy by guessing common classes while failing to detect dangerous, sight-threatening diseases like **Glaucoma**.

### 5.2 Why Class-Weighted Loss Over Resampling?
1. **Against Under-sampling**: Discarding 900+ images per majority class would throw away $\approx 60\%$ of our clinical data.
2. **Against Over-sampling**: Artificially repeating minority images in memory leads to memorization and overfitting.
3. **The Class-Weighted Solution**: We compute inverse-frequency weights dynamically:

$$w_c = \frac{N_{\text{total}}}{C \times N_c}$$

*(Where $N_{\text{total}} = 9943$, $C = 12$, and $N_c$ is the image count for class $c$)*

- **Glaucoma ($N_c = 295$)** receives a loss multiplier of $\approx \mathbf{2.81}$
- **Pinguecula ($N_c = 256$)** receives a loss multiplier of $\approx \mathbf{3.23}$
- **Pterygium ($N_c = 1200$)** receives a loss multiplier of $\approx \mathbf{0.69}$

When backpropagating loss:

$$\mathcal{L}_{\text{weighted}} = - w_c \cdot \log(p_{\text{true}})$$

Misclassifying a Glaucoma or Pinguecula image incurs a **$4\times$ heavier penalty**, forcing MobileNetV3 to learn distinct representations for rare conditions.

---

## 6. Model Architecture: MobileNetV3-Large

### 6.1 What is MobileNetV3-Large?
MobileNetV3-Large is a modern lightweight convolutional neural network developed by Google for high-efficiency mobile vision. It contains **$\sim 5.4\text{ Million}$ parameters** and is built on four core innovations:

1. **Depthwise Separable Convolutions**: Splits traditional convolution into a spatial depthwise filter ($3 \times 3$ or $5 \times 5$) followed by a pointwise $1 \times 1$ channel combination. Reduces computation by $8\times$ to $9\times$.
2. **Inverted Residuals & Linear Bottlenecks**: Expands feature channels in the inner block before contracting them back, preserving complex non-linear ocular features.
3. **Squeeze-and-Excitation (SE) Channel Attention**: Computes channel-wise importance dynamically, allowing the network to highlight relevant pathology regions (e.g. lens opacity) while ignoring irrelevant background skin.
4. **Hard-Swish ($\text{h-swish}$) Activation**:

$$\text{h-swish}(x) = x \cdot \frac{\text{ReLU6}(x + 3)}{6}$$

Eliminates expensive exponential computations required by standard sigmoid/swish, drastically accelerating inference on mobile CPUs.

### 6.2 Why Are We Using MobileNetV3-Large?
MobileNetV3-Large directly answers our project synopsis:
- **Instant Latency**: Executes inference in $<50\text{ms}$ on low-end smartphone processors without dedicated GPUs.
- **Minimal Memory Footprint**: Requires $<25\text{MB}$ of RAM, making it feasible to run directly inside mobile web browsers via ONNX Runtime or TensorFlow.js.
- **On-Device Privacy**: Patient images can be analyzed directly on the user's phone without transmitting sensitive medical photos over external cloud networks.

### 6.3 What Happens If We Do NOT Use MobileNetV3? (Impact on Project)
If MobileNetV3 is omitted and our project relies only on heavy architectures (ResNet-50, DenseNet-121, Vision Transformers):
1. **Deployment Failure**: ResNet-50 (25.6M) and ViT (86M) cannot run on consumer phones without high cloud hosting costs and network latency.
2. **Thermal Throttling & Battery Drain**: Running large networks on mobile CPUs causes severe overheating, frame drops ($>1.5\text{s}$ per frame), and poor user experience.
3. **Loss of Project Purpose**: The project would fail its primary goal of being an accessible, early-warning check for users in remote or rural areas with limited connectivity.

---

## 7. Training Regimen (2-Phase Transfer Learning)

To train MobileNetV3-Large without corrupting pretrained ImageNet weights, I designed a **two-phase training strategy**:

```
                       Phase 1: Warmup (3 Epochs)
      Pretrained ImageNet Backbone                New Dense Head (12 Classes)
   ┌─────────────────────────────────┐           ┌────────────────────────────┐
   │ 🔒 FROZEN BACKBONE              │ ───────▶  │ 🔓 TRAINABLE HEAD          │
   │ (Weights preserved, lr = 0)     │           │ (lr = 1e-3, AdamW)         │
   └─────────────────────────────────┘           └────────────────────────────┘
                                     ▼
                     Phase 2: Fine-Tuning (~20 Epochs)
   ┌─────────────────────────────────┐           ┌────────────────────────────┐
   │ 🔓 TOP BACKBONE LAYERS UNFROZEN │ ───────▶  │ 🔓 TRAINABLE HEAD          │
   │ (Fine feature adaptation)       │           │                            │
   │  lr = 1e-4 / 1e-5 (Cosine Decay)│           │  lr = 1e-4 / 1e-5          │
   └─────────────────────────────────┘           └────────────────────────────┘
```

### Phase 1: Warmup (Epochs 1–3)
- **Objective**: Adapt the newly initialized classification head (`Linear(960, 1280) -> Hardswish -> Dropout(0.2) -> Linear(1280, 12)`) without disturbing backbone features.
- **Backbone**: Frozen (`requires_grad = False`).
- **Optimizer**: AdamW, $\text{lr} = 10^{-3}$, weight decay $= 10^{-4}$.

### Phase 2: Fine-Tuning (Epochs 4–23)
- **Objective**: Adapt deep convolutional features to specific ocular structures (vascular patterns, corneal haze, eyelid inflammation).
- **Backbone**: Unfrozen (`requires_grad = True`).
- **Learning Rate**: $\text{lr} = 10^{-4}$ decayed via **Cosine Annealing** down to $10^{-6}$.
- **Loss Function**: `nn.CrossEntropyLoss(weight=class_weights)`.

### On-the-Fly Data Augmentation
To simulate real-world mobile photography (uneven lighting, tilted camera angles), training incorporates:
- Random Horizontal Flip ($p = 0.5$)
- Random Rotation ($\pm 15^\circ$)
- Color Jitter (Brightness $\pm 0.1$, Contrast $\pm 0.1$)
- Standard ImageNet Normalization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$)

---

## 8. Clinical Explainability: Grad-CAM Integration

In clinical applications, a deep learning model cannot be a "black box." I implemented **Grad-CAM (Gradient-Weighted Class Activation Mapping)**:
1. Captures forward activation maps and backpropagated gradients at MobileNetV3's final convolutional layer (`model.features[-1]`).
2. Computes the weighted combination of activations to produce a 2D spatial importance heatmap.
3. Overlays the heatmap in jet-colormap over the original eye photo.
4. **Clinical Benefit**: Validates whether the model is predicting based on actual ocular lesions (e.g., lens opacity in Cataract, or conjunctival injection in Conjunctivitis) rather than skin background, eyelashes, or camera reflections.

---

## 9. Key Project Decisions & Design Questions Answered

### Q1: Should We Use 12 Granular Classes or a 5-Class Scheme?
- **Decision**: **12 Granular Classes**.
- **Rationale**: Distinguishing between *Bacterial*, *Viral*, and *Allergic Conjunctivitis* is clinically vital because Bacterial requires antibiotic eye drops, Viral is contagious ("eye flu") requiring quarantine, and Allergic requires antihistamines. MobileNetV3 has sufficient capacity to handle 12 classes when trained with class-weighted loss.

### Q2: How Should Duplicate / Conflicting Images Be Handled?
- **Decision**: **Filtered Out (Clean Single-Label Ground Truth)**.
- **Rationale**: Leaving 456 conflicting images in the dataset injects label noise and produces gradient cancellation. Isolating them creates clean, verifiable evaluation metrics.

### Q3: What Is the Evaluation Focus?
- **Decision**: **High Recall (Sensitivity) for High-Risk Conditions**.
- **Rationale**: A false positive (telling a patient with minor irritation to see a doctor) is acceptable; a false negative (missing Glaucoma or Uveitis until permanent vision loss occurs) is catastrophic. Class weighting ensures minority disease recall stays high.

---

## 10. File & Pipeline Deliverables

1. [`datasets/processed/prepare_data.py`](file:///c:/Users/RIYA/OneDrive/Apps/Desktop/projects/eye_disease/Eye-Disease-Detection/datasets/processed/prepare_data.py): Automated audit, perceptual duplicate filtering, and stratified 70/15/15 splitting pipeline.
2. [`train_mobilenetv3.py`](file:///c:/Users/RIYA/OneDrive/Apps/Desktop/projects/eye_disease/Eye-Disease-Detection/train_mobilenetv3.py): End-to-end PyTorch training pipeline featuring 2-phase warmup/fine-tuning, class weighting, test evaluation, and Grad-CAM generation.
3. [`reports/audit_summary.csv`](file:///c:/Users/RIYA/OneDrive/Apps/Desktop/projects/eye_disease/Eye-Disease-Detection/reports/audit_summary.csv): Summary metrics of raw vs clean data.
4. [`reports/cross_class_conflicts.csv`](file:///c:/Users/RIYA/OneDrive/Apps/Desktop/projects/eye_disease/Eye-Disease-Detection/reports/cross_class_conflicts.csv): Complete log of the 456 excluded duplicate images.
5. [`reports/split_distribution.csv`](file:///c:/Users/RIYA/OneDrive/Apps/Desktop/projects/eye_disease/Eye-Disease-Detection/reports/split_distribution.csv): Exact per-class split counts.

---

## 11. How to Execute My Pipeline

### Step 1: Prepare and Audit the Data
```bash
python datasets/processed/prepare_data.py
```

### Step 2: Train MobileNetV3-Large & Generate Reports
```bash
python train_mobilenetv3.py
```
*(Executes 3 warmup epochs + 20 fine-tuning epochs, saves `models/mobilenetv3_large_best.pt`, plots learning curves, outputs test classification report, confusion matrix, and saves Grad-CAM heatmaps to `reports/gradcam_samples/`).*

