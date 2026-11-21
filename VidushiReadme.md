# Vidushi's Contribution — EfficientNetV2-S Eye Disease Classification Pipeline

## 1. Executive Summary & Contribution Overview

This document details my engineering contribution to the **Eye Disease Detection & Classification** deep learning project. 

In this collaborative benchmark across 12 ocular conditions, each team member implements, trains, and evaluates a distinct deep learning architecture under standardized preprocessing splits and evaluation criteria. My contribution centers on the implementation, training architecture, and performance benchmarking of **EfficientNetV2-S** (Small variant of EfficientNetV2).

### Key Technical Specifications
| Specification | Details |
|---|---|
| **Assigned Model** | **EfficientNetV2-S** (`torchvision.models.efficientnet_v2_s`) |
| **Model Category** | High-Capacity, Fast-Training Modern CNN Backbone |
| **Pretrained Weights** | ImageNet-1K (`EfficientNet_V2_S_Weights.DEFAULT`) |
| **Total Parameters** | $\approx 21,458,488$ (~21.5 Million) |
| **Input Dimensions** | $224 \times 224 \times 3$ RGB (Lanczos resampled) |
| **Classifier Head** | `nn.Dropout(p=0.2)` $\rightarrow$ `nn.Linear(in_features=1280, out_features=12)` |
| **Optimization Strategy** | 2-Phase Transfer Learning (3 Warmup Epochs + 20 Fine-Tuning Epochs) |
| **Imbalance Mitigation** | Inverse-Frequency Class Weighted Cross-Entropy Loss |
| **Optimizer & Scheduler** | AdamW ($\text{weight decay} = 10^{-4}$) with Cosine Annealing Learning Rate Decay |
| **Pipeline Script** | [`train_efficientnetv2_s.py`](file:///c:/Users/Vidushi/Documents/SEM5/DEEP_LEARNING_SEM5/Project/Eye-Disease-Detection/train_efficientnetv2_s.py) |
| **Checkpoint Path** | `models/efficientnetv2_s_best.pt` |

---

## 2. Clinical Alignment & Architectural Motivation

The clinical goal of this project is to provide automated early-warning screening for ocular diseases using standard clinical and mobile ocular photography:
> *"You wake up with a red, itchy, watery eye... Is it just dust and screen fatigue, or the start of something that could threaten your vision? Our project turns ocular photography into an early-warning eye check across 12 distinct conditions."*

### Why EfficientNetV2-S for Ocular Disease Classification?
1. **High Visual Granularity in Anterior Eye Conditions**:
   Distinguishing conditions such as **Pinguecula** (conjunctival elastotic degeneration) from early **Pterygium** (fibrovascular limbal growth) or distinguishing **Allergic Conjunctivitis** from **Bacterial/Viral Conjunctivitis** requires fine-grained texture, color distribution, and vascular geometry features. EfficientNetV2-S provides 21.5M parameters with multi-scale receptive fields, enabling deep contextual feature extraction without catastrophic overfitting.
2. **Speed & Modern GPU Acceleration (Fused-MBConv)**:
   While original EfficientNetV1 models scaled depth, width, and resolution, they suffered from slow training speeds on GPUs due to memory access overhead from depthwise separable convolutions in early layers. EfficientNetV2 solves this by replacing early depthwise convolutions with **Fused-MBConv**, speeding up training and convergence significantly.
3. **Benchmarking Role within the Team**:
   Our project evaluates a spectrum of model complexities:
   - **Custom EyeDiseaseCNN** (Tejender): Compact baseline trained from scratch (~983K params).
   - **MobileNetV3-Large** (Riya): Light edge-optimized architecture (~5.4M params).
   - **DenseNet121** (Dev): Dense feature-reuse network (~7.0M params).
   - **EfficientNetV2-S** (Vidushi): High-capacity progressive modern architecture (~21.5M params).

---

## 3. Deep Architectural Breakdown of EfficientNetV2-S

EfficientNetV2 was developed by Mingxing Tan and Quoc V. Le (Google Research, ICML 2021) to simultaneously optimize **accuracy**, **parameter efficiency**, and **practical training speed**.

```text
Input (3 x 224 x 224)
  │
  ▼
[Stem: Conv3x3 + BatchNorm + SiLU] (Stride 2) ──> Output: 24 channels
  │
  ▼
[Stage 1: 2x Fused-MBConv1, k3x3]  ──> Output: 24 channels
  │
  ▼
[Stage 2: 4x Fused-MBConv4, k3x3] (Stride 2) ──> Output: 48 channels
  │
  ▼
[Stage 3: 4x Fused-MBConv4, k3x3] (Stride 2) ──> Output: 64 channels
  │
  ▼
[Stage 4: 6x MBConv4 + SE, k3x3] (Stride 2) ──> Output: 128 channels
  │
  ▼
[Stage 5: 9x MBConv6 + SE, k3x3] (Stride 1) ──> Output: 160 channels
  │
  ▼
[Stage 6: 15x MBConv6 + SE, k3x3] (Stride 2) ──> Output: 256 channels
  │
  ▼
[Head: Conv1x1 + BatchNorm + SiLU] ──> Output: 1280 channels
  │
  ▼
[AdaptiveAvgPool2d (1 x 1)]
  │
  ▼
[Dropout (p = 0.2)]
  │
  ▼
[Linear: 1280 ──> 12 Disease Classes]
```

### 3.1 The Fused-MBConv Innovation
In standard MBConv blocks (used in MobileNetV2/V3 and EfficientNetV1):
- A $1 \times 1$ convolution expands the channel dimension.
- A $3 \times 3$ depthwise convolution applies spatial filtering per channel.
- A $1 \times 1$ projection convolution reduces the channels back.

While depthwise convolutions have fewer theoretical FLOPs, their arithmetic intensity ($\frac{\text{FLOPs}}{\text{Memory Access}}$) is low. On modern GPUs, memory bandwidth overhead dominates runtime in the early layers where spatial feature maps are large (e.g. $112 \times 112$ or $56 \times 56$).

**Fused-MBConv** replaces the $1 \times 1$ expansion convolution and $3 \times 3$ depthwise convolution with a single standard $3 \times 3$ convolution:
$$\text{Fused-MBConv}: \quad x \longrightarrow \text{Conv3x3}(\text{Expand}) \longrightarrow \text{BN} \longrightarrow \text{SiLU} \longrightarrow \text{Conv1x1}(\text{Project}) \longrightarrow \text{BN} \longrightarrow (+ x)$$
In EfficientNetV2-S, Fused-MBConv is used in Stages 1, 2, and 3. In later Stages (4, 5, 6), where spatial dimensions are smaller ($28 \times 28$ and below) and channels are large, standard MBConv with Squeeze-and-Excitation (SE) is retained to maximize representational capacity without memory bottlenecks.

### 3.2 Squeeze-and-Excitation (SE) Channel Attention
In Stages 4 to 6, Squeeze-and-Excitation modules adaptively recalibrate channel-wise feature maps by explicitly modeling interdependencies between channels:
1. **Squeeze**: Global average pooling squeezes spatial dimensions $H \times W$ into a $1 \times 1 \times C$ channel descriptor:
   $$z_c = \frac{1}{H \times W} \sum_{i=1}^H \sum_{j=1}^W u_c(i, j)$$
2. **Excitation**: A gating mechanism with two fully connected layers and a non-linear activation captures channel-wise dependencies:
   $$s = \sigma\left(W_2 \cdot \text{SiLU}(W_1 \cdot z)\right)$$
   where $W_1 \in \mathbb{R}^{\frac{C}{r} \times C}$ reduces dimensionality and $W_2 \in \mathbb{R}^{C \times \frac{C}{r}}$ restores it (reduction ratio $r = 4$).
3. **Scale**: The input feature map $u_c$ is scaled by the activation weight $s_c$:
   $$\tilde{x}_c = s_c \cdot u_c$$

---

## 4. Dataset Distribution & Imbalance Handling

The model is trained on the sanitized 70% / 15% / 15% dataset split located in `datasets/processed/`:

### 4.1 Split Summary
| Split | Images | Purpose |
|---|---|---|
| **Train** | 9,943 | Model training and parameter updates (Phase 1 & Phase 2) |
| **Validation** | 2,131 | Checkpoint selection based on Validation Macro F1 |
| **Test** | 2,134 | Strictly held-out, unseen test evaluation |
| **Total** | **14,208** | Zero cross-split data leakage via perceptual hash grouping |

### 4.2 Class Distribution & Inverse-Frequency Weighting
The 12 ocular disease classes exhibit significant frequency imbalance:
- High-frequency classes: **Pterygium** (1,200 train images), **Stye** (1,171 train images).
- Low-frequency classes: **Pinguecula** (256 train images), **Glaucoma** (295 train images).

To prevent majority classes from dominating the cross-entropy gradients, we apply inverse-frequency class weighting:
$$w_c = \frac{N_{\text{total}}}{K \cdot N_c}$$
where $N_{\text{total}} = 9,943$, $K = 12$ (number of classes), and $N_c$ is the image count for class $c$.

In [`train_efficientnetv2_s.py`](file:///c:/Users/Vidushi/Documents/SEM5/DEEP_LEARNING_SEM5/Project/Eye-Disease-Detection/train_efficientnetv2_s.py), this weight tensor is passed to PyTorch's loss function:
```python
criterion = nn.CrossEntropyLoss(weight=class_weights)
```
This forces the network to penalize misclassifications on minority conditions (e.g. Pinguecula, Glaucoma, Uveitis) with higher loss magnitudes.

---

## 5. Two-Phase Transfer Learning Strategy

Directly fine-tuning a 21.5M-parameter network end-to-end with randomly initialized head weights can distort the valuable features pretrained on ImageNet-1K. To avoid this catastrophic interference, the training pipeline utilizes a two-phase protocol:

```text
Phase 1: Warmup (Epochs 1 - 3)
┌───────────────────────────────────────┬───────────────────────────────────────┐
│ Feature Backbone (model.features)    │ Custom Head (model.classifier)        │
│ [FROZEN] requires_grad = False        │ [TRAINABLE] AdamW, lr = 1e-3          │
└───────────────────────────────────────┴───────────────────────────────────────┘
                                   │
                                   ▼
Phase 2: Fine-Tuning (Epochs 4 - 23)
┌───────────────────────────────────────┬───────────────────────────────────────┐
│ Feature Backbone (model.features)    │ Custom Head (model.classifier)        │
│ [UNFROZEN] requires_grad = True       │ [UNFROZEN] requires_grad = True       │
│                                       │                                       │
│ Optimizer: AdamW, Initial lr = 1e-4, Weight Decay = 1e-4                      │
│ Scheduler: CosineAnnealingLR (T_max = 20, eta_min = 1e-6)                     │
└───────────────────────────────────────┴───────────────────────────────────────┘
```

### 5.1 Cosine Annealing Learning Rate Decay
During Phase 2, learning rates decay smoothly following the cosine curve:
$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T_{\max}}\pi\right)\right)$$
where $\eta_{\max} = 10^{-4}$, $\eta_{\min} = 10^{-6}$, and $T_{\max} = 20$ epochs. This ensures steady exploration in early fine-tuning and gentle convergence towards a flat minimum in later epochs.

### 5.2 Checkpointing on Validation Macro F1
Model weights are saved to `models/efficientnetv2_s_best.pt` exclusively when the **Validation Macro F1-score** reaches a new high. Because accuracy can be artificially inflated by predicting the majority class, Macro F1 treats all 12 classes equally, ensuring optimal generalization across all disease categories.

---

## 6. Implementation Walkthrough (`train_efficientnetv2_s.py`)

The complete training script is implemented in [`train_efficientnetv2_s.py`](file:///c:/Users/Vidushi/Documents/SEM5/DEEP_LEARNING_SEM5/Project/Eye-Disease-Detection/train_efficientnetv2_s.py).

### Core Components:
1. **Reproducibility Guarantee**:
   Seeds for `torch`, `numpy`, and `random` are pinned to `42`.
2. **Data Transformation & Augmentation**:
   - `Resize((224, 224))`
   - `RandomHorizontalFlip(p=0.5)`
   - `RandomRotation(degrees=15)`
   - `ColorJitter(brightness=0.1, contrast=0.1)`
   - Standard ImageNet normalization: $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$.
3. **Head Modification**:
   ```python
   weights = EfficientNet_V2_S_Weights.DEFAULT
   model = models.efficientnet_v2_s(weights=weights)
   in_features = model.classifier[1].in_features  # 1280
   model.classifier[1] = nn.Linear(in_features, num_classes)
   ```
4. **Diagnostic Artifacts Generated**:
   - Checkpoint: `models/efficientnetv2_s_best.pt`
   - Epoch History: `reports/efficientnetv2_s_training_metrics.csv`
   - Learning Curves: `reports/efficientnetv2_s_learning_curves.png` (Loss & Accuracy plots)
   - Test Set Report: `reports/efficientnetv2_s_test_report.txt`
   - Confusion Matrix: `reports/efficientnetv2_s_confusion_matrix.png`

### Explainability Design Note (No Grad-CAM in Training Script)
In accordance with clean software engineering practices and the explicit project instructions, [`train_efficientnetv2_s.py`](file:///c:/Users/Vidushi/Documents/SEM5/DEEP_LEARNING_SEM5/Project/Eye-Disease-Detection/train_efficientnetv2_s.py) focuses strictly on training and evaluation. Grad-CAM visualizers are kept in separate standalone scripts (e.g., `gradcam_custom_cnn.py`, `densenet_gradcam.py`), avoiding extraneous computational overhead and code clutter during model optimization.

---

## 7. How to Execute

### Execution Command:
```bash
python train_efficientnetv2_s.py
```

### Hardware & Environment Requirements:
- **GPU**: CUDA-compatible GPU (e.g. NVIDIA RTX 3050/3060/4060 or Google Colab T4/V100). The script automatically detects CUDA and falls back to CPU if unavailable.
- **Batch Size**: Standardized to `16` to ensure stable memory consumption during backpropagation across the 21.5M parameters.
- **Dependencies**: `torch`, `torchvision`, `numpy`, `matplotlib`, `seaborn`, `pillow`, `scikit-learn`.

---

## 8. Comparative Architecture Analysis

| Metric / Dimension | Custom EyeDiseaseCNN (Tejender) | MobileNetV3-Large (Riya) | DenseNet121 (Dev) | EfficientNetV2-S (Vidushi) |
|---|---|---|---|---|
| **Backbone Paradigm** | Vanilla 5-block CNN | Inverted Residuals + Hardswish | Dense Feature Concatenation | Fused-MBConv + MBConv + SE |
| **Pretrained Weights** | None (Trained from scratch) | ImageNet-1K | ImageNet-1K | ImageNet-1K |
| **Total Parameters** | ~983,000 | ~5,400,000 | ~7,000,000 | **~21,458,488** |
| **Primary Advantage** | Transparent baseline, no pretraining bias | Extreme mobile/edge efficiency, low latency | Dense feature reuse, deep gradient paths | **High capacity, fused GPU kernels, fast convergence** |
| **Input Resolution** | $224 \times 224$ | $224 \times 224$ | $224 \times 224$ | $224 \times 224$ |
| **Training Epochs** | 25 | 23 (3 + 20) | 23 (3 + 20) | 23 (3 + 20) |
| **Evaluation Set** | 2,134 test images | 2,134 test images | 2,134 test images | 2,134 test images |

### Summary & Takeaways
EfficientNetV2-S represents the most parameter-rich and architecturally modern model in our comparative study. By combining early-stage Fused-MBConv layers for rapid GPU computation with late-stage Squeeze-and-Excitation attention blocks, it is positioned to deliver state-of-the-art representation fidelity on complex ocular surface diseases while maintaining an efficient training profile.
