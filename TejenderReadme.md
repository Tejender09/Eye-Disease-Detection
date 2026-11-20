# Tejender's Contribution — Custom CNN Eye Disease Classifier

## 1. Contribution Overview

My contribution is a custom convolutional neural network baseline for
12-class eye disease classification. Unlike the MobileNetV3-Large and
DenseNet121 models, this model is trained entirely from scratch without
ImageNet pretrained weights.

The purpose of this model is to provide a simple, transparent baseline for
comparing a task-specific CNN against the project's transfer-learning models.

## 2. Dataset

The model uses the cleaned dataset in `datasets/processed`, which contains:

| Split | Images |
|---|---:|
| Train | 9,943 |
| Validation | 2,131 |
| Test | 2,134 |

The 12 classes are:

1. Allergic Conjunctivitis
2. Bacterial Conjunctivitis
3. Blepharitis
4. Cataract
5. Glaucoma
6. Healthy
7. Pinguecula
8. Pterygium
9. Stye
10. Uveitis
11. Viral Conjunctivitis
12. Xanthelasma

The preprocessing pipeline creates leakage-controlled 70% / 15% / 15% train,
validation, and test splits. Images are RGB and resized to 224x224 pixels.

## 3. Model Architecture

The implementation is in `train_custom_cnn.py` as `EyeDiseaseCNN`.

| Block | Configuration |
|---|---|
| Input | 3-channel RGB, 224x224 |
| Convolution block 1 | Conv 3→32, BatchNorm, ReLU, MaxPool |
| Convolution block 2 | Conv 32→64, BatchNorm, ReLU, MaxPool |
| Convolution block 3 | Conv 64→128, BatchNorm, ReLU, MaxPool |
| Convolution block 4 | Conv 128→256, BatchNorm, ReLU, MaxPool |
| Convolution block 5 | Conv 256→256, BatchNorm, ReLU |
| Feature reduction | Adaptive average pooling |
| Classifier | Dropout 0.4, Linear 256→12 |

The model has approximately 984,012 trainable parameters.

## 4. Training Configuration

| Setting | Value |
|---|---|
| Training method | From scratch |
| Epochs | 25 |
| Batch size | 32 |
| Optimizer | AdamW |
| Initial learning rate | 0.001 |
| Minimum learning rate | 0.000001 |
| Weight decay | 0.0001 |
| Scheduler | CosineAnnealingLR |
| Random seed | 42 |
| Device | CUDA GPU when available, otherwise CPU |

### Data augmentation

Training images use:

- Random horizontal flip with probability 0.5
- Random rotation up to 15 degrees
- Brightness and contrast jitter of 0.1
- ImageNet normalization

Validation and test images use resizing and ImageNet normalization only.

### Class imbalance handling

The loss function is weighted using inverse class frequency:

```text
weight = total training images / (number of classes × class images)
```

This gives larger penalties to errors on minority classes such as Pinguecula,
Glaucoma, and Uveitis.

## 5. Hardware

Training is intended to run on:

- NVIDIA RTX 5070 GPU
- AMD Ryzen 9 9955HX CPU
- 32 GB RAM

With CUDA correctly detected, the 25-epoch run is expected to take
approximately 3–10 minutes, followed by a few minutes for evaluation and
Grad-CAM generation. Actual time depends on the installed PyTorch build,
drivers, and thermal or power limits.

## 6. How to Run

### Training Pipeline
To train the baseline CNN from scratch on the processed dataset:

```bash
python train_custom_cnn.py
```

The script automatically:
1. Loads the processed train, validation, and test splits (224×224 RGB).
2. Computes inverse-frequency class weights for cross-entropy loss.
3. Trains `EyeDiseaseCNN` for 25 epochs using AdamW + CosineAnnealingLR.
4. Checkpoints the best model based on validation macro-F1 (`models/custom_cnn_best.pt`).
5. Evaluates on the unseen test set (2,134 images), generating test reports and confusion matrices.
6. Generates initial Grad-CAM samples for test classes.

### Standalone Grad-CAM Visualizer
To generate or inspect Grad-CAM heatmaps without re-training:

```bash
# Generate sample heatmaps for all 12 classes
python gradcam_custom_cnn.py

# Explain a specific eye image
python gradcam_custom_cnn.py --image "datasets/processed/test/Cataract/Cataract_00000.jpg"
```

## 7. Generated Outputs

| Output | Location | Description |
|---|---|---|
| Best model checkpoint | `models/custom_cnn_best.pt` | Model weights with highest validation macro-F1 (Epoch 23) |
| Training metrics | `reports/custom_cnn_training_metrics.csv` | Epoch-by-epoch loss, accuracy, and F1 records |
| Learning curves | `reports/custom_cnn_learning_curves.png` | Train vs validation loss and accuracy curves |
| Test report | `reports/custom_cnn_test_report.txt` | Complete test set classification report |
| Confusion matrix | `reports/custom_cnn_confusion_matrix.png` | 12×12 normalized/raw test confusion matrix |
| Grad-CAM samples | `reports/gradcam_samples/gradcam_custom_cnn_*.png` | Heatmaps and overlays for each disease category |

## 8. Experimental Results

The model was trained for 25 epochs on an NVIDIA CUDA-enabled GPU (~32s per epoch, total training time ~14 minutes). The optimal checkpoint was attained at **Epoch 23** with a validation macro-F1 of **0.7458**.

### Overall Test Set Performance (2,134 Unseen Images)

| Metric | Score |
|---|---|
| **Test Accuracy** | **75.54%** |
| **Macro Precision** | **0.7431** |
| **Macro Recall** | **0.7703** |
| **Macro F1-Score** | **0.7511** |
| **Weighted F1-Score** | **0.7567** |

### Per-Class Performance Breakdown

| Disease Class | Precision | Recall | F1-Score | Support (Test Images) |
|---|---|---|---|---|
| **Pinguecula** | 0.9649 | 1.0000 | **0.9821** | 55 |
| **Xanthelasma** | 0.9333 | 0.9785 | **0.9554** | 186 |
| **Pterygium** | 0.9500 | 0.8837 | **0.9157** | 258 |
| **Cataract** | 0.8610 | 0.8136 | **0.8366** | 236 |
| **Healthy** | 0.8230 | 0.8037 | **0.8132** | 214 |
| **Blepharitis** | 0.6936 | 0.7617 | **0.7261** | 214 |
| **Bacterial Conjunctivitis** | 0.6894 | 0.7351 | **0.7115** | 151 |
| **Glaucoma** | 0.5773 | 0.8750 | **0.6957** | 64 |
| **Stye** | 0.7315 | 0.6295 | **0.6767** | 251 |
| **Viral Conjunctivitis** | 0.6479 | 0.6359 | **0.6419** | 217 |
| **Allergic Conjunctivitis** | 0.6471 | 0.5238 | **0.5789** | 210 |
| **Uveitis** | 0.3983 | 0.6026 | **0.4796** | 78 |

### Comparative Analysis: Custom CNN vs. DenseNet121

Both models were evaluated on the exact same 12-class test split (2,134 images):

| Model | Pretrained Weights? | Parameters | Test Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|---|---|
| **EyeDiseaseCNN (Custom Baseline)** | **None (From Scratch)** | **983,052** | **75.54%** | **0.7431** | **0.7703** | **0.7511** |
| DenseNet121 (Dev) | ImageNet-1k | 7,000,000+ | 92.88% | 0.9183 | 0.9276 | 0.9223 |

**Key Takeaways**:
1. **Strong Baseline Efficiency**: With under 1 million parameters (~14% the size of DenseNet121) and zero pretraining, `EyeDiseaseCNN` achieved over **75.5% test accuracy** and **0.7511 macro F1**.
2. **High Precision on Structural Conditions**: Conditions with distinctive visual landmarks such as *Pinguecula* (0.9821 F1), *Xanthelasma* (0.9554 F1), *Pterygium* (0.9157 F1), and *Cataract* (0.8366 F1) were classified with very high accuracy.
3. **Challenges with Redness / Conjunctival Overlap**: Similar to the transfer-learning models, differentiating between *Allergic*, *Viral*, and *Bacterial Conjunctivitis* remains the most ambiguous challenge due to subtle clinical symptom overlap in external eye photographs.

## 9. Explainability & Grad-CAM

Model interpretability is implemented via Gradient-weighted Class Activation Mapping (Grad-CAM):
- **Target Layer**: The final 256-channel convolutional layer (`model.features[-3]`).
- **Autograd Hook Safety**: Hook-safe implementation ensures non-destructive backward gradients across all activation layers.
- **Visual Validation**: Inspection of generated heatmaps in `reports/gradcam_samples/` confirms that the network focuses on physiological landmarks:
  - Cataract: Concentrated activation on the pupil and clouded lens.
  - Xanthelasma & Blepharitis: Activation focused directly on eyelid margins and lipid plaques.
  - Pterygium & Pinguecula: Sclera and corneal boundaries.

This project is for academic research and educational experimentation and is not a clinical medical diagnostic system.
