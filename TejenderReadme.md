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

From the project root:

```bash
python train_custom_cnn.py
```

The script automatically:

1. Loads the processed train, validation, and test folders.
2. Discovers the class names.
3. Computes class weights.
4. Trains the CNN for 25 epochs.
5. Saves the checkpoint with the best validation macro-F1.
6. Evaluates the best checkpoint on the unseen test set.
7. Generates reports, a confusion matrix, learning curves, and Grad-CAM
   visualizations.

The console should show `cuda` as the selected device when the GPU is active.

## 7. Generated Outputs

| Output | Location |
|---|---|
| Best model checkpoint | `models/custom_cnn_best.pt` |
| Training metrics | `reports/custom_cnn_training_metrics.csv` |
| Learning curves | `reports/custom_cnn_learning_curves.png` |
| Test report | `reports/custom_cnn_test_report.txt` |
| Confusion matrix | `reports/custom_cnn_confusion_matrix.png` |
| Grad-CAM samples | `reports/gradcam_samples/gradcam_custom_cnn_*.png` |

The checkpoint is selected using validation macro-F1 rather than training
accuracy, which gives minority classes more influence during model selection.

## 8. Evaluation Metrics

The test evaluation reports:

- Overall accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Per-class precision, recall, and F1
- Confusion matrix

Final custom CNN scores will be added here after the first complete training
run. Results should be compared with the existing MobileNetV3-Large and
DenseNet121 models using the same processed test split and evaluation metrics.

## 9. Explainability

Grad-CAM is generated from the final convolutional activation layer. The
heatmaps show which image regions influenced each prediction and help check
whether the model focuses on ocular features instead of irrelevant background
regions.

This project is for academic experimentation and is not a medical diagnostic
system.
