# Dev's Contribution — Eye Disease Classification Using Deep Learning

## 1. Introduction

This document describes my individual contribution to the Eye Disease Classification project.

My primary responsibility is developing, training, and experimenting with different deep learning architectures for image classification.

The objective is to compare different CNN architectures and investigate their ability to classify eye disease images.

**My assigned models:**
- ResNet50
- DenseNet121# Dev's Contribution — Eye Disease Classification Using Deep Learning

## 1. Introduction

This project is about classifying external eye images into 12 eye conditions using deep learning. It is a team project, and each team member trains a different model on the same dataset so the models can be compared fairly later.

The models in our team are:

- MobileNetV3-Large (Riya)
- EfficientNetV2-B0 (Rohit)
- **DenseNet121 (Dev, this document)**
- EfficientNetV2-S (Vidushi)
- EyeDiseaseCNN (Tejender)

My part is **DenseNet121**. I worked on my own branch (`dev-models`) and only pushed to `main` after my full task was done.

## 2. Project Objectives

1. Implement and train DenseNet121 for 12-class eye disease classification.
2. Keep the setup the same as Riya's MobileNetV3-Large pipeline, so all models can be compared fairly.
3. Handle the class imbalance in the dataset.
4. Evaluate the model on the unseen test set with accuracy, precision, recall, F1 and a confusion matrix.
5. Generate Grad-CAM heatmaps to see which part of the eye the model looks at.

## 3. Dataset

The dataset is already split into train, val and test folders (`datasets/processed`), roughly 70% / 15% / 15%.

| Split | Images |
|-------|--------|
| Train | 9,943 |
| Val   | 2,131 |
| Test  | 2,134 |

**12 classes:** Allergic Conjunctivitis, Bacterial Conjunctivitis, Blepharitis, Cataract, Glaucoma, Healthy, Pinguecula, Pterygium, Stye, Uveitis, Viral Conjunctivitis, Xanthelasma.

The data is imbalanced. Pinguecula (256 train images) and Glaucoma (295) have far fewer images than Pterygium (1,200) and Stye (1,171).

## 4. What I Did, Step by Step

1. Pulled Riya's work into my branch and read her `train_mobilenetv3.py` to copy her setup.
2. Wrote `train_densenet121.py` following the same conventions.
3. Found and fixed a problem in the Grad-CAM code. The usual hook-based method does not work on DenseNet121's `norm5` layer, because an in-place ReLU follows it. I replaced it with a hook-free version.
4. Ran a 1-epoch test of the whole pipeline first, to catch errors before the long run.
5. Trained the full model (23 epochs), then evaluated the best checkpoint on the test set.

## 5. Setup (kept the same as Riya's)

| Item | Value |
|------|-------|
| Framework | PyTorch |
| Image size | 224 x 224 |
| Batch size | 32 |
| Normalization | ImageNet mean and std |
| Train augmentation | Random horizontal flip, rotation up to 15 degrees, small brightness and contrast jitter |
| Val / Test | Resize only |
| Loss | CrossEntropyLoss with inverse-frequency class weights |
| Optimizer | AdamW, weight decay 1e-4 |
| Random seed | 42 |
| Best model chosen by | Validation macro F1 |

## 6. DenseNet121 Details

- Pretrained ImageNet weights from torchvision.
- The last layer was replaced with `Dropout(0.2)` + `Linear(1024 -> 12)`.
- **Phase 1 (3 epochs):** backbone frozen, only the new head trains, LR 1e-3.
- **Phase 2 (20 epochs):** whole model unfrozen, LR 1e-4 decaying to 1e-6 with cosine annealing.
- Total: 23 epochs.

## 7. Results

The best checkpoint came from epoch 23 (validation macro F1 = 0.9182).

**Test set:**

| Metric | Score |
|--------|-------|
| Accuracy | 92.88% |
| Macro Precision | 0.9183 |
| Macro Recall | 0.9276 |
| Macro F1 | 0.9223 |

**F1 score per class:**

| Class | F1 |
|-------|----|
| Pinguecula | 1.0000 |
| Xanthelasma | 1.0000 |
| Pterygium | 0.9864 |
| Stye | 0.9706 |
| Healthy | 0.9600 |
| Cataract | 0.9590 |
| Blepharitis | 0.9583 |
| Bacterial Conjunctivitis | 0.9024 |
| Glaucoma | 0.8489 |
| Uveitis | 0.8447 |
| Viral Conjunctivitis | 0.8266 |
| Allergic Conjunctivitis | 0.8104 |

## 8. What I Noticed

- **Strong classes:** Pinguecula and Xanthelasma were classified perfectly. Pinguecula is one of the smallest classes, so the class weights seem to have helped.
- **Main confusion:** the three conjunctivitis types get mixed up with each other. The biggest single error was 26 Viral Conjunctivitis images predicted as Allergic, followed by 18 Allergic images predicted as Viral. These conditions look similar in photos, so this is not surprising.
- **Cataract and Glaucoma** were also confused a little (14 Cataract images predicted as Glaucoma).
- **Small classes:** Glaucoma and Uveitis have only 64 and 78 test images, so their scores change a lot with just a few mistakes.
- **Overfitting:** train accuracy reached about 99%, while val and test accuracy were about 93%. Validation loss stayed flat and did not rise, and the test result matched the validation result, so the 93% looks like an honest number.

## 9. Limitations

- This is a student project on a public dataset. It is not meant for real medical use.
- I used the same setup as Riya on purpose, so I did not tune DenseNet121 further. Things like stronger augmentation, label smoothing or a larger image size might improve the weak classes, but I have not tried them.
- I have not yet compared my results with the other team members' models. I will add that once everyone's results are available.

## 10. Files

| File | What it is |
|------|-----------|
| `train_densenet121.py` | Full training and evaluation script |
| `models/densenet121_best.pt` | Best model checkpoint (about 28 MB) |
| `reports/densenet121_training_metrics.csv` | Per-epoch training log |
| `reports/densenet121_learning_curves.png` | Loss and accuracy curves |
| `reports/densenet121_test_report.txt` | Test accuracy and per-class report |
| `reports/densenet121_confusion_matrix.png` | Confusion matrix |
| `reports/gradcam_samples/gradcam_densenet121_*.png` | Grad-CAM images, one per class |

## 11. How to Run

```
python train_densenet121.py
```

I trained on a laptop with an NVIDIA RTX 3050 (6 GB). Warmup epochs took about 71 seconds each
- Custom CNN
- Vision Transformer (optional)

## 2. Project Objectives

1. Implement different deep learning architectures.
2. Train and validate models using the prepared dataset.
3. Experiment with hyperparameters and optimization techniques.
4. Monitor training and validation performance.
5. Identify and address overfitting.
6. Save the best-performing model checkpoints.
7. Share model checkpoints and training results with the evaluation team.
8. Contribute to the final model comparison.

## 3. Model Selection

Three main approaches are being investigated:

| Model | Architecture Type | Main Concept |
|---|---|---|
| Custom CNN | CNN from scratch | Baseline feature learning |
| ResNet50 | Deep CNN | Residual connections |
| DenseNet121 | Deep CNN | Dense connectivity |
| Vision Transformer | Transformer | Self-attention |

### Why these models?

**Custom CNN:** Establishes a baseline and allows experimentation with a network trained from scratch.

**ResNet50:** Uses residual connections to improve information and gradient flow in deeper networks.

**DenseNet121:** Uses dense connections to encourage feature reuse and improve information propagation.

**Vision Transformer:** Provides an opportunity to compare convolution-based learning with attention-based image processing.

## 4. ResNet50

### Overview

ResNet50 is a 50-layer residual convolutional neural network.

Its key feature is the residual or skip connection.

The residual learning equation is:

y = F(x) + x

Where:
- x represents the original input.
- F(x) represents the learned residual mapping.
- y represents the output.

### Implementation Plan

1. Load the pretrained ResNet50 architecture.
2. Replace the original classification layer with a new classifier matching the dataset classes.
3. Configure the optimizer and loss function.
4. Train the model using the training dataset.
5. Monitor validation loss and accuracy.
6. Save the best checkpoint.

### Experiment Details

- Architecture: ResNet50
- Pretrained weights: To be configured
- Classification classes: To match the finalized dataset
- Optimizer: To be finalized
- Loss function: CrossEntropyLoss

## 5. DenseNet121

### Overview

DenseNet121 is a convolutional neural network that uses dense connectivity.

Each layer receives feature maps from preceding layers within its dense block.

### Key Advantages

- Feature reuse.
- Improved information flow.
- Better gradient propagation.
- Efficient feature representation.

### Implementation Plan

1. Load DenseNet121.
2. Replace its classifier with a task-specific classification head.
3. Configure training parameters.
4. Train using the prepared dataset.
5. Monitor validation performance.
6. Save the best checkpoint.

## 6. Custom CNN

### Overview

A custom CNN is developed from scratch without relying on pretrained ImageNet weights.

The purpose is to establish a baseline for comparison against pretrained architectures.

### Planned Architecture

- Input image: 224 × 224 × 3
- Five convolutional blocks
- Batch Normalization
- ReLU activation
- Max Pooling
- Global Average Pooling
- Fully Connected Layer
- Dropout
- Final Classification Layer

### Objective

To understand how a CNN learns image features directly from the available training dataset.

## 7. Vision Transformer (Optional)

Vision Transformer uses image patches and self-attention mechanisms rather than relying primarily on traditional convolution operations.

The experiment will be considered depending on available time and GPU resources.

## 8. Training Methodology

The models will use a common experimental dataset and fixed train, validation and test splits.

The training process includes:

1. Input image preprocessing.
2. Forward propagation.
3. Loss calculation.
4. Backpropagation.
5. Optimizer updates.
6. Validation after training epochs.
7. Best-checkpoint selection.

### Evaluation During Training

- Training loss
- Validation loss
- Training accuracy
- Validation accuracy

## 9. Model Evaluation

The trained models will be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Inference Time
- Model Parameters

Macro-averaged metrics will also be considered to understand performance across all classes.

## 10. Challenges and Improvements

This section will document the issues encountered during implementation.

Potential areas of investigation:

- Overfitting.
- Learning-rate selection.
- GPU memory limitations.
- Training time.
- Class-wise performance differences.
- Checkpoint management.

Actual challenges and solutions will be added as experiments progress.

## 11. Experimental Results

| Model | Best Validation Accuracy | Test Accuracy | Status |
|---|---|---|---|
| ResNet50 | Pending | Pending | Not yet documented |
| DenseNet121 | Pending | Pending | Not yet documented |
| Custom CNN | Pending | Pending | Not yet documented |
| Vision Transformer | Pending | Pending | Optional |

Results will be updated using actual experimental outputs.

## 12. Individual Contribution

My responsibilities include:

- Developing ResNet50.
- Developing DenseNet121.
- Building a Custom CNN baseline.
- Exploring Vision Transformer if resources permit.
- Managing model training and checkpoints.
- Recording experimental configurations.
- Analyzing training and validation behavior.
- Sharing results with the evaluation team.

## 13. Conclusion

My contribution focuses on investigating different deep learning architectures for eye disease image classification.

By comparing a custom CNN, residual networks, dense networks and potentially transformer-based architectures, the project aims to identify a suitable model based on classification performance and computational efficiency.

All conclusions will be based on experimental evaluation using the finalized dataset and independent test set.