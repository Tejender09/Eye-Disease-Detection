# Dev's Contribution — Eye Disease Classification Using Deep Learning

## 1. Introduction

This document describes my individual contribution to the Eye Disease Classification project.

My primary responsibility is developing, training, and experimenting with different deep learning architectures for image classification.

The objective is to compare different CNN architectures and investigate their ability to classify eye disease images.

**My assigned models:**
- ResNet50
- DenseNet121
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