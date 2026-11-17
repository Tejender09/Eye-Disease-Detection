# Dev's Contribution — Eye Disease Classification Using Deep Learning

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
2. Wrote `Dev_train_densenet121.py` following the same conventions.
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
| `Dev_train_densenet121.py` | Full training and evaluation script |
| `models/densenet121_best.pt` | Best model checkpoint (about 28 MB) |
| `reports/densenet121_training_metrics.csv` | Per-epoch training log |
| `reports/densenet121_learning_curves.png` | Loss and accuracy curves |
| `reports/densenet121_test_report.txt` | Test accuracy and per-class report |
| `reports/densenet121_confusion_matrix.png` | Confusion matrix |
| `reports/gradcam_samples/gradcam_densenet121_*.png` | Grad-CAM images, one per class |

## 11. How to Run

```
python Dev_train_densenet121.py
```

I trained on a laptop with an NVIDIA RTX 3050 (6 GB). Warmup epochs took about 71 seconds each and fine-tuning epochs about 150 seconds each, so the full run took roughly 55 minutes. The script needs the dataset in `datasets/processed` and creates the `models` and `reports` outputs by itself.