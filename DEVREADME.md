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
5. Provide a separate, reusable Grad-CAM file for DenseNet121, so the integration step can call it later.

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
6. Moved Grad-CAM out of the training script into its own file, `densenet_gradcam.py`. The trained checkpoint was not changed and the model was not retrained.

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
| `Dev_train_densenet121.py` | Training and evaluation script (no Grad-CAM inside) |
| `densenet_gradcam.py` | Separate Grad-CAM file for DenseNet121 |
| `models/densenet121_best.pt` | Best model checkpoint (about 28 MB) |
| `reports/densenet121_training_metrics.csv` | Per-epoch training log |
| `reports/densenet121_learning_curves.png` | Loss and accuracy curves |
| `reports/densenet121_test_report.txt` | Test accuracy and per-class report |
| `reports/densenet121_confusion_matrix.png` | Confusion matrix |
| `reports/gradcam_samples/gradcam_densenet121_*.png` | Sample Grad-CAM images, one per class (made before the split) |

## 11. How to Run Training

```
python Dev_train_densenet121.py
```

I trained on a laptop with an NVIDIA RTX 3050 (6 GB). Warmup epochs took about 71 seconds each and fine-tuning epochs about 150 seconds each, so the full run took roughly 55 minutes. The script needs the dataset in `datasets/processed` and creates the `models` and `reports` outputs by itself.

Running it again will retrain the model and overwrite `models/densenet121_best.pt`, so only do that if you really want to retrain.

## 12. Grad-CAM (separate file)

### Why it was separated

Grad-CAM used to run at the end of the training script. I moved it into its own file so that:

- training and explainability are not mixed together,
- each teammate can keep a model-specific Grad-CAM file, and
- the integration teammate can import one function per model and combine them.

The trained model and checkpoint were not changed.

### Where it is

`densenet_gradcam.py` in the project root, next to `Dev_train_densenet121.py`.

### What it needs

The checkpoint `models/densenet121_best.pt`. It must contain `model_state_dict` and `classes`, which the training script saves.

### Functions

| Function | What it does |
|----------|--------------|
| `load_densenet121(checkpoint_path=..., device=None)` | Loads the trained model and returns `(model, class_names)` |
| `generate_densenet_gradcam(model, image, target_class=None)` | Returns `(cam, target_class)` |
| `make_overlay(image, cam)` | Returns a 224 x 224 RGB image with the heatmap on top |
| `save_gradcam_figure(image, cam, pred_name, save_path, true_name=None)` | Saves a 3-panel picture (original, heatmap, overlay) |

Parameters of `generate_densenet_gradcam`:

- `model`: the model from `load_densenet121()`
- `image`: an image path or a PIL image
- `target_class`: class index to explain. If left as `None`, it uses the predicted class.

It returns `cam` (a 7 x 7 numpy array scaled from 0 to 1) and the class index that was explained.

The function does not use hooks, because DenseNet121 applies an in-place ReLU after `norm5`, which breaks backward hooks.

### Example (from Python)

```python
from densenet_gradcam import load_densenet121, generate_densenet_gradcam, make_overlay

model, classes = load_densenet121()
cam, pred_idx = generate_densenet_gradcam(model, "path/to/eye_image.jpg")
print("Predicted:", classes[pred_idx])

overlay = make_overlay("path/to/eye_image.jpg", cam)   # RGB numpy image
```

### Example (from the command line)

```
python densenet_gradcam.py "path/to/eye_image.jpg" --out heatmap.png
```

### Expected output

- The terminal prints the predicted class name.
- A PNG is saved with three panels: the original image, the activation heatmap, and the overlay. Red or yellow areas are where the model looked most for its prediction.
- The result should look like the samples in `reports/gradcam_samples/`.

### Note for the integration teammate

Each model has its own Grad-CAM file with its own function. The integration file only needs to pick the model and call the matching function, for example `generate_densenet_gradcam(model, image)` for DenseNet121.