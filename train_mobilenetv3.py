import os
from pathlib import Path
import tempfile
os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "mpl_cache")

import sys
import time
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models import MobileNet_V3_Large_Weights

from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, precision_score

# ---------------- CONFIGURATION ----------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "datasets" / "processed"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
GRADCAM_DIR = REPORTS_DIR / "gradcam_samples"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_WORKERS = 0  # Safe default on Windows
RANDOM_SEED = 42

# Training Hyperparameters
WARMUP_EPOCHS = 3
FINE_TUNE_EPOCHS = 20
TOTAL_EPOCHS = WARMUP_EPOCHS + FINE_TUNE_EPOCHS

WARMUP_LR = 1e-3
FINE_TUNE_LR = 1e-4
WEIGHT_DECAY = 1e-4

# Deterministic seeding
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- DATASET & TRANSFORMS ----------------

class EyeDataset(Dataset):
    """Custom dataset loading 224x224 RGB images with assigned class labels."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_folder = self.root_dir / class_name
            class_idx = self.class_to_idx[class_name]
            for img_path in class_folder.glob("*.jpg"):
                self.samples.append((str(img_path), class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, path

def get_data_transforms():
    # Normalization according to ImageNet standard
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    return train_transform, eval_transform

# ---------------- CLASS WEIGHT COMPUTATION ----------------

def compute_class_weights(train_dataset):
    """Compute inverse frequency class weights to penalize minority misclassifications."""
    class_counts = np.zeros(len(train_dataset.classes), dtype=np.float32)
    for _, label in train_dataset.samples:
        class_counts[label] += 1
    
    total_samples = len(train_dataset)
    num_classes = len(train_dataset.classes)
    
    # Balanced weighting: total / (num_classes * count)
    weights = total_samples / (num_classes * class_counts)
    
    print("\nCalculated Class Weights for Training:")
    for c_name, w in zip(train_dataset.classes, weights):
        print(f" - {c_name:25s}: weight = {w:.3f}")
    
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)

# ---------------- MODEL CREATION (MobileNetV3-Large) ----------------

def build_mobilenetv3(num_classes):
    """Load pretrained MobileNetV3-Large and adapt head for multi-class classification."""
    weights = MobileNet_V3_Large_Weights.DEFAULT
    model = models.mobilenet_v3_large(weights=weights)
    
    # The default classifier in mobilenet_v3_large:
    # (0): Linear(in_features=960, out_features=1280, bias=True)
    # (1): Hardswish()
    # (2): Dropout(p=0.2, inplace=True)
    # (3): Linear(in_features=1280, out_features=1000, bias=True)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    
    return model

def freeze_backbone(model):
    """Freeze all feature extraction layers for Phase 1 (Warmup)."""
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

def unfreeze_backbone(model):
    """Unfreeze all layers for Phase 2 (Fine-Tuning)."""
    for param in model.parameters():
        param.requires_grad = True

# ---------------- GRAD-CAM EXPLAINABILITY ----------------

class GradCAM:
    """Gradient-weighted Class Activation Mapping for MobileNetV3."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_heatmap(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
            
        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()

        # Global average pool the gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        # Weighted combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = torch.clamp(cam, min=0).cpu().numpy()
        
        # Normalize between 0 and 1
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        return cam, target_class

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

def save_gradcam_overlay(orig_img_path, cam, pred_class_name, true_class_name, save_path):
    """Overlay Grad-CAM heatmap on original image and save figure."""
    orig_img = Image.open(orig_img_path).convert("RGB").resize(IMAGE_SIZE)
    cam_resized = Image.fromarray(np.uint8(255 * cam)).resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    heatmap = plt.cm.jet(np.array(cam_resized) / 255.0)[:, :, :3]
    
    overlay = 0.5 * (np.array(orig_img) / 255.0) + 0.5 * heatmap
    overlay = np.clip(overlay, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(orig_img)
    axes[0].set_title(f"Original: {true_class_name}", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title(f"Activation Heatmap", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Grad-CAM: Pred={pred_class_name}", fontsize=10)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

# ---------------- TRAINING & EVALUATION FUNCTIONS ----------------

def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in dataloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    total = len(all_labels)
    epoch_loss = running_loss / total
    epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    return epoch_loss, epoch_acc, macro_f1, macro_recall, all_labels, all_preds

# ---------------- MAIN TRAINING PIPELINE ----------------

def main():
    print(f"============================================================")
    print(f"  MobileNetV3-Large Training Pipeline (Riya's Contribution) ")
    print(f"  Target Device: {DEVICE}")
    print(f"============================================================\n")

    train_transform, eval_transform = get_data_transforms()

    train_dataset = EyeDataset(TRAIN_DIR, transform=train_transform)
    val_dataset = EyeDataset(VAL_DIR, transform=eval_transform)
    test_dataset = EyeDataset(TEST_DIR, transform=eval_transform)

    print(f"Dataset summary:")
    print(f" - Train images: {len(train_dataset)}")
    print(f" - Validation images: {len(val_dataset)}")
    print(f" - Test images: {len(test_dataset)}")
    print(f" - Number of classes: {len(train_dataset.classes)}\n")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Calculate class-weighted loss
    class_weights = compute_class_weights(train_dataset)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Build model
    model = build_mobilenetv3(num_classes=len(train_dataset.classes)).to(DEVICE)

    # ---------------- PHASE 1: WARMUP (3 EPOCHS) ----------------
    print("\n------------------------------------------------------------")
    print(f"PHASE 1: Warmup Training ({WARMUP_EPOCHS} Epochs - Backbone Frozen)")
    print("------------------------------------------------------------")

    freeze_backbone(model)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=WARMUP_LR, weight_decay=WEIGHT_DECAY)

    history = []
    best_val_f1 = 0.0
    best_model_path = MODELS_DIR / "mobilenetv3_large_best.pt"

    for epoch in range(1, WARMUP_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_f1, val_recall, _, _ = evaluate(model, val_loader, criterion)
        elapsed = time.time() - t0

        print(f"Epoch [{epoch:02d}/{TOTAL_EPOCHS}] (Warmup) | Time: {elapsed:.1f}s | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%, F1: {val_f1:.4f}, Recall: {val_recall:.4f}")

        history.append({
            "epoch": epoch,
            "phase": "Warmup",
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "val_recall": val_recall
        })

    # ---------------- PHASE 2: FINE-TUNING (20 EPOCHS) ----------------
    print("\n------------------------------------------------------------")
    print(f"PHASE 2: Fine-Tuning ({FINE_TUNE_EPOCHS} Epochs - Full Backbone Unfrozen)")
    print("------------------------------------------------------------")

    unfreeze_backbone(model)
    optimizer = optim.AdamW(model.parameters(), lr=FINE_TUNE_LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINE_TUNE_EPOCHS, eta_min=1e-6)

    for epoch in range(WARMUP_EPOCHS + 1, TOTAL_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_f1, val_recall, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()
        elapsed = time.time() - t0

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch:02d}/{TOTAL_EPOCHS}] (Fine-Tune) [lr: {current_lr:.1e}] | Time: {elapsed:.1f}s | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%, F1: {val_f1:.4f}, Recall: {val_recall:.4f}")

        history.append({
            "epoch": epoch,
            "phase": "Fine-Tune",
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "val_recall": val_recall
        })

        # Checkpoint based on Validation Macro F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": train_dataset.classes,
                "val_f1": val_f1,
                "epoch": epoch
            }, best_model_path)
            print(f" --> Best checkpoint saved to {best_model_path.name} (Val F1: {val_f1:.4f})")

    # ---------------- SAVE TRAINING LOGS & PLOTS ----------------
    metrics_csv = REPORTS_DIR / "mobilenetv3_training_metrics.csv"
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f"\nTraining history saved to {metrics_csv}")

    # Plot training curves
    epochs = [h["epoch"] for h in history]
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [h["train_loss"] for h in history], label="Train Loss", color="royalblue")
    plt.plot(epochs, [h["val_loss"] for h in history], label="Val Loss", color="firebrick")
    plt.axvline(x=WARMUP_EPOCHS, color="gray", linestyle="--", label="Warmup End")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MobileNetV3-Large: Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, [h["train_acc"]*100 for h in history], label="Train Acc", color="royalblue")
    plt.plot(epochs, [h["val_acc"]*100 for h in history], label="Val Acc", color="firebrick")
    plt.axvline(x=WARMUP_EPOCHS, color="gray", linestyle="--", label="Warmup End")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("MobileNetV3-Large: Accuracy Curve")
    plt.legend()

    plt.tight_layout()
    curves_path = REPORTS_DIR / "mobilenetv3_learning_curves.png"
    plt.savefig(curves_path, dpi=150)
    plt.close()
    print(f"Learning curves plot saved to {curves_path}")

    # ---------------- TEST SET EVALUATION ----------------
    print("\n============================================================")
    print("  Evaluating Best Model Checkpoint on Unseen Test Set")
    print("============================================================\n")

    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc, test_f1, test_recall, y_true, y_pred = evaluate(model, test_loader, criterion)

    print(f"Test Set Results:")
    print(f" - Overall Accuracy: {test_acc*100:.2f}%")
    print(f" - Macro Precision:  {precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f" - Macro Recall:     {test_recall:.4f}")
    print(f" - Macro F1-Score:   {test_f1:.4f}\n")

    cls_report = classification_report(y_true, y_pred, target_names=train_dataset.classes, digits=4)
    print("Classification Report:\n" + cls_report)

    with open(REPORTS_DIR / "mobilenetv3_test_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Test Accuracy: {test_acc*100:.2f}%\nMacro F1: {test_f1:.4f}\n\n")
        f.write(cls_report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("MobileNetV3-Large Test Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    cm_path = REPORTS_DIR / "mobilenetv3_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    # ---------------- GRAD-CAM EXPLAINABILITY SAMPLES ----------------
    print("\nGenerating Grad-CAM explainability heatmaps for sample test images...")
    grad_cam = GradCAM(model, target_layer=model.features[-1])

    # Sample 1 image from each class in test set
    samples_by_class = {}
    for path, label in test_dataset.samples:
        if label not in samples_by_class:
            samples_by_class[label] = path
        if len(samples_by_class) == len(train_dataset.classes):
            break

    for class_idx, sample_path in samples_by_class.items():
        class_name = train_dataset.classes[class_idx]
        raw_img = Image.open(sample_path).convert("RGB")
        input_tensor = eval_transform(raw_img).unsqueeze(0).to(DEVICE)

        cam, pred_idx = grad_cam.generate_heatmap(input_tensor)
        pred_name = train_dataset.classes[pred_idx]

        save_name = f"gradcam_{class_name.replace(' ', '_')}.png"
        save_path = GRADCAM_DIR / save_name
        save_gradcam_overlay(sample_path, cam, pred_name, class_name, save_path)

    grad_cam.remove_hooks()
    print(f"Grad-CAM samples saved to {GRADCAM_DIR}\n")
    print("MobileNetV3-Large training & evaluation pipeline completed successfully!")

if __name__ == "__main__":
    main()

