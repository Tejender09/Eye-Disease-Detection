import csv
import argparse
import os
import tempfile
import time
from pathlib import Path

os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "mpl_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


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
BATCH_SIZE = 32  # Batch size
NUM_WORKERS = 0
RANDOM_SEED = 42
EPOCHS = 25
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class EyeDataset(Dataset):
    """Load RGB images and labels from class-named folders."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = sorted(
            directory.name
            for directory in self.root_dir.iterdir()
            if directory.is_dir()
        )
        self.class_to_idx = {
            class_name: index
            for index, class_name in enumerate(self.classes)
        }
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            for image_path in sorted(class_dir.glob("*.jpg")):
                self.samples.append(
                    (str(image_path), self.class_to_idx[class_name])
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label, image_path


def get_data_transforms():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_transform, eval_transform


class EyeDiseaseCNN(nn.Module):
    """Compact CNN baseline trained from scratch for the 12 disease classes."""

    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs):
        features = self.features(inputs)
        return self.classifier(self.pool(features))


def compute_class_weights(dataset):
    counts = np.zeros(len(dataset.classes), dtype=np.float32)
    for _, label in dataset.samples:
        counts[label] += 1
    weights = len(dataset) / (len(dataset.classes) * counts)
    for class_name, weight in zip(dataset.classes, weights):
        print(f" - {class_name:25s}: weight = {weight:.3f}")
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


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
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            running_loss += criterion(outputs, labels).item() * images.size(0)
            all_predictions.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    labels_array = np.array(all_labels)
    predictions_array = np.array(all_predictions)
    return (
        running_loss / len(all_labels),
        np.mean(labels_array == predictions_array),
        f1_score(labels_array, predictions_array, average="macro", zero_division=0),
        recall_score(
            labels_array, predictions_array, average="macro", zero_division=0
        ),
        all_labels,
        all_predictions,
    )


class GradCAM:
    """Grad-CAM implementation for the final convolutional activation map."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
        self.handles = [
            target_layer.register_forward_hook(self._forward_hook),
            target_layer.register_full_backward_hook(self._backward_hook),
        ]

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_inputs, grad_outputs):
        self.gradients = grad_outputs[0].detach()

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, target_class].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = torch.clamp(cam, min=0).cpu().numpy()
        maximum = np.max(cam)
        if maximum > 0:
            cam = cam / maximum
        return cam, target_class

    def close(self):
        for handle in self.handles:
            handle.remove()


def save_gradcam_overlay(image_path, cam, predicted_name, true_name, output_path):
    image = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    cam_image = Image.fromarray(np.uint8(255 * cam)).resize(
        IMAGE_SIZE, Image.Resampling.BILINEAR
    )
    heatmap = plt.cm.jet(np.array(cam_image) / 255.0)[:, :, :3]
    overlay = np.clip(
        0.5 * (np.array(image) / 255.0) + 0.5 * heatmap,
        0,
        1,
    )
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title(f"Original: {true_name}")
    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Activation Heatmap")
    axes[2].imshow(overlay)
    axes[2].set_title(f"Grad-CAM: Pred={predicted_name}")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_training_curves(history):
    epochs = [item["epoch"] for item in history]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, [item["train_loss"] for item in history], label="Train")
    axes[0].plot(epochs, [item["val_loss"] for item in history], label="Validation")
    axes[0].set_title("Custom CNN Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[1].plot(
        epochs,
        [item["train_acc"] * 100 for item in history],
        label="Train",
    )
    axes[1].plot(
        epochs,
        [item["val_acc"] * 100 for item in history],
        label="Validation",
    )
    axes[1].set_title("Custom CNN Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(REPORTS_DIR / "custom_cnn_learning_curves.png", dpi=150)
    plt.close(figure)


def generate_gradcam_samples(model, test_dataset, eval_transform):
    grad_cam = GradCAM(model, model.features[-3])
    samples_by_class = {}
    for image_path, label in test_dataset.samples:
        samples_by_class.setdefault(label, image_path)
    for class_index, image_path in samples_by_class.items():
        image = Image.open(image_path).convert("RGB")
        input_tensor = eval_transform(image).unsqueeze(0).to(DEVICE)
        cam, predicted_index = grad_cam.generate(input_tensor)
        output_path = GRADCAM_DIR / (
            f"gradcam_custom_cnn_{test_dataset.classes[class_index].replace(' ', '_')}.png"
        )
        save_gradcam_overlay(
            image_path,
            cam,
            test_dataset.classes[predicted_index],
            test_dataset.classes[class_index],
            output_path,
        )
    grad_cam.close()


def main(resume=False):
    print("=" * 60)
    print("  Custom EyeDiseaseCNN Training Pipeline")
    print(f"  Device: {DEVICE}")
    print("=" * 60)
    train_transform, eval_transform = get_data_transforms()
    train_dataset = EyeDataset(TRAIN_DIR, train_transform)
    val_dataset = EyeDataset(VAL_DIR, eval_transform)
    test_dataset = EyeDataset(TEST_DIR, eval_transform)
    if train_dataset.classes != val_dataset.classes or train_dataset.classes != test_dataset.classes:
        raise ValueError("Train, validation, and test classes do not match.")
    print(
        f"Dataset: train={len(train_dataset)}, val={len(val_dataset)}, "
        f"test={len(test_dataset)}, classes={len(train_dataset.classes)}"
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    class_weights = compute_class_weights(train_dataset)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    model = EyeDiseaseCNN(len(train_dataset.classes)).to(DEVICE)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Trainable parameters: {parameter_count:,}")
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )
    best_val_f1 = -1.0
    best_model_path = MODELS_DIR / "custom_cnn_best.pt"
    resume_model_path = MODELS_DIR / "custom_cnn_resume.pt"
    history = []
    start_epoch = 1
    if resume:
        checkpoint_path = (
            resume_model_path if resume_model_path.exists() else best_model_path
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Cannot resume: checkpoint not found at {checkpoint_path}"
            )
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_val_f1 = checkpoint.get("val_f1", best_val_f1)
        history = checkpoint.get("history", [])
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"Resuming from epoch {start_epoch} using {checkpoint_path}")
    for epoch in range(start_epoch, EPOCHS + 1):
        start_time = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer
        )
        val_loss, val_acc, val_f1, val_recall, _, _ = evaluate(
            model, val_loader, criterion
        )
        scheduler.step()
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "val_recall": val_recall,
        })
        elapsed = time.time() - start_time
        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] | {elapsed:.1f}s | "
            f"Train loss={train_loss:.4f}, acc={train_acc * 100:.2f}% | "
            f"Val loss={val_loss:.4f}, acc={val_acc * 100:.2f}%, "
            f"F1={val_f1:.4f}, recall={val_recall:.4f}"
        )
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": train_dataset.classes,
                "val_f1": val_f1,
                "epoch": epoch,
            }, best_model_path)
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "classes": train_dataset.classes,
            "val_f1": best_val_f1,
            "epoch": epoch,
            "history": history,
        }, resume_model_path)
    metrics_path = REPORTS_DIR / "custom_cnn_training_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    save_training_curves(history)
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, test_acc, test_f1, test_recall, y_true, y_pred = evaluate(
        model, test_loader, criterion
    )
    test_precision = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    report = classification_report(
        y_true, y_pred, target_names=train_dataset.classes, digits=4
    )
    report_path = REPORTS_DIR / "custom_cnn_test_report.txt"
    with report_path.open("w", encoding="utf-8") as file:
        file.write(f"Test Accuracy: {test_acc * 100:.2f}%\n")
        file.write(f"Macro Precision: {test_precision:.4f}\n")
        file.write(f"Macro Recall: {test_recall:.4f}\n")
        file.write(f"Macro F1: {test_f1:.4f}\n\n")
        file.write(report)
    confusion = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=train_dataset.classes,
        yticklabels=train_dataset.classes,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Custom CNN Test Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "custom_cnn_confusion_matrix.png", dpi=150)
    plt.close()
    generate_gradcam_samples(model, test_dataset, eval_transform)
    print(f"Test accuracy: {test_acc * 100:.2f}%")
    print(f"Test macro F1: {test_f1:.4f}")
    print(f"Best checkpoint: {best_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the custom eye disease CNN.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest custom CNN checkpoint.",
    )
    main(resume=parser.parse_args().resume)
