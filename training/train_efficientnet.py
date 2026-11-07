
import sys
import csv
import random
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from contextlib import nullcontext
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.efficientnetb0 import build_efficientnetb0


# ==============================
# 1. CONFIGURATION
# ==============================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DATA_DIR = ROOT / "datasets" / "final" / "external_eye"
RUN_DIR = ROOT / "runs" / "efficientnetb0"

RUN_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 0.0001
PATIENCE = 6

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_AMP = DEVICE.type == "cuda"

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ==============================
# 2. IMAGE TRANSFORMATIONS
# ==============================

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=8),
    transforms.ColorJitter(
        brightness=0.08,
        contrast=0.08
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==============================
# 3. LOAD DATASET
# ==============================

train_dataset = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    DATA_DIR / "val",
    transform=val_transform
)

test_dataset = datasets.ImageFolder(
    DATA_DIR / "test"
)

if train_dataset.class_to_idx != val_dataset.class_to_idx:
    raise ValueError("Training and validation classes do not match.")

if train_dataset.class_to_idx != test_dataset.class_to_idx:
    raise ValueError("Training and testing classes do not match.")

classes = train_dataset.classes
num_classes = len(classes)

print("\nClasses:", classes)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Testing images:", len(test_dataset))

if num_classes != 5:
    raise ValueError("Expected exactly five classes.")


# ==============================
# 4. DATA LOADERS
# ==============================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=USE_AMP
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=USE_AMP
)


# ==============================
# 5. CLASS WEIGHTS
# ==============================

class_counts = torch.bincount(
    torch.tensor(train_dataset.targets),
    minlength=num_classes
)

class_weights = (
    len(train_dataset) /
    (num_classes * class_counts.float())
)

class_weights = class_weights.to(DEVICE)

print("\nClass distribution:")

for name, count, weight in zip(
    classes,
    class_counts.tolist(),
    class_weights.tolist()
):
    print(f"{name}: {count} images, weight={weight:.4f}")


# ==============================
# 6. BUILD MODEL
# ==============================

model = build_efficientnetb0(
    num_classes=num_classes,
    pretrained=True
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0001
)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=USE_AMP
)


# ==============================
# 7. TRAINING FUNCTION
# ==============================

def train_one_epoch():

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        context = (
            torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16
            )
            if USE_AMP else nullcontext()
        )

        with context:

            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


# ==============================
# 8. VALIDATION FUNCTION
# ==============================

def validate():

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            context = (
                torch.amp.autocast(
                    device_type="cuda",
                    dtype=torch.float16
                )
                if USE_AMP else nullcontext()
            )

            with context:

                outputs = model(images)
                loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


# ==============================
# 9. MODEL TRAINING
# ==============================

best_val_loss = float("inf")
early_stop_counter = 0

history = []

print("\nStarting EfficientNet-B0 training...\n")

for epoch in range(EPOCHS):

    train_loss, train_acc = train_one_epoch()

    val_loss, val_acc = validate()

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_acc:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_acc:.4f} "
        f"LR: {current_lr:.7f}"
    )

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "learning_rate": current_lr
    })

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        early_stop_counter = 0

        torch.save({
            "model_name": "EfficientNet-B0",
            "model_state_dict": model.state_dict(),
            "classes": classes,
            "class_to_idx": train_dataset.class_to_idx,
            "epoch": epoch + 1,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "image_size": IMAGE_SIZE
        }, RUN_DIR / "best_model.pth")

        print("Best model saved!")

    else:

        early_stop_counter += 1

    if early_stop_counter >= PATIENCE:

        print("\nEarly stopping triggered.")
        break


# ==============================
# 10. SAVE TRAINING HISTORY
# ==============================

with open(
    RUN_DIR / "training_history.csv",
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=history[0].keys()
    )

    writer.writeheader()
    writer.writerows(history)


print("\nTraining completed!")
print("Best validation loss:", best_val_loss)
print("Model saved at:", RUN_DIR / "best_model.pth")
print("History saved at:", RUN_DIR / "training_history.csv")
