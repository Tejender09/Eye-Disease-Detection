"""Grad-CAM visualization for Custom EyeDiseaseCNN.

Loads a trained EyeDiseaseCNN checkpoint and generates Grad-CAM overlays:
  1. On all classes from a dataset directory (default: datasets/processed/test).
  2. On a single user-specified image via `--image <path>`.

Target layer:
  The final Conv2d layer in self.features (model.features[-3]).
  All ReLU modules have inplace=False set to prevent backward-hook autograd issues.
"""

import argparse
import os
import tempfile
from pathlib import Path

os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "mpl_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "custom_cnn_best.pt"
DEFAULT_TEST_DIR = PROJECT_ROOT / "datasets" / "processed" / "test"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "gradcam_samples"
IMAGE_SIZE = (224, 224)


class EyeDiseaseCNN(nn.Module):
    """Compact CNN baseline matching train_custom_cnn.py."""

    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
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


class GradCAM:
    """Robust Grad-CAM implementation for convolutional layers."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # Ensure all ReLUs are out-of-place to avoid autograd view+inplace conflicts
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
        self.model.zero_grad()
        output = self.model(input_tensor)
        probabilities = torch.softmax(output, dim=1).squeeze().detach().cpu().numpy()

        if target_class is None:
            target_class = int(output.argmax(dim=1).item())

        score = output[0, target_class]
        score.backward(retain_graph=False)

        # Global average pool of gradients across spatial dimensions
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = torch.clamp(cam, min=0).cpu().numpy()

        maximum = np.max(cam)
        if maximum > 0:
            cam = cam / maximum

        return cam, target_class, probabilities

    def close(self):
        for handle in self.handles:
            handle.remove()


def get_eval_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def save_gradcam_overlay(image_path, cam, predicted_name, true_name, confidence, output_path):
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
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    axes[0].imshow(image)
    axes[0].set_title(f"Original\n({true_name})" if true_name else "Original Image", fontsize=11)

    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Activation Heatmap\n(Last Conv Layer)", fontsize=11)

    axes[2].imshow(overlay)
    pred_title = f"Grad-CAM Overlay\nPred: {predicted_name}"
    if confidence is not None:
        pred_title += f" ({confidence * 100:.1f}%)"
    axes[2].set_title(pred_title, fontsize=11)

    for axis in axes:
        axis.axis("off")

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    classes = checkpoint["classes"]
    model = EyeDiseaseCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Ensure inplace is False everywhere
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False

    return model, classes


def process_single_image(image_path, model, classes, device, output_path, target_class=None, true_name=None):
    transform = get_eval_transform()
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Target last Conv2d layer (features[-3])
    target_layer = model.features[-3]
    grad_cam = GradCAM(model, target_layer)

    target_idx = None
    if target_class is not None:
        if isinstance(target_class, str):
            if target_class in classes:
                target_idx = classes.index(target_class)
            else:
                raise ValueError(f"Target class '{target_class}' not found in model classes: {classes}")
        else:
            target_idx = int(target_class)

    cam, pred_idx, probs = grad_cam.generate(input_tensor, target_class=target_idx)
    grad_cam.close()

    pred_name = classes[pred_idx]
    confidence = float(probs[pred_idx])

    save_gradcam_overlay(
        image_path=image_path,
        cam=cam,
        predicted_name=pred_name,
        true_name=true_name,
        confidence=confidence,
        output_path=output_path,
    )
    print(f"[Grad-CAM] Saved: {output_path.name} | Pred: {pred_name} ({confidence * 100:.2f}%)")
    return pred_name, confidence


def run_benchmark_dataset(model, classes, data_dir, output_dir, device):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating Grad-CAM overlays for each class in: {data_dir}")
    print(f"Output directory: {output_dir}\n")

    found_samples = {}
    for class_name in classes:
        class_folder = data_dir / class_name
        if class_folder.is_dir():
            jpg_files = sorted(class_folder.glob("*.jpg"))
            if jpg_files:
                found_samples[class_name] = jpg_files[0]

    if not found_samples:
        print(f"No sample images found under {data_dir}.")
        return

    for class_name, sample_path in found_samples.items():
        clean_name = class_name.replace(" ", "_")
        output_path = output_dir / f"gradcam_custom_cnn_{clean_name}.png"
        process_single_image(
            image_path=sample_path,
            model=model,
            classes=classes,
            device=device,
            output_path=output_path,
            true_name=class_name,
        )

    print(f"\nAll {len(found_samples)} Grad-CAM sample figures successfully saved to {output_dir}!\n")


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmaps for Custom EyeDiseaseCNN.")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT),
                        help="Path to trained model weights (.pt).")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_TEST_DIR),
                        help="Path to test dataset root containing class directories.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Directory to save generated Grad-CAM images.")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a single image to explain (optional).")
    parser.add_argument("--target-class", type=str, default=None,
                        help="Target class index or name to explain (default: argmax prediction).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on ('cuda' or 'cpu').")

    args = parser.parse_args()
    device = torch.device(args.device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print("=" * 60)
    print("  Custom EyeDiseaseCNN Grad-CAM Visualizer")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device    : {device}")
    print("=" * 60)

    model, classes = load_model(checkpoint_path, device)
    print(f"Loaded model with {len(classes)} classes.")

    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at: {image_path}")
        output_dir = Path(args.output_dir)
        output_path = output_dir / f"gradcam_custom_cnn_single_{image_path.stem}.png"
        process_single_image(
            image_path=image_path,
            model=model,
            classes=classes,
            device=device,
            output_path=output_path,
            target_class=args.target_class,
            true_name=image_path.parent.name if image_path.parent.name in classes else None,
        )
    else:
        run_benchmark_dataset(
            model=model,
            classes=classes,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            device=device,
        )


if __name__ == "__main__":
    main()
