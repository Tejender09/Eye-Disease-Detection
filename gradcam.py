"""gradcam.py
==========
Centralized Grad-CAM and Diagnostic Engine for Eye Disease Detection.
Supports all 4 project architectures:
  1. custom_cnn
  2. mobilenetv3
  3. densenet121
  4. efficientnetv2_s
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
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
GRADCAM_DIR = REPORTS_DIR / "gradcam_samples"
TEST_DATASET_DIR = PROJECT_ROOT / "datasets" / "processed" / "test"

DEFAULT_CHECKPOINTS = {
    "custom_cnn": MODELS_DIR / "custom_cnn_best.pt",
    "mobilenetv3": MODELS_DIR / "mobilenetv3_large_best.pt",
    "densenet121": MODELS_DIR / "densenet121_best.pt",
    "efficientnetv2_s": MODELS_DIR / "efficientnetv2_s_best.pt",
}

MODEL_DISPLAY_NAMES = {
    "custom_cnn": "Custom EyeDiseaseCNN",
    "mobilenetv3": "MobileNetV3-Large",
    "densenet121": "DenseNet121",
    "efficientnetv2_s": "EfficientNetV2-S",
}

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ==============================================================================
# MODEL ARCHITECTURES & LOADERS
# ==============================================================================

class EyeDiseaseCNN(nn.Module):
    """Compact CNN baseline matching train_custom_cnn.py."""
    def __init__(self, num_classes=12):
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

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def load_model(model_name: str, checkpoint_path=None, device=None):
    """Universal loader that initializes any model and loads trained weights."""
    model_name = model_name.lower()
    if model_name not in DEFAULT_CHECKPOINTS:
        raise ValueError(f"Unknown model: {model_name}. Supported: {list(DEFAULT_CHECKPOINTS.keys())}")

    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CHECKPOINTS[model_name]
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    classes = ckpt["classes"]
    num_classes = len(classes)

    if model_name == "custom_cnn":
        model = EyeDiseaseCNN(num_classes=num_classes)
    elif model_name == "mobilenetv3":
        model = models.mobilenet_v3_large(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    elif model_name == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes),
        )
    elif model_name == "efficientnetv2_s":
        model = models.efficientnet_v2_s(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    for mod in model.modules():
        if hasattr(mod, "inplace"):
            mod.inplace = False

    return model, classes


# ==============================================================================
# GRAD-CAM GENERATION ENGINES
# ==============================================================================

def generate_gradcam(model, model_name: str, image_or_path, target_class=None, device=None):
    """Universal Grad-CAM generator.

    Returns:
    --------
    cam : np.ndarray (224, 224) normalized in [0, 1]
    pred_class_idx : int
    probabilities : np.ndarray (num_classes,)
    overlay : np.ndarray (224, 224, 3) float32 in [0, 1]
    raw_image : PIL.Image (224, 224)
    """
    model_name = model_name.lower()
    if device is None:
        device = next(model.parameters()).device

    if isinstance(image_or_path, (str, Path)):
        raw_image = Image.open(image_or_path).convert("RGB")
    elif isinstance(image_or_path, Image.Image):
        raw_image = image_or_path.convert("RGB")
    else:
        raise ValueError("image_or_path must be a file path or PIL Image.")

    resized_image = raw_image.resize((224, 224))
    input_tensor = EVAL_TRANSFORM(raw_image).unsqueeze(0).to(device)

    model.eval()
    model.zero_grad()

    if model_name == "densenet121":
        feats = model.features(input_tensor)
        feats.retain_grad()
        out = F.relu(feats)
        out = F.adaptive_avg_pool2d(out, (1, 1)).flatten(1)
        logits = model.classifier(out)

        probs = torch.softmax(logits, dim=1).squeeze().detach().cpu().numpy()
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        logits[0, target_class].backward()
        weights = feats.grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * F.relu(feats).detach()).sum(dim=1).squeeze()
        cam = torch.clamp(cam, min=0).cpu().numpy()

    else:
        if model_name == "custom_cnn":
            target_layer = model.features[-3]
        elif model_name == "mobilenetv3":
            target_layer = model.features[-1][0]
        elif model_name == "efficientnetv2_s":
            target_layer = model.features[-1][0]
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        activations = None
        gradients = None

        def forward_hook(module, inp, out):
            nonlocal activations
            activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            nonlocal gradients
            gradients = grad_out[0].detach()

        h1 = target_layer.register_forward_hook(forward_hook)
        h2 = target_layer.register_full_backward_hook(backward_hook)

        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).squeeze().detach().cpu().numpy()

        if target_class is None:
            target_class = int(output.argmax(dim=1).item())

        output[0, target_class].backward()
        h1.remove()
        h2.remove()

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1).squeeze()
        cam = torch.clamp(cam, min=0).cpu().numpy()

    max_val = np.max(cam)
    if max_val > 0:
        cam = cam / max_val
    else:
        cam = np.zeros_like(cam)

    cam_pil = Image.fromarray(np.uint8(255 * cam)).resize((224, 224), Image.Resampling.BILINEAR)
    cam_resized = np.array(cam_pil) / 255.0
    heatmap = plt.cm.jet(cam_resized)[:, :, :3]
    orig_np = np.array(resized_image) / 255.0
    overlay = np.clip(0.55 * orig_np + 0.45 * heatmap, 0.0, 1.0)

    return cam_resized, target_class, probs, overlay, resized_image


def generate_all_gradcam(models_dict, image_or_path, target_class=None, device=None):
    """Run inference and Grad-CAM generation across all loaded models in one pass."""
    results = {}
    for model_name, (model, classes) in models_dict.items():
        cam, pred_idx, probs, overlay, resized_img = generate_gradcam(
            model=model,
            model_name=model_name,
            image_or_path=image_or_path,
            target_class=target_class,
            device=device,
        )
        pred_label = classes[pred_idx]
        confidence = float(probs[pred_idx])

        top3_indices = np.argsort(probs)[::-1][:3]
        top3 = [(classes[idx], float(probs[idx])) for idx in top3_indices]

        results[model_name] = {
            "predicted_idx": pred_idx,
            "predicted_class": pred_label,
            "confidence": confidence,
            "probabilities": probs,
            "top3": top3,
            "cam": cam,
            "overlay": overlay,
            "classes": classes,
        }
    return results


# ==============================================================================
# VISUALIZATION & FIGURE BUILDERS
# ==============================================================================

def plot_single_gradcam(overlay, raw_image, cam, pred_name, conf, true_name=None, out_path=None):
    """Saves a 3-panel figure: Original, Heatmap, and Overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(raw_image)
    axes[0].set_title(f"Original\n({true_name})" if true_name else "Original Image", fontsize=11)
    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Activation Heatmap", fontsize=11)
    axes[2].imshow(overlay)
    axes[2].set_title(f"Grad-CAM: {pred_name}\n({conf * 100:.1f}%)", fontsize=11)

    for ax in axes:
        ax.axis("off")
    fig.tight_layout()

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path
    return fig


def plot_all_models_comparison(results_dict, raw_image, out_path=None):
    """Saves a 4-column side-by-side Grad-CAM comparison across all 4 models."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    model_order = ["custom_cnn", "mobilenetv3", "densenet121", "efficientnetv2_s"]

    for idx, model_name in enumerate(model_order):
        if model_name in results_dict:
            res = results_dict[model_name]
            axes[idx].imshow(res["overlay"])
            axes[idx].set_title(
                f"{MODEL_DISPLAY_NAMES[model_name]}\n"
                f"Pred: {res['predicted_class']}\n"
                f"Conf: {res['confidence'] * 100:.1f}%",
                fontsize=10,
                fontweight="bold" if model_name == "efficientnetv2_s" else "normal"
            )
        axes[idx].axis("off")

    fig.tight_layout()
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path
    return fig


# ==============================================================================
# CLI HANDLER
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Centralized Eye Disease Grad-CAM Engine.")
    parser.add_argument("--model", type=str, default="efficientnetv2_s",
                        choices=["custom_cnn", "mobilenetv3", "densenet121", "efficientnetv2_s", "all"],
                        help="Model to visualize or 'all' for comparative overlay.")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to an ocular image.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Optional override path to model weights.")
    parser.add_argument("--target-class", type=int, default=None,
                        help="Optional integer target class index to explain.")
    parser.add_argument("--out", type=str, default=None,
                        help="Output path to save the generated figure.")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run batch Grad-CAM across all 12 classes in test dataset.")
    parser.add_argument("--device", type=str, default=None,
                        help="Compute device ('cuda' or 'cpu').")

    args = parser.parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print("  Unified Ocular Grad-CAM Engine")
    print(f"  Target: {args.model.upper()} | Device: {device}")
    print("=" * 65)

    if args.benchmark:
        if args.model == "all":
            selected_models = list(DEFAULT_CHECKPOINTS.keys())
        else:
            selected_models = [args.model]

        for m_name in selected_models:
            print(f"\nRunning benchmark for: {m_name}")
            model, classes = load_model(m_name, args.checkpoint, device)
            out_dir = GRADCAM_DIR / m_name
            out_dir.mkdir(parents=True, exist_ok=True)

            for class_name in classes:
                class_folder = TEST_DATASET_DIR / class_name
                if class_folder.is_dir():
                    samples = sorted(class_folder.glob("*.jpg"))
                    if samples:
                        sample_img = samples[0]
                        cam, pred_idx, probs, overlay, raw_img = generate_gradcam(
                            model, m_name, sample_img, device=device
                        )
                        save_file = out_dir / f"gradcam_{m_name}_{class_name.replace(' ', '_')}.png"
                        plot_single_gradcam(
                            overlay=overlay,
                            raw_image=raw_img,
                            cam=cam,
                            pred_name=classes[pred_idx],
                            conf=probs[pred_idx],
                            true_name=class_name,
                            out_path=save_file
                        )
                        print(f" - [{class_name}] -> Pred: {classes[pred_idx]} ({probs[pred_idx]*100:.1f}%) -> {save_file.name}")
        print("\nBenchmark completed successfully!")
        return

    if not args.image:
        default_test_sample = TEST_DATASET_DIR / "Cataract" / "Cataract_00000.jpg"
        if default_test_sample.exists():
            args.image = str(default_test_sample)
            print(f"No image provided. Using sample: {args.image}")
        else:
            parser.error("Please provide an image with --image <path> or use --benchmark.")

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if args.model == "all":
        models_dict = {}
        for m_name in DEFAULT_CHECKPOINTS.keys():
            m_obj, classes = load_model(m_name, device=device)
            models_dict[m_name] = (m_obj, classes)

        results = generate_all_gradcam(models_dict, image_path, target_class=args.target_class, device=device)
        raw_img = Image.open(image_path).convert("RGB")
        out_file = args.out or (GRADCAM_DIR / f"all_models_comparison_{image_path.stem}.png")
        saved_path = plot_all_models_comparison(results, raw_img, out_path=out_file)
        print(f"\n4-Model Grad-CAM comparison saved to: {saved_path}")

    else:
        model, classes = load_model(args.model, args.checkpoint, device)
        cam, pred_idx, probs, overlay, raw_img = generate_gradcam(
            model=model,
            model_name=args.model,
            image_or_path=image_path,
            target_class=args.target_class,
            device=device,
        )
        pred_label = classes[pred_idx]
        confidence = float(probs[pred_idx])

        print(f"\nPrediction Result for {MODEL_DISPLAY_NAMES[args.model]}:")
        print(f" - Predicted Class: {pred_label}")
        print(f" - Confidence     : {confidence * 100:.2f}%\n")
        print("Top 3 Differential Diagnoses:")
        top3_indices = np.argsort(probs)[::-1][:3]
        for rank, idx in enumerate(top3_indices, 1):
            print(f"  {rank}. {classes[idx]:25s}: {probs[idx]*100:.2f}%")

        out_file = args.out or (GRADCAM_DIR / f"gradcam_{args.model}_{image_path.stem}.png")
        saved_path = plot_single_gradcam(
            overlay=overlay,
            raw_image=raw_img,
            cam=cam,
            pred_name=pred_label,
            conf=confidence,
            out_path=out_file,
        )
        print(f"\nGrad-CAM figure saved to: {saved_path}")


if __name__ == "__main__":
    main()
