"""
densenet_gradcam.py
===================
Standalone Grad-CAM utility for the DenseNet121 eye-disease model.
Does NOT import anything from Dev_train_densenet121.py.

Usage
-----
# One image, infer class automatically:
python densenet_gradcam.py path/to/image.jpg

# Force a specific class index:
python densenet_gradcam.py path/to/image.jpg --class_idx 4

# Custom checkpoint:
python densenet_gradcam.py path/to/image.jpg --checkpoint path/to/model.pt

# Save the figure to a specific path:
python densenet_gradcam.py path/to/image.jpg --out my_gradcam.png
"""

import argparse
import os
from pathlib import Path
import tempfile

os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "mpl_cache")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

PROJECT_ROOT    = Path(__file__).resolve().parent
DEFAULT_CKPT    = PROJECT_ROOT / "models" / "densenet121_best.pt"

# ---------------------------------------------------------------------------
# PREPROCESSING  (identical to the eval transform in Dev_train_densenet121.py)
# ---------------------------------------------------------------------------

_EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225],
    ),
])

# ---------------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------------

def load_densenet121(checkpoint_path=None, device=None):
    """Load the trained DenseNet121 from a checkpoint.

    Parameters
    ----------
    checkpoint_path : str or Path, optional
        Path to the .pt checkpoint file.  Defaults to
        ``models/densenet121_best.pt`` relative to this script.
    device : torch.device or str, optional
        Target device.  Defaults to CUDA if available, else CPU.

    Returns
    -------
    model : torch.nn.Module
        DenseNet121 in eval mode on the requested device.
    class_names : list[str]
        Ordered list of class names stored in the checkpoint.
    """
    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CKPT
    checkpoint_path = Path(checkpoint_path)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Load checkpoint first so we know num_classes before building the model.
    ckpt = torch.load(checkpoint_path, map_location=device)
    class_names = ckpt["classes"]
    num_classes  = len(class_names)

    # Build architecture with weights=None (we load our own weights below).
    model = models.densenet121(weights=None)

    # Replace head to match the trained architecture:
    #   Sequential( Dropout(0.2), Linear(1024, num_classes) )
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(1024, num_classes),
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path.name}  "
          f"(epoch {ckpt.get('epoch', '?')}, val F1 {ckpt.get('val_f1', '?'):.4f})")
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Device : {device}\n")

    return model, class_names

# ---------------------------------------------------------------------------
# GRAD-CAM  (hook-free — avoids DenseNet's in-place ReLU after norm5)
# ---------------------------------------------------------------------------

def generate_densenet_gradcam(model, image, target_class=None):
    """Compute a Grad-CAM heatmap for DenseNet121 without hooks.

    Why hook-free?
    torchvision's DenseNet applies an in-place ReLU immediately after norm5.
    A forward hook would see the tensor *after* that in-place op, corrupting
    the activations; a backward hook fails because in-place ops break the
    autograd graph.  Instead we re-run only ``model.features()`` with grad
    enabled, retain_grad() the raw feature map, then reconstruct the rest of
    the forward pass with a non-in-place F.relu so backward() works cleanly.

    Parameters
    ----------
    model : torch.nn.Module
        A DenseNet121 loaded with ``load_densenet121()``.
    image : str, Path, or PIL.Image.Image
        Input image.  A file path is opened and converted to RGB automatically.
    target_class : int, optional
        Class index to explain.  If None, the predicted (argmax) class is used.

    Returns
    -------
    cam : np.ndarray, shape (H, W), dtype float32, values in [0, 1]
        Grad-CAM activation map (7×7 for a 224×224 input).
    target_class : int
        The class index that was explained.
    """
    # --- determine the device the model lives on ---
    device = next(model.parameters()).device

    # --- load / preprocess image ---
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")

    x = _EVAL_TRANSFORM(image).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    model.eval()
    model.zero_grad()

    # --- forward through feature extractor only ---
    # Need gradients on the raw feature map *before* the in-place ReLU.
    feats = model.features(x)   # (1, 1024, 7, 7) for 224×224 input
    feats.retain_grad()         # keep grad on this non-leaf tensor

    # --- reconstruct the rest of the forward pass with no in-place ops ---
    out    = F.relu(feats)                       # non-in-place relu
    out    = F.adaptive_avg_pool2d(out, (1, 1))  # global average pool → (1, 1024, 1, 1)
    out    = out.flatten(1)                      # (1, 1024)
    logits = model.classifier(out)               # (1, num_classes)

    # --- pick target class ---
    if target_class is None:
        target_class = torch.argmax(logits, dim=1).item()

    # --- backward ---
    logits[0, target_class].backward()

    # --- build CAM ---
    # Channel importance = global mean of gradients over spatial dims.
    weights = feats.grad.mean(dim=(2, 3), keepdim=True)   # (1, 1024, 1, 1)
    # Weighted sum of ReLU-activated feature maps, collapse channels → (H, W).
    cam = (weights * F.relu(feats).detach()).sum(dim=1).squeeze()
    cam = torch.clamp(cam, min=0).cpu().numpy().astype(np.float32)

    # Normalise to [0, 1]
    if cam.max() > 0:
        cam = cam / cam.max()

    return cam, target_class

# ---------------------------------------------------------------------------
# OVERLAY & FIGURE HELPERS
# ---------------------------------------------------------------------------

def make_overlay(image, cam):
    """Blend the original image with a jet-coloured Grad-CAM heatmap.

    Parameters
    ----------
    image : PIL.Image.Image
        Original image (any size; will be resized to 224×224).
    cam : np.ndarray
        Grad-CAM array, values in [0, 1].

    Returns
    -------
    overlay : np.ndarray, shape (224, 224, 3), dtype uint8
        50% original + 50% jet heatmap.
    """
    img_224  = image.convert("RGB").resize((224, 224))
    cam_pil  = Image.fromarray(np.uint8(255 * cam)).resize(
                    (224, 224), Image.Resampling.BILINEAR)
    heatmap  = plt.cm.jet(np.array(cam_pil) / 255.0)[:, :, :3]  # (224,224,3)

    blend   = 0.5 * (np.array(img_224) / 255.0) + 0.5 * heatmap
    blend   = np.clip(blend, 0, 1)
    overlay = (blend * 255).astype(np.uint8)
    return overlay


def save_gradcam_figure(image, cam, pred_name, save_path, true_name=None):
    """Save a 3-panel Grad-CAM figure (original | heatmap | overlay).

    Matches the visual style of the figures already in
    ``reports/gradcam_samples/``.

    Parameters
    ----------
    image : str, Path, or PIL.Image.Image
        Original image.
    cam : np.ndarray
        Grad-CAM array from ``generate_densenet_gradcam()``.
    pred_name : str
        Predicted class label shown in the overlay panel title.
    save_path : str or Path
        Output file path (PNG recommended).
    true_name : str, optional
        Ground-truth label shown in the original-image panel title.
        If None, the title reads ``"Original"``.
    """
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")

    orig_title = f"Original: {true_name}" if true_name else "Original"
    overlay    = make_overlay(image, cam)

    cam_pil    = Image.fromarray(np.uint8(255 * cam)).resize(
                     (224, 224), Image.Resampling.BILINEAR)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image.resize((224, 224)))
    axes[0].set_title(orig_title, fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(np.array(cam_pil), cmap="jet")
    axes[1].set_title("Activation Heatmap", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Grad-CAM: Pred={pred_name}", fontsize=10)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")

# ---------------------------------------------------------------------------
# COMMAND-LINE ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a DenseNet121 Grad-CAM heatmap for a single image."
    )
    parser.add_argument(
        "image",
        help="Path to the input image (JPG / PNG).",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CKPT),
        help=f"Path to the .pt checkpoint (default: {DEFAULT_CKPT}).",
    )
    parser.add_argument(
        "--class_idx",
        type=int,
        default=None,
        help="Class index to explain (default: argmax / predicted class).",
    )
    parser.add_argument(
        "--true_name",
        default=None,
        help="Ground-truth class label to show in the figure title (optional).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG path.  Defaults to <image_stem>_gradcam.png "
             "saved next to the input image.",
    )
    args = parser.parse_args()

    # --- load model ---
    model, class_names = load_densenet121(
        checkpoint_path=args.checkpoint,
    )

    # --- run Grad-CAM ---
    image_path = Path(args.image)
    pil_image  = Image.open(image_path).convert("RGB")

    cam, pred_idx = generate_densenet_gradcam(
        model, pil_image, target_class=args.class_idx
    )
    pred_name = class_names[pred_idx]
    print(f"Predicted class : [{pred_idx}] {pred_name}")

    # --- determine output path ---
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = image_path.parent / f"{image_path.stem}_gradcam.png"

    # --- save figure ---
    save_gradcam_figure(
        pil_image, cam, pred_name, out_path, true_name=args.true_name
    )


if __name__ == "__main__":
    main()
