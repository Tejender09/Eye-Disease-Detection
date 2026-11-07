
import torch
import torch.nn as nn

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)


def build_efficientnetb0(num_classes=5, pretrained=True):

    if pretrained:
        weights = EfficientNet_B0_Weights.DEFAULT
    else:
        weights = None

    model = efficientnet_b0(weights=weights)

    input_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        input_features,
        num_classes
    )

    return model


if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = build_efficientnetb0(num_classes=5)
    model = model.to(device)

    dummy_input = torch.randn(
        2, 3, 224, 224
    ).to(device)

    output = model(dummy_input)

    print("Model: EfficientNet-B0")
    print("Device:", device)
    print("Output shape:", output.shape)
    print("Number of classes:", output.shape[1])
