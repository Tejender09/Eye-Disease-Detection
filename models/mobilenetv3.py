
import torch
import torch.nn as nn
from torchvision.models import (
    mobilenet_v3_large,
    MobileNet_V3_Large_Weights
)


def build_mobilenetv3(num_classes=5, pretrained=True):

    if pretrained:
        weights = MobileNet_V3_Large_Weights.DEFAULT
    else:
        weights = None

    model = mobilenet_v3_large(weights=weights)

    # Replace the final classification layer
    input_features = model.classifier[3].in_features

    model.classifier[3] = nn.Linear(
    input_features,
    num_classes
)
    return model


if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = build_mobilenetv3(num_classes=5)
    model = model.to(device)

    dummy_input = torch.randn(
        2, 3, 224, 224
    ).to(device)

    output = model(dummy_input)

    print("Model: MobileNetV3-Large")
    print("Device:", device)
    print("Output shape:", output.shape)
    print("Number of classes:", output.shape[1])

