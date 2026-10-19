"""Leaf-disease image classifier: a small pretrained network, fine-tuned on tomato and potato leaves.

Transfer learning in one sentence: start from MobileNetV3-small (already good at seeing edges,
textures, shapes from millions of ImageNet photos) and teach only the last layers, then the rest
gently, what our 10 leaf conditions look like.

The classifier also says "I'm not sure": if the highest probability is below `tau`, predict()
returns (None, prob) and the agent asks the farmer for a better photo.
"""
import io
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet statistics the base network expects

EVAL_TF = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])
TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),  # field photos vary in light
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


class LeafClassifier:
    def __init__(self, model: nn.Module, classes: list[str], tau: float = 0.0):
        self.model, self.classes, self.tau = model.eval(), classes, tau

    @classmethod
    def load(cls, path: str | Path) -> "LeafClassifier":
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        model = build_model(len(ckpt["classes"]), pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
        return cls(model, ckpt["classes"], ckpt.get("tau", 0.0))

    @torch.no_grad()
    def probabilities(self, image_bytes: bytes) -> torch.Tensor:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        logits = self.model(EVAL_TF(img).unsqueeze(0))
        return torch.softmax(logits, dim=1)[0]

    def predict(self, image_bytes: bytes) -> tuple[str | None, float]:
        probs = self.probabilities(image_bytes)
        p, idx = probs.max(dim=0)
        if float(p) < self.tau:
            return None, float(p)
        return self.classes[int(idx)], float(p)
