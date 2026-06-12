"""Learned calibration-frame detector (EfficientNet-B0).

Optional refinement over the FFT rule in detect.py. The FFT rule already
separates calibration grids from tissue on clean frames; this model is trained
on the FFT-bootstrapped labels to catch borderline cases (partial or damaged
grids, unusual tissue textures). Inference runs on cheap 256x256 thumbnails.

Requires torch + torchvision; intended to run on Colab. Import is deferred so
the FFT-only path works without these dependencies installed.
"""

from __future__ import annotations

from pathlib import Path

INPUT_SIZE = 256


class CalibrationDetector:
    def __init__(self, weights_path: str, device: str | None = None):
        import torch
        from torchvision import models

        self._torch = torch
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = models.efficientnet_b0(weights=None)
        self._model.classifier[1] = torch.nn.Linear(
            self._model.classifier[1].in_features, 1
        )
        self._model.load_state_dict(torch.load(weights_path, map_location=self._device))
        self._model.to(self._device).eval()
        self._transform = _build_transform()

    def predict(self, thumbnail_paths: list[str]) -> list[float]:
        from PIL import Image

        batch = self._torch.stack(
            [self._transform(Image.open(path).convert("RGB")) for path in thumbnail_paths]
        ).to(self._device)
        with self._torch.no_grad():
            logits = self._model(batch).squeeze(1)
            probabilities = self._torch.sigmoid(logits)
        return probabilities.cpu().tolist()


def _build_transform():
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def weights_exist(weights_path: str) -> bool:
    return Path(weights_path).is_file()
