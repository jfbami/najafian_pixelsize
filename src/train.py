"""Fine-tune EfficientNet-B0 on FFT-bootstrapped calibration labels.

Reads labels.csv (from bootstrap.py), trains a binary classifier on grid
thumbnails, and saves weights. Designed for a Colab GPU session. The labels are
generated automatically by the FFT detector, so no manual annotation is needed;
spot-check a sample of labels before training.

Requires torch + torchvision.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
VALIDATION_FRACTION = 0.2
INPUT_SIZE = 256


@dataclass
class LabeledFrame:
    path: str
    label: int


def load_labels(labels_csv: Path) -> list[LabeledFrame]:
    with labels_csv.open(newline="") as handle:
        return [
            LabeledFrame(path=row["path"], label=int(row["is_calibration"]))
            for row in csv.DictReader(handle)
        ]


def train(labels_csv: Path, weights_out: Path) -> dict:
    import torch
    from torch.utils.data import DataLoader, random_split

    frames = load_labels(labels_csv)
    dataset = _GridThumbnailDataset(frames)
    val_size = int(len(dataset) * VALIDATION_FRACTION)
    train_set, val_set = random_split(dataset, [len(dataset) - val_size, val_size])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    for epoch in range(EPOCHS):
        _train_epoch(model, train_loader, optimizer, loss_fn, device)
        scheduler.step()

    metrics = _evaluate(model, val_loader, device)
    weights_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_out)
    return metrics


def _build_model(device: str):
    import torch
    from torchvision import models

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)
    return model.to(device)


def _train_epoch(model, loader, optimizer, loss_fn, device) -> None:
    model.train()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device).float()
        optimizer.zero_grad()
        logits = model(images).squeeze(1)
        loss_fn(logits, labels).backward()
        optimizer.step()


def _evaluate(model, loader, device) -> dict:
    import torch

    model.eval()
    true_positive = false_positive = false_negative = 0
    with torch.no_grad():
        for images, labels in loader:
            predictions = (torch.sigmoid(model(images.to(device)).squeeze(1)) > 0.5).int().cpu()
            labels = labels.int()
            true_positive += int(((predictions == 1) & (labels == 1)).sum())
            false_positive += int(((predictions == 1) & (labels == 0)).sum())
            false_negative += int(((predictions == 0) & (labels == 1)).sum())
    precision = true_positive / (true_positive + false_positive + 1e-9)
    recall = true_positive / (true_positive + false_negative + 1e-9)
    return {"precision": precision, "recall": recall}


class _GridThumbnailDataset:
    def __init__(self, frames: list[LabeledFrame]):
        from torchvision import transforms

        self._frames = frames
        self._transform = transforms.Compose(
            [
                transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, index: int):
        from PIL import Image

        frame = self._frames[index]
        image = self._transform(Image.open(frame.path).convert("RGB"))
        return image, frame.label
