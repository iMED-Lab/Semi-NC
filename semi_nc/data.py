from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def build_augmentation(config: Dict[str, Any]) -> A.Compose:
    p = float(config.get("probability", 0.5))
    transforms: List[A.BasicTransform] = []
    flip_candidates: List[A.BasicTransform] = []
    if config.get("horizontal_flip", True):
        flip_candidates.append(A.HorizontalFlip(p=1.0))
    if config.get("vertical_flip", True):
        flip_candidates.append(A.VerticalFlip(p=1.0))
    if flip_candidates:
        transforms.append(A.OneOf(flip_candidates, p=p))
    transforms.append(A.Rotate(limit=int(config.get("rotate_limit", 90)), p=p, border_mode=cv2.BORDER_REFLECT))
    if config.get("gaussian_noise", True):
        transforms.append(A.GaussNoise(p=p))
    if config.get("gaussian_blur", True):
        transforms.append(A.GaussianBlur(p=p))
    if config.get("color_jitter", True):
        transforms.append(A.ColorJitter(p=p))
    if config.get("random_brightness_contrast", True):
        transforms.append(A.RandomBrightnessContrast(p=p))
    return A.Compose(transforms)


class SegmentationManifestDataset(Dataset):
    """CSV dataset: `image,mask_0,...,mask_{C-1}`; unlabeled needs `image` only."""

    def __init__(
        self,
        manifest: Union[str, Path],
        num_classes: int,
        image_size: Optional[Sequence[int]] = None,
        augmentation: Optional[A.Compose] = None,
        unlabeled: bool = False,
    ) -> None:
        self.manifest = Path(manifest)
        self.root = self.manifest.parent
        self.num_classes = int(num_classes)
        self.image_size = tuple(image_size) if image_size else None
        self.augmentation = augmentation
        self.unlabeled = unlabeled
        with self.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise ValueError(f"Empty manifest: {self.manifest}")

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        image_path = self._resolve(row["image"])
        image = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
        masks: List[np.ndarray] = []
        if not self.unlabeled:
            for class_index in range(self.num_classes):
                key = f"mask_{class_index}"
                if key not in row or not row[key]:
                    raise ValueError(f"Missing {key} in row {index} of {self.manifest}")
                mask = np.asarray(Image.open(self._resolve(row[key])).convert("L"), dtype=np.uint8)
                masks.append((mask > 0).astype(np.float32))
        if self.image_size:
            height, width = int(self.image_size[0]), int(self.image_size[1])
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
            masks = [cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST) for mask in masks]
        if self.augmentation is not None:
            transformed = self.augmentation(image=image, masks=masks)
            image, masks = transformed["image"], transformed["masks"]
        sample: Dict[str, Any] = {
            "image": torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0).float(),
            "image_path": str(image_path),
            "patient_id": row.get("patient_id", ""),
        }
        if masks:
            sample["mask"] = torch.from_numpy(np.stack(masks)).float()
        return sample
