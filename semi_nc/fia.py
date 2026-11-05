from __future__ import annotations

import random
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


def _as_range(value: Any) -> tuple[float, float]:
    if isinstance(value, (list, tuple)):
        return float(value[0]), float(value[1])
    return float(value), float(value)


class MaskCorruptor:
    """Five paper-defined pseudo-label errors: dilation, forgery,
    truncation, rotation and translation.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.probability = float(config["corruption_probability"])
        self.dilation_radius = _as_range(config["dilation_radius"])
        self.forgery_fraction = _as_range(config["forgery_fraction"])
        self.truncation_fraction = _as_range(config["truncation_fraction"])
        self.rotation_degrees = float(config["rotation_degrees"])
        self.translation_fraction = float(config["translation_fraction"])

    def dilation(self, mask: torch.Tensor) -> torch.Tensor:
        radius = random.randint(int(self.dilation_radius[0]), int(self.dilation_radius[1]))
        return F.max_pool2d(mask, kernel_size=2 * radius + 1, stride=1, padding=radius)

    def forgery(self, mask: torch.Tensor) -> torch.Tensor:
        b, c, h, w = mask.shape
        out = mask.clone()
        low, high = self.forgery_fraction
        fraction = random.uniform(low, high)
        ph = max(1, int(h * fraction**0.5))
        pw = max(1, int(w * fraction**0.5))
        for index in range(b):
            donor = (index + random.randint(1, max(1, b - 1))) % b if b > 1 else index
            sy = random.randint(0, max(0, h - ph))
            sx = random.randint(0, max(0, w - pw))
            dy = random.randint(0, max(0, h - ph))
            dx = random.randint(0, max(0, w - pw))
            out[index, :, dy : dy + ph, dx : dx + pw] = torch.maximum(
                out[index, :, dy : dy + ph, dx : dx + pw],
                mask[donor, :, sy : sy + ph, sx : sx + pw],
            )
        return out

    def truncation(self, mask: torch.Tensor) -> torch.Tensor:
        b, _, h, w = mask.shape
        out = mask.clone()
        low, high = self.truncation_fraction
        for index in range(b):
            fraction = random.uniform(low, high)
            if random.random() < 0.5:
                length = max(1, int(w * fraction))
                if random.random() < 0.5:
                    out[index, :, :, :length] = 0
                else:
                    out[index, :, :, w - length :] = 0
            else:
                length = max(1, int(h * fraction))
                if random.random() < 0.5:
                    out[index, :, :length, :] = 0
                else:
                    out[index, :, h - length :, :] = 0
        return out

    def _affine(self, mask: torch.Tensor, rotate: bool, translate: bool) -> torch.Tensor:
        b = mask.shape[0]
        if rotate:
            angles = torch.empty(b, device=mask.device).uniform_(
                -self.rotation_degrees, self.rotation_degrees
            )
        else:
            angles = torch.zeros(b, device=mask.device)
        radians = angles * torch.pi / 180.0
        if translate:
            tx = torch.empty(b, device=mask.device).uniform_(
                -self.translation_fraction, self.translation_fraction
            ) * 2
            ty = torch.empty(b, device=mask.device).uniform_(
                -self.translation_fraction, self.translation_fraction
            ) * 2
        else:
            tx = torch.zeros(b, device=mask.device)
            ty = torch.zeros(b, device=mask.device)
        theta = torch.zeros(b, 2, 3, device=mask.device, dtype=mask.dtype)
        theta[:, 0, 0] = torch.cos(radians)
        theta[:, 0, 1] = -torch.sin(radians)
        theta[:, 1, 0] = torch.sin(radians)
        theta[:, 1, 1] = torch.cos(radians)
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty
        grid = F.affine_grid(theta, mask.shape, align_corners=False)
        return F.grid_sample(mask, grid, mode="nearest", padding_mode="zeros", align_corners=False)

    def rotation(self, mask: torch.Tensor) -> torch.Tensor:
        return self._affine(mask, rotate=True, translate=False)

    def translation(self, mask: torch.Tensor) -> torch.Tensor:
        return self._affine(mask, rotate=False, translate=True)

    def __call__(self, mask: torch.Tensor) -> torch.Tensor:
        corrupted = mask.float().clone()
        operations = (
            self.dilation,
            self.forgery,
            self.truncation,
            self.rotation,
            self.translation,
        )
        for operation in operations:
            if random.random() < self.probability:
                corrupted = operation(corrupted)
        return (corrupted >= 0.5).to(mask.dtype)


def foreground_aware_amplification(
    image: torch.Tensor, mask: torch.Tensor, lambda_distance: float = 3.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """FIA from Eq. (4): x_e=x/(1+log(1+lambda*|SDF(M)|)).

    Foreground pixels are explicitly reset to weight one as stated in Sec. 2.2.1.
    Multi-class masks are united because the transform amplifies all foreground
    anatomy for the shared encoder.
    """
    if mask.shape[-2:] != image.shape[-2:]:
        mask = F.interpolate(mask.float(), size=image.shape[-2:], mode="nearest")
    foreground = mask.amax(dim=1, keepdim=True) > 0.5
    maps = []
    for binary in foreground.detach().cpu().numpy():
        fg = binary[0].astype(bool)
        if not fg.any():
            maps.append(torch.ones(fg.shape, dtype=torch.float32))
            continue
        outside_distance = distance_transform_edt(~fg)
        weight = 1.0 / (1.0 + np.log1p(float(lambda_distance) * outside_distance))
        weight[fg] = 1.0
        maps.append(torch.from_numpy(weight.astype(np.float32)))
    weight_map = torch.stack(maps, dim=0).unsqueeze(1).to(image.device, image.dtype)
    return image * weight_map, weight_map
