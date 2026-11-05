from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch
import torch.nn.functional as F


def binary_dice_loss(
    logits: torch.Tensor, target: torch.Tensor, sample_mask: Optional[torch.Tensor] = None, eps: float = 1e-6
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    target = target.float()
    spatial_dims = tuple(range(2, logits.ndim))
    intersection = (probabilities * target).sum(dim=spatial_dims)
    denominator = probabilities.sum(dim=spatial_dims) + target.sum(dim=spatial_dims)
    # Eq. (4) is evaluated independently for every task/class, then averaged.
    loss = (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean(dim=1)
    if sample_mask is not None:
        weights = sample_mask.float().view(-1)
        return (loss * weights).sum() / weights.sum().clamp_min(1.0)
    return loss.mean()


def segmentation_loss(
    logits: torch.Tensor, target: torch.Tensor, sample_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    per_pixel = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none")
    bce_per_sample = per_pixel.flatten(1).mean(dim=1)
    if sample_mask is not None:
        weights = sample_mask.float().view(-1)
        bce = (bce_per_sample * weights).sum() / weights.sum().clamp_min(1.0)
    else:
        bce = bce_per_sample.mean()
    dice = binary_dice_loss(logits, target, sample_mask=sample_mask)
    return 0.5 * bce + 0.5 * dice


def _phi(feature: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    attention = feature.mean(dim=1, keepdim=True)
    minimum = attention.amin(dim=(-2, -1), keepdim=True)
    maximum = attention.amax(dim=(-2, -1), keepdim=True)
    return (attention - minimum) / (maximum - minimum).clamp_min(eps)


def representation_knowledge_distillation_loss(
    student_features: Sequence[torch.Tensor],
    teacher_features: Sequence[torch.Tensor],
    sample_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Layer-wise L2 loss in Eq. (5), after channel average and min-max scaling."""
    if len(student_features) != len(teacher_features):
        raise ValueError("Teacher and student must expose the same regularized layers")
    losses = []
    for student, teacher in zip(student_features, teacher_features):
        student_map = _phi(student)
        teacher_map = _phi(teacher.detach())
        if teacher_map.shape[-2:] != student_map.shape[-2:]:
            teacher_map = F.interpolate(teacher_map, student_map.shape[-2:], mode="bilinear", align_corners=False)
        per_sample = (student_map - teacher_map).square().flatten(1).sum(dim=1)
        if sample_mask is not None:
            weights = sample_mask.float().view(-1)
            losses.append((per_sample * weights).sum() / weights.sum().clamp_min(1.0))
        else:
            losses.append(per_sample.mean())
    return torch.stack(losses).sum()
