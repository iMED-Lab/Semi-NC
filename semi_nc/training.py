from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional

import torch
from torch import nn

from .config import dynamic_gamma
from .fia import MaskCorruptor, foreground_aware_amplification
from .losses import representation_knowledge_distillation_loss, segmentation_loss
from .prototypes import (
    PrototypePool,
    categorical_probabilities,
    pixel_prototype_contrastive_terms,
    update_pools_from_batch,
)


def probability_confidence(probabilities: torch.Tensor) -> torch.Tensor:
    """Per-class Bernoulli confidence maps."""
    return torch.maximum(probabilities, 1.0 - probabilities)


def sample_confidence(probabilities: torch.Tensor) -> torch.Tensor:
    """A-006: mean maximum category confidence over predicted foreground."""
    foreground_masks = probabilities >= 0.5
    foreground_confidence = probabilities.masked_fill(~foreground_masks, 0.0).amax(dim=1)
    background_confidence = categorical_probabilities(probabilities)[:, 0]
    values = []
    for index in range(probabilities.shape[0]):
        foreground = foreground_masks[index].any(dim=0)
        region = foreground_confidence[index][foreground]
        values.append(region.mean() if region.numel() else background_confidence[index].mean())
    return torch.stack(values)


def freeze(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


@dataclass
class TrainStepResult:
    loss: torch.Tensor
    segmentation: torch.Tensor
    rkd: torch.Tensor
    pcl: torch.Tensor
    gamma: float
    valid_unlabeled: int


class SemiNCTrainer:
    """Algorithm 2, implemented as one differentiable mini-batch step."""

    def __init__(self, student: nn.Module, teacher: nn.Module, config: Dict[str, Any]) -> None:
        self.student = student
        self.teacher = freeze(teacher)
        self.config = config
        proto_cfg = config["prototype"]
        self.pool = PrototypePool(
            num_categories=int(config["num_classes"]) + 1,
            capacity=int(proto_cfg["pool_capacity"]),
        )
        self.corruptor = MaskCorruptor(config["fia"])

    def step(
        self,
        labeled_images: torch.Tensor,
        labeled_masks: torch.Tensor,
        unlabeled_images: torch.Tensor,
        epoch: int,
    ) -> TrainStepResult:
        training_cfg = self.config["training"]
        proto_cfg = self.config["prototype"]
        gamma = dynamic_gamma(epoch, float(training_cfg["confidence_floor"]))

        student_images = torch.cat((labeled_images, unlabeled_images), dim=0)
        student_out = self.student(student_images)
        labeled_count = labeled_images.shape[0]
        labeled_logits = student_out["logits"][:labeled_count]
        unlabeled_logits = student_out["logits"][labeled_count:]
        unlabeled_probabilities = torch.sigmoid(unlabeled_logits)
        pseudo_masks = (unlabeled_probabilities.detach() >= 0.5).float()
        pseudo_confidence = probability_confidence(unlabeled_probabilities.detach())
        valid_unlabeled = sample_confidence(unlabeled_probabilities.detach()) > gamma

        labeled_seg = segmentation_loss(labeled_logits, labeled_masks)
        unlabeled_seg = segmentation_loss(unlabeled_logits, pseudo_masks, valid_unlabeled)
        seg_loss = labeled_seg + unlabeled_seg

        teacher_masks = torch.cat((labeled_masks, pseudo_masks), dim=0)
        corrupted_masks = self.corruptor(teacher_masks)
        teacher_images, _ = foreground_aware_amplification(
            student_images, corrupted_masks, float(self.config["fia"]["lambda_distance"])
        )
        with torch.no_grad():
            teacher_out = self.teacher(teacher_images)
            teacher_valid_unlabeled = sample_confidence(
                teacher_out["probabilities"][labeled_count:]
            ) > gamma
        valid_rkd = torch.cat(
            (
                torch.ones(labeled_count, device=student_images.device, dtype=torch.bool),
                teacher_valid_unlabeled,
            )
        )
        rkd_loss = representation_knowledge_distillation_loss(
            student_out["regularization_features"],
            teacher_out["regularization_features"],
            sample_mask=valid_rkd,
        )

        pcl_loss = student_images.sum() * 0.0
        if epoch > int(training_cfg["early_epochs"]):
            embeddings = student_out["embedding"]
            update_kwargs = dict(
                num_prototypes=int(proto_cfg["foreground_prototypes"]),
                iterations=int(proto_cfg["sgc_iterations"]),
                spatial_weight_denominator=float(proto_cfg["spatial_weight_denominator"]),
            )
            labeled_candidates = update_pools_from_batch(
                self.pool,
                embeddings[:labeled_count],
                labeled_masks,
                confidence=None,
                gamma=gamma,
                labeled=True,
                apply_update=False,
                **update_kwargs,
            )
            unlabeled_candidates = update_pools_from_batch(
                self.pool,
                embeddings[labeled_count:],
                pseudo_masks,
                confidence=pseudo_confidence,
                gamma=gamma,
                labeled=False,
                apply_update=False,
                **update_kwargs,
            )
            for category in set(labeled_candidates) | set(unlabeled_candidates):
                candidates = []
                if category in labeled_candidates:
                    candidates.append(labeled_candidates[category])
                if category in unlabeled_candidates:
                    candidates.append(unlabeled_candidates[category])
                self.pool.update(category, torch.cat(candidates, dim=0))
            category_probs = categorical_probabilities(student_out["probabilities"])
            pcl_terms = pixel_prototype_contrastive_terms(
                embeddings,
                category_probs,
                labeled_masks=labeled_masks,
                pseudo_masks=pseudo_masks,
                pseudo_confidence=pseudo_confidence,
                pool=self.pool,
                query_count=int(proto_cfg["query_count"]),
                negative_count=int(proto_cfg["negative_key_count"]),
                delta=float(proto_cfg["query_threshold_delta"]),
                rank_low=float(proto_cfg["negative_rank_low"]),
                rank_high=float(proto_cfg["negative_rank_high"]),
                temperature=float(proto_cfg["temperature"]),
            )
            if pcl_terms:
                pcl_loss = torch.stack(pcl_terms).mean()

        total = (
            seg_loss
            + float(training_cfg["alpha_rkd"]) * rkd_loss
            + float(training_cfg["beta_pcl"]) * pcl_loss
        )
        return TrainStepResult(
            loss=total,
            segmentation=seg_loss.detach(),
            rkd=rkd_loss.detach(),
            pcl=pcl_loss.detach(),
            gamma=gamma,
            valid_unlabeled=int(valid_unlabeled.sum()),
        )


def teacher_bootcamp_step(
    teacher: nn.Module,
    images: torch.Tensor,
    masks: torch.Tensor,
    config: Dict[str, Any],
    corruptor: Optional[MaskCorruptor] = None,
) -> torch.Tensor:
    """Sec. 2.2.1: train teacher on FIA images produced from corrupted masks M*."""
    corruptor = corruptor or MaskCorruptor(config["fia"])
    corrupted = corruptor(masks)
    enhanced, _ = foreground_aware_amplification(
        images, corrupted, float(config["fia"]["lambda_distance"])
    )
    output = teacher(enhanced)
    return segmentation_loss(output["logits"], corrupted)
