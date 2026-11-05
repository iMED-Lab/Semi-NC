from __future__ import annotations

import torch

from third_party.asgnet_sgc import Weighted_GAP as source_weighted_gap
from third_party.asgnet_sgc import place_seed_points_fixed as source_place_seed_points
from third_party.asgnet_sgc import sp_center_iter as source_sp_center_iter


def masked_average_pool(feature: torch.Tensor, mask: torch.Tensor, eps: float = 5e-4) -> torch.Tensor:
    """Directly call the supplied ASGNet Weighted_GAP source."""
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    result = source_weighted_gap(feature.unsqueeze(0), mask.unsqueeze(0).to(feature.dtype))
    return result.squeeze(0).squeeze(-1).squeeze(-1)


def superpixel_guided_clustering(
    feature: torch.Tensor,
    mask: torch.Tensor,
    num_prototypes: int = 3,
    iterations: int = 10,
    spatial_weight_denominator: float = 100.0,
) -> torch.Tensor:
    """Direct, device-agnostic extraction of the supplied ASGNet SGC core.

    This retains ASGNet's coordinate concatenation, exp(-distance) association,
    per-centroid normalization, and weighted centroid update. Source equations:
    `ASGNet-main/model/asgnet.py::sp_center_iter`, corresponding to SGC Eqs. 1-3.
    """
    if feature.ndim != 3 or mask.ndim not in (2, 3):
        raise ValueError("Expected feature [C,H,W] and mask [H,W] or [1,H,W]")
    if float(spatial_weight_denominator) != 100.0:
        raise ValueError("The directly called ASGNet SGC source fixes the spatial denominator at 100")
    binary = mask.squeeze(0) > 0.5
    if binary.sum() < max(2, num_prototypes):
        return masked_average_pool(feature, binary).unsqueeze(0)

    # Directly invoke the MaskSLIC seed placement supplied with ASGNet; only
    # the count is fixed to K=3 as explicitly required by Semi-NC.
    seeds = source_place_seed_points(binary, num_prototypes).to(feature.device)
    if seeds.shape[0] <= 1:
        return masked_average_pool(feature, binary).unsqueeze(0)

    _, h, w = feature.shape
    y_coords = torch.arange(h, device=feature.device, dtype=feature.dtype).view(h, 1).expand(h, w)
    x_coords = torch.arange(w, device=feature.device, dtype=feature.dtype).view(1, w).expand(h, w)
    initial = torch.cat(
        (
            feature[:, seeds[:, 0], seeds[:, 1]],
            seeds.transpose(0, 1).to(feature.dtype),
        ),
        dim=0,
    )
    # Required by the task: invoke the reference SGC source directly.
    centers = source_sp_center_iter(feature, binary.unsqueeze(0).float(), initial, int(iterations))
    return centers.transpose(0, 1).contiguous()
