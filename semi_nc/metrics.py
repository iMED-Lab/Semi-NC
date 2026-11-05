from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree


def dice_score(prediction: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    return float((2 * np.logical_and(prediction, target).sum() + eps) / (prediction.sum() + target.sum() + eps))


def false_discovery_rate(prediction: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    false_positive = np.logical_and(prediction, ~target).sum()
    return float(false_positive / (prediction.sum() + eps))


def _boundary(binary: np.ndarray) -> np.ndarray:
    binary = binary.astype(bool)
    return np.argwhere(np.logical_xor(binary, binary_erosion(binary)))


def modified_hausdorff_distance(prediction: np.ndarray, target: np.ndarray) -> float:
    """Dubuisson-Jain MHD: max of the two directed mean nearest distances."""
    pred_points, target_points = _boundary(prediction), _boundary(target)
    if len(pred_points) == 0 and len(target_points) == 0:
        return 0.0
    if len(pred_points) == 0 or len(target_points) == 0:
        return float(np.hypot(*prediction.shape[-2:]))
    pred_to_target = cKDTree(target_points).query(pred_points, k=1)[0].mean()
    target_to_pred = cKDTree(pred_points).query(target_points, k=1)[0].mean()
    return float(max(pred_to_target, target_to_pred))


def segmentation_metrics(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    return {
        "dice": dice_score(prediction, target),
        "fdr": false_discovery_rate(prediction, target),
        "mhd": modified_hausdorff_distance(prediction, target),
    }
