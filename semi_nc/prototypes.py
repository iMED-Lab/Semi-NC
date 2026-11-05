from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Union

import torch
import torch.nn.functional as F

from .sgc import masked_average_pool, superpixel_guided_clustering


class PrototypePool:
    """Per-class dynamic pools with Algorithm 1 greedy cosine merging."""

    def __init__(self, num_categories: int, capacity: int = 256) -> None:
        self.num_categories = int(num_categories)
        self.capacity = int(capacity)
        self._pools: Dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return sum(pool.shape[0] for pool in self._pools.values())

    def get(self, category: int) -> Optional[torch.Tensor]:
        return self._pools.get(int(category))

    def update(self, category: int, prototypes: torch.Tensor) -> None:
        if prototypes.numel() == 0:
            return
        category = int(category)
        incoming = prototypes.detach()
        current = self._pools.get(category)
        pool = incoming if current is None else torch.cat((current.to(incoming.device), incoming), dim=0)
        self._pools[category] = self._merge_to_capacity(pool)

    def _merge_to_capacity(self, pool: torch.Tensor) -> torch.Tensor:
        while pool.shape[0] > self.capacity:
            normalized = F.normalize(pool, dim=1)
            similarity = normalized @ normalized.transpose(0, 1)
            similarity.fill_diagonal_(-float("inf"))
            flat_index = similarity.argmax()
            i = int(flat_index // pool.shape[0])
            j = int(flat_index % pool.shape[0])
            merged = (pool[i] + pool[j]) / 2.0
            keep = torch.ones(pool.shape[0], dtype=torch.bool, device=pool.device)
            keep[i] = False
            keep[j] = False
            pool = torch.cat((pool[keep], merged.unsqueeze(0)), dim=0)
        return pool

    def state_dict(self) -> Dict[str, object]:
        return {
            "capacity": self.capacity,
            "num_categories": self.num_categories,
            "pools": {str(key): value.detach().cpu() for key, value in self._pools.items()},
        }

    def load_state_dict(self, state: Mapping[str, object], device: Union[torch.device, str] = "cpu") -> None:
        self.capacity = int(state["capacity"])
        self.num_categories = int(state["num_categories"])
        pools = state.get("pools", {})
        self._pools = {int(key): value.to(device) for key, value in pools.items()}  # type: ignore[union-attr]


def categorical_probabilities(foreground_probabilities: torch.Tensor) -> torch.Tensor:
    """Append an explicit background score to independent foreground sigmoids."""
    background = torch.prod(1.0 - foreground_probabilities, dim=1, keepdim=True)
    return torch.cat((background, foreground_probabilities), dim=1)


def update_pools_from_batch(
    pool: PrototypePool,
    embeddings: torch.Tensor,
    masks: torch.Tensor,
    confidence: Optional[torch.Tensor],
    gamma: float,
    labeled: bool,
    num_prototypes: int,
    iterations: int,
    spatial_weight_denominator: float,
    apply_update: bool = True,
) -> Dict[int, torch.Tensor]:
    """Sec. 2.3.1 construction for foreground and background categories."""
    target_size = embeddings.shape[-2:]
    source_masks = masks > 0.5
    source_confidence = confidence
    masks = F.interpolate(source_masks.float(), size=target_size, mode="nearest")
    b, num_foreground, _, _ = masks.shape
    candidates_by_category: Dict[int, List[torch.Tensor]] = {
        category: [] for category in range(num_foreground + 1)
    }
    for batch_index in range(b):
        union = masks[batch_index].amax(dim=0) > 0.5
        background = ~union
        if background.any():
            candidates_by_category[0].append(
                masked_average_pool(embeddings[batch_index], background).unsqueeze(0)
            )
        for foreground_index in range(num_foreground):
            class_id = foreground_index + 1
            class_mask = masks[batch_index, foreground_index] > 0.5
            if labeled:
                high_mask = class_mask
                low_mask = torch.zeros_like(class_mask)
            else:
                if confidence is None:
                    raise ValueError("Unlabeled pool update requires confidence maps")
                source_class = source_masks[batch_index, foreground_index]
                source_class_confidence = source_confidence[batch_index, foreground_index]
                high_mask = F.interpolate(
                    (source_class & (source_class_confidence > gamma))[None, None].float(),
                    size=target_size,
                    mode="nearest",
                )[0, 0].bool()
                low_mask = F.interpolate(
                    (source_class & (source_class_confidence <= gamma))[None, None].float(),
                    size=target_size,
                    mode="nearest",
                )[0, 0].bool()
            if high_mask.any():
                prototypes = superpixel_guided_clustering(
                    embeddings[batch_index],
                    high_mask,
                    num_prototypes=num_prototypes,
                    iterations=iterations,
                    spatial_weight_denominator=spatial_weight_denominator,
                )
                candidates_by_category[class_id].append(prototypes)
            if low_mask.any():
                candidates_by_category[class_id].append(
                    masked_average_pool(embeddings[batch_index], low_mask).unsqueeze(0),
                )
    collected = {
        category: torch.cat(candidates, dim=0)
        for category, candidates in candidates_by_category.items()
        if candidates
    }
    if apply_update:
        for category, prototypes in collected.items():
            pool.update(category, prototypes)
    return collected


def _sample_rows(rows: torch.Tensor, count: int) -> torch.Tensor:
    if rows.shape[0] >= count:
        indices = torch.randperm(rows.shape[0], device=rows.device)[:count]
    else:
        indices = torch.randint(rows.shape[0], (count,), device=rows.device)
    return rows[indices]


def pixel_prototype_contrastive_terms(
    embeddings: torch.Tensor,
    category_probabilities: torch.Tensor,
    labeled_masks: Optional[torch.Tensor],
    pseudo_masks: Optional[torch.Tensor],
    pseudo_confidence: Optional[torch.Tensor],
    pool: PrototypePool,
    query_count: int = 64,
    negative_count: int = 128,
    delta: float = 0.6,
    rank_low: float = 3.0,
    rank_high: float = 9.0,
    temperature: float = 0.1,
) -> List[torch.Tensor]:
    """Return the per-query Eq. (6) terms, with Q queries per class/batch."""
    b, channels, h, w = embeddings.shape
    probabilities = F.interpolate(
        category_probabilities, size=(h, w), mode="bilinear", align_corners=False
    )
    num_categories = probabilities.shape[1]
    if num_categories < 3:
        raise ValueError(
            "Rank-percentile negative sampling requires background plus at least two foreground classes"
        )
    losses: List[torch.Tensor] = []
    candidates_by_category: List[List[torch.Tensor]] = [
        [] for _ in range(num_categories)
    ]
    probabilities_by_category: List[List[torch.Tensor]] = [
        [] for _ in range(num_categories)
    ]
    labeled_count = 0 if labeled_masks is None else labeled_masks.shape[0]
    for batch_index in range(b):
        if batch_index < labeled_count:
            foreground = F.interpolate(
                labeled_masks[batch_index : batch_index + 1].float(), (h, w), mode="nearest"
            )[0] > 0.5
            category_masks = [~foreground.amax(dim=0)] + [mask for mask in foreground]
        else:
            if pseudo_masks is None or pseudo_confidence is None:
                raise ValueError("Unlabeled queries require pseudo masks and confidences")
            pseudo_index = batch_index - labeled_count
            source_foreground = pseudo_masks[pseudo_index] > 0.5
            source_confidence = pseudo_confidence[pseudo_index]
            source_background = ~source_foreground.amax(dim=0)
            source_background_confidence = category_probabilities[batch_index, 0]
            foreground = F.interpolate(
                source_foreground[None].float(), (h, w), mode="nearest"
            )[0] > 0.5
            valid_background = F.interpolate(
                (source_background & (source_background_confidence > delta))[None, None].float(),
                (h, w),
                mode="nearest",
            )[0, 0].bool()
            category_masks = [valid_background] + [
                F.interpolate(
                    (source_foreground[index] & (source_confidence[index] > delta))[None, None].float(),
                    (h, w),
                    mode="nearest",
                )[0, 0].bool()
                for index in range(foreground.shape[0])
            ]
        flat_features = embeddings[batch_index].permute(1, 2, 0)
        flat_probabilities = probabilities[batch_index].permute(1, 2, 0)
        for category, category_mask in enumerate(category_masks):
            if category_mask.any():
                candidates_by_category[category].append(flat_features[category_mask])
                probabilities_by_category[category].append(flat_probabilities[category_mask])

    for category in range(num_categories):
        positive_pool = pool.get(category)
        if positive_pool is None or not candidates_by_category[category]:
            continue
        candidates = torch.cat(candidates_by_category[category], dim=0)
        probability_candidates = torch.cat(probabilities_by_category[category], dim=0)
        if candidates.shape[0] >= query_count:
            query_indices = torch.randperm(candidates.shape[0], device=candidates.device)[:query_count]
        else:
            query_indices = torch.randint(candidates.shape[0], (query_count,), device=candidates.device)
        queries = candidates[query_indices]
        query_probabilities = probability_candidates[query_indices]
        positive_pool = positive_pool.to(queries.device)
        positive_similarity = F.normalize(queries, dim=1) @ F.normalize(positive_pool, dim=1).T
        positives = positive_pool[positive_similarity.argmax(dim=1)]

        order = query_probabilities.argsort(dim=1, descending=True)
        ranks = torch.empty_like(order, dtype=queries.dtype)
        rank_values = torch.linspace(0.0, 10.0, num_categories, device=queries.device)
        ranks.scatter_(1, order, rank_values.expand_as(order))
        for q_index, query in enumerate(queries):
            eligible_classes = [
                class_id
                for class_id in range(num_categories)
                if class_id != category
                and rank_low <= float(ranks[q_index, class_id]) <= rank_high
                and pool.get(class_id) is not None
            ]
            if not eligible_classes:
                continue
            negative_rows = torch.cat(
                [pool.get(class_id).to(query.device) for class_id in eligible_classes], dim=0  # type: ignore[union-attr]
            )
            negatives = _sample_rows(negative_rows, negative_count)
            positive_logit = F.cosine_similarity(query.unsqueeze(0), positives[q_index].unsqueeze(0))
            negative_logits = F.cosine_similarity(query.unsqueeze(0), negatives, dim=1)
            logits = torch.cat((positive_logit, negative_logits), dim=0) / temperature
            losses.append(-F.log_softmax(logits, dim=0)[0])
    return losses


def pixel_prototype_contrastive_loss(
    embeddings: torch.Tensor,
    category_probabilities: torch.Tensor,
    labeled_masks: Optional[torch.Tensor],
    pseudo_masks: Optional[torch.Tensor],
    pseudo_confidence: Optional[torch.Tensor],
    pool: PrototypePool,
    query_count: int = 64,
    negative_count: int = 128,
    delta: float = 0.6,
    rank_low: float = 3.0,
    rank_high: float = 9.0,
    temperature: float = 0.1,
) -> torch.Tensor:
    losses = pixel_prototype_contrastive_terms(
        embeddings, category_probabilities, labeled_masks, pseudo_masks,
        pseudo_confidence, pool, query_count, negative_count, delta,
        rank_low, rank_high, temperature,
    )
    if not losses:
        return embeddings.sum() * 0.0
    return torch.stack(losses).mean()
