from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


class DeformableAttention2D(nn.Module):
    """Multi-head deformable attention implementing Eq. (1).

    Each query location predicts K offsets and K normalized attention weights per
    head. Values are sampled with bilinear interpolation at p_q + delta p_hqk.
    """

    def __init__(
        self,
        channels: int,
        heads: int = 4,
        points: int = 4,
        offset_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by heads")
        self.channels = channels
        self.heads = heads
        self.points = points
        self.head_dim = channels // heads
        self.offset_scale = float(offset_scale)
        self.query = nn.Conv2d(channels, channels, 1, bias=False)
        self.value = nn.Conv2d(channels, channels, 1, bias=False)
        self.offsets = nn.Conv2d(channels, heads * points * 2, 3, padding=1)
        self.weights = nn.Conv2d(channels, heads * points, 3, padding=1)
        self.output = nn.Conv2d(channels, channels, 1, bias=False)
        nn.init.zeros_(self.offsets.weight)
        nn.init.zeros_(self.offsets.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        query = self.query(x)
        value = self.value(x).view(b, self.heads, self.head_dim, h, w)
        offsets = torch.tanh(self.offsets(query)).view(
            b, self.heads, self.points, 2, h, w
        )
        offsets = offsets * self.offset_scale
        weights = self.weights(query).view(b, self.heads, self.points, h, w)
        weights = weights.softmax(dim=2)

        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype),
            torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        base = torch.stack((xx, yy), dim=-1).view(1, 1, 1, h, w, 2)
        grid = base + offsets.permute(0, 1, 2, 4, 5, 3)

        value = value.reshape(b * self.heads, self.head_dim, h, w)
        sampled = []
        for point in range(self.points):
            point_grid = grid[:, :, point].reshape(b * self.heads, h, w, 2)
            point_value = F.grid_sample(
                value,
                point_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            ).view(b, self.heads, self.head_dim, h, w)
            sampled.append(point_value)
        sampled_values = torch.stack(sampled, dim=2)
        attended = (sampled_values * weights.unsqueeze(3)).sum(dim=2)
        attended = attended.reshape(b, c, h, w)
        return self.output(attended)


class MultiScaleDeformableAttention2D(nn.Module):
    """Joint deformable attention over the concatenated multi-scale sequence."""

    def __init__(
        self,
        channels: int,
        levels: int,
        heads: int = 4,
        points: int = 4,
        offset_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by heads")
        self.channels = channels
        self.levels = levels
        self.heads = heads
        self.points = points
        self.head_dim = channels // heads
        self.offset_scale = float(offset_scale)
        self.query = nn.Linear(channels, channels, bias=False)
        self.values = nn.ModuleList(
            [nn.Conv2d(channels, channels, 1, bias=False) for _ in range(levels)]
        )
        self.offsets = nn.Linear(channels, heads * levels * points * 2)
        self.weights = nn.Linear(channels, heads * levels * points)
        self.output = nn.Linear(channels, channels, bias=False)
        nn.init.zeros_(self.offsets.weight)
        nn.init.zeros_(self.offsets.bias)

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != self.levels:
            raise ValueError("TDA received a different number of scales than configured")
        batch = features[0].shape[0]
        tokens = torch.cat(
            [feature.flatten(2).transpose(1, 2) for feature in features], dim=1
        )
        query = self.query(tokens)
        token_count = query.shape[1]
        offsets = torch.tanh(self.offsets(query)).view(
            batch, token_count, self.heads, self.levels, self.points, 2
        )
        offsets = offsets * self.offset_scale
        weights = self.weights(query).view(
            batch, token_count, self.heads, self.levels, self.points
        )
        weights = weights.flatten(-2).softmax(dim=-1).view_as(weights)

        base_coordinates = []
        for feature in features:
            height, width = feature.shape[-2:]
            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, height, device=feature.device, dtype=feature.dtype),
                torch.linspace(-1, 1, width, device=feature.device, dtype=feature.dtype),
                indexing="ij",
            )
            base_coordinates.append(torch.stack((xx, yy), dim=-1).reshape(-1, 2))
        base = torch.cat(base_coordinates, dim=0).view(1, token_count, 1, 2)

        attended = query.new_zeros(batch, self.heads, self.head_dim, token_count)
        for level, (value_projection, feature) in enumerate(zip(self.values, features)):
            height, width = feature.shape[-2:]
            value = value_projection(feature).reshape(
                batch * self.heads, self.head_dim, height, width
            )
            for point in range(self.points):
                grid = base + offsets[:, :, :, level, point, :]
                grid = grid.permute(0, 2, 1, 3).reshape(
                    batch * self.heads, token_count, 1, 2
                )
                sampled = F.grid_sample(
                    value,
                    grid,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                ).view(batch, self.heads, self.head_dim, token_count)
                point_weights = weights[:, :, :, level, point].permute(0, 2, 1)
                attended = attended + sampled * point_weights.unsqueeze(2)
        attended = attended.permute(0, 3, 1, 2).reshape(batch, token_count, self.channels)
        return self.output(attended)


class ExternalBatchAttention(nn.Module):
    """Eq. (2) query-to-mini-batch external attention.

    Keys are pooled from all multi-scale encoder features in the mini-batch.
    Values are the shared learnable memory U_V. Flattening batch and memory-token
    axes makes every sample attend to representations from every other sample.
    """

    def __init__(
        self,
        channels: int,
        heads: int = 4,
        memory_tokens: int = 16,
    ) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by heads")
        self.channels = channels
        self.heads = heads
        self.memory_tokens = memory_tokens
        self.query = nn.Linear(channels, channels, bias=False)
        self.key = nn.Linear(channels, channels, bias=False)
        self.memory_value = nn.Parameter(torch.empty(memory_tokens, channels))
        self.output = nn.Linear(channels, channels, bias=False)
        nn.init.normal_(self.memory_value, std=channels**-0.5)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, token_count, c = x.shape
        q = self.query(x)
        pooled = F.adaptive_avg_pool1d(
            context.transpose(1, 2), self.memory_tokens
        ).transpose(1, 2)
        keys = self.key(pooled)
        keys = keys.reshape(1, b * self.memory_tokens, c).expand(b, -1, -1)
        values = self.memory_value.repeat(b, 1)
        values = values.view(1, b * self.memory_tokens, c).expand(b, -1, -1)

        qh = q.view(b, token_count, self.heads, c // self.heads).transpose(1, 2)
        kh = keys.view(b, -1, self.heads, c // self.heads).transpose(1, 2)
        vh = values.view(b, -1, self.heads, c // self.heads).transpose(1, 2)
        scale = (c // self.heads) ** -0.5
        attention = torch.matmul(qh, kh.transpose(-1, -2)) * scale
        attention = attention.softmax(dim=-1)
        out = torch.matmul(attention, vh).transpose(1, 2).reshape(b, token_count, c)
        return self.output(out)


class DEAM(nn.Module):
    """Fig. 4(b): per-scale tokens, TDA/TEA, Add&Norm, FFN, split."""

    def __init__(
        self,
        input_channels: Sequence[int],
        output_channels: int,
        heads: int,
        points: int,
        offset_scale: float,
        memory_tokens: int,
    ) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, output_channels, 1, bias=False),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(inplace=True),
                )
                for channels in input_channels
            ]
        )
        self.tda = MultiScaleDeformableAttention2D(
            output_channels,
            levels=len(input_channels),
            heads=heads,
            points=points,
            offset_scale=offset_scale,
        )
        self.tea = ExternalBatchAttention(
            output_channels,
            heads=heads,
            memory_tokens=memory_tokens,
        )
        self.attention_fusion = nn.Linear(output_channels * 2, output_channels)
        self.norm_attention = nn.LayerNorm(output_channels)
        self.ffn = nn.Sequential(
            nn.Linear(output_channels, output_channels * 4),
            nn.GELU(),
            nn.Linear(output_channels * 4, output_channels),
        )
        self.norm_ffn = nn.LayerNorm(output_channels)

    @staticmethod
    def _tokens(feature: torch.Tensor) -> torch.Tensor:
        return feature.flatten(2).transpose(1, 2)

    def forward(self, features: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if len(features) != len(self.projections):
            raise ValueError("DEAM received a different number of scales than configured")
        projected = [projection(feature) for feature, projection in zip(features, self.projections)]
        base_tokens = torch.cat([self._tokens(feature) for feature in projected], dim=1)
        deformable_tokens = self.tda(projected)
        external_tokens = self.tea(deformable_tokens, base_tokens)
        fused_attention = self.attention_fusion(
            torch.cat((deformable_tokens, external_tokens), dim=-1)
        )
        tokens = self.norm_attention(base_tokens + fused_attention)
        tokens = self.norm_ffn(tokens + self.ffn(tokens))

        outputs = []
        start = 0
        for feature in projected:
            height, width = feature.shape[-2:]
            length = height * width
            scale_tokens = tokens[:, start : start + length]
            outputs.append(scale_tokens.transpose(1, 2).reshape(feature.shape))
            start += length
        return outputs
