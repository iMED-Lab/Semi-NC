from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .attention import DEAM


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.body(x) + self.skip(x), inplace=True)


class DecoderBlock(nn.Module):
    """The 3x3 decoder convolution shown in Fig. 4."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TaskDecoder(nn.Module):
    def __init__(self, widths: Sequence[int], output_channels: int = 1) -> None:
        super().__init__()
        reversed_widths = list(reversed(widths))
        self.blocks = nn.ModuleList()
        current = reversed_widths[0]
        for skip_width in reversed_widths[1:]:
            self.blocks.append(DecoderBlock(current + skip_width, skip_width))
            current = skip_width
        self.head = nn.Conv2d(current, output_channels, 1)

    def forward(
        self, bottleneck: torch.Tensor, skips: Sequence[torch.Tensor], output_size: Sequence[int]
    ) -> tuple[torch.Tensor, List[torch.Tensor]]:
        x = bottleneck
        intermediates: List[torch.Tensor] = []
        for block, skip in zip(self.blocks, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat((x, skip), dim=1))
            intermediates.append(x)
        logits = self.head(F.interpolate(x, size=output_size, mode="bilinear", align_corners=False))
        return logits, intermediates


class SemiNCNet(nn.Module):
    """Pro-Extract network in Fig. 4 with two task-specific decoders."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        model_cfg = config["model"]
        widths = list(model_cfg["widths"])
        in_channels = int(config.get("image_channels", 1))
        self.num_classes = int(config["num_classes"])
        self.widths = widths

        encoders = []
        previous = in_channels
        for width in widths:
            encoders.append(ResidualBlock(previous, width))
            previous = width
        self.encoders = nn.ModuleList(encoders)
        self.pools = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=2, stride=2) for _ in range(len(widths) - 1)]
        )

        attention_dim = int(model_cfg.get("attention_dim", 128))
        common = dict(
            heads=int(model_cfg["attention_heads"]),
            points=int(model_cfg["deformable_points"]),
            offset_scale=float(model_cfg["deformable_offset_scale"]),
            memory_tokens=int(model_cfg["external_memory_tokens"]),
        )
        # Fig. 4 applies three densely connected DEAM stages to x2:x4, then
        # consolidates those outputs with x5 in the final DEAM.
        self.deams = nn.ModuleList(
            [
                DEAM(widths[1 : level + 2], output_channels=attention_dim, **common)
                for level in range(3)
            ]
        )
        self.final_deam = DEAM(
            [attention_dim, attention_dim, attention_dim, widths[-1]],
            output_channels=attention_dim,
            **common,
        )
        self.channel_adjust = nn.Sequential(
            nn.Conv2d(attention_dim, widths[-1], 1, bias=False),
            nn.BatchNorm2d(widths[-1]),
            nn.ReLU(inplace=True),
        )
        self.skip_adjust = nn.ModuleList(
            [
                nn.Identity(),
                nn.Sequential(
                    nn.Conv2d(attention_dim, widths[2], 1, bias=False),
                    nn.BatchNorm2d(widths[2]),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(attention_dim, widths[3], 1, bias=False),
                    nn.BatchNorm2d(widths[3]),
                    nn.ReLU(inplace=True),
                ),
            ]
        )

        self.decoders = nn.ModuleList([TaskDecoder(widths) for _ in range(self.num_classes)])
        self.embedding = nn.Identity()

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        output_size = x.shape[-2:]
        encoder_features: List[torch.Tensor] = []
        current = x
        for index, encoder in enumerate(self.encoders):
            if index:
                current = self.pools[index - 1](current)
            current = encoder(current)
            encoder_features.append(current)

        deam_features: List[torch.Tensor] = []
        for level, deam in enumerate(self.deams):
            refined_scales = deam(encoder_features[1 : level + 2])
            deam_features.append(refined_scales[-1])
        bottleneck = self.channel_adjust(
            self.final_deam([*deam_features, encoder_features[-1]])[-1]
        )
        # Fig. 4 routes the DEAM-refined x2/x3/x4 features into the decoder.
        # x1 remains the highest-resolution encoder skip.
        refined_skips = [
            adjust(feature) for adjust, feature in zip(self.skip_adjust, deam_features)
        ]
        skips = [encoder_features[0], *refined_skips, bottleneck]

        logits_per_task: List[torch.Tensor] = []
        decoder_features: List[torch.Tensor] = []
        for decoder in self.decoders:
            task_logits, task_features = decoder(bottleneck, skips, output_size)
            logits_per_task.append(task_logits)
            decoder_features.extend(task_features)
        logits = torch.cat(logits_per_task, dim=1)
        embedding = self.embedding(bottleneck)
        return {
            "logits": logits,
            "probabilities": torch.sigmoid(logits),
            "embedding": embedding,
            "encoder_features": encoder_features,
            "attention_features": [*deam_features, bottleneck],
            "decoder_features": decoder_features,
            "regularization_features": [
                *encoder_features,
                *deam_features,
                bottleneck,
                *decoder_features,
            ],
        }
