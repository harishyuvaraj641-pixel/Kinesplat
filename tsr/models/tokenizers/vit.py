"""A self-contained Vision Transformer (ViT-B/16) encoder.

TripoSR was trained against ``transformers`` 4.35, where ``ViTModel`` stored its
parameters as ``encoder.layer.0.attention.attention.query`` and so on.  Later
``transformers`` releases renamed every one of those tensors (``layers.0.
attention.q_proj``), so a 2024 checkpoint can no longer be loaded into the
installed ``ViTModel`` -- ``load_state_dict`` either fails outright or, worse,
silently leaves the image encoder randomly initialised.

This module implements the exact same architecture with the original parameter
names.  Only the published structure of ``facebook/dino-vitb16`` is needed
(12 layers, 768 hidden, 12 heads, 3072 intermediate, 16 px patches, GELU,
layer-norm eps 1e-12); the weights themselves come from TripoSR's checkpoint,
which fine-tuned this encoder.  Keeping it here means the model runs with plain
PyTorch and no version-pinned ``transformers`` install.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ViTOutput:
    last_hidden_state: torch.Tensor
    pooler_output: torch.Tensor | None = None


class ViTPatchEmbeddings(nn.Module):
    """Turn (N, 3, H, W) pixels into (N, H/16 * W/16, 768) patch tokens."""

    def __init__(self, hidden_size: int = 768, patch_size: int = 16) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(
            3, hidden_size, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # (N, C, H, W) -> (N, C, H/P, W/P) -> (N, H/P * W/P, C)
        return self.projection(pixel_values).flatten(2).transpose(1, 2)


class ViTEmbeddings(nn.Module):
    def __init__(self, num_channels: int = 3, image_size: int = 224,
                 patch_size: int = 16, hidden_size: int = 768) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        num_patches = (image_size // patch_size) ** 2
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, num_patches + 1, hidden_size)
        )
        self.patch_embeddings = ViTPatchEmbeddings(hidden_size, patch_size)
        self.dropout = nn.Identity()  # DINO ViT-B never uses dropout

    def interpolate_pos_encoding(self, embeddings: torch.Tensor,
                                 height: int, width: int) -> torch.Tensor:
        """Bilinearly resize the learned 14x14 position grid to a new size.

        TripoSR conditions on 512 px images, i.e. 32x32 patches, while the
        encoder was pre-trained at 224 px (14x14).  This mirrors the
        ``interpolate_pos_encoding=True`` path of the original implementation.
        """
        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
            return self.position_embeddings

        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        patch_size = self.patch_embeddings.patch_size
        height = height // patch_size
        width = width // patch_size
        # Shift by a small epsilon so fp32 rounding cannot pick a zero-sized
        # interpolation grid on unusual input sizes.
        w0 = width + 0.1
        h0 = height + 0.1
        patch_pos_embed = patch_pos_embed.reshape(
            1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim
        )
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(
            patch_pos_embed, scale_factor=(h0 / math.sqrt(num_positions),
                                           w0 / math.sqrt(num_positions)),
            mode="bicubic", align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.reshape(1, dim, -1).transpose(1, 2)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(self, pixel_values: torch.Tensor,
                interpolate_pos_encoding: bool = False) -> torch.Tensor:
        _, _, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values)
        cls_tokens = self.cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(
                embeddings, height, width
            )
        else:
            embeddings = embeddings + self.position_embeddings
        return self.dropout(embeddings)


class ViTAttention(nn.Module):
    def __init__(self, hidden_size: int = 768, num_heads: int = 12) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_attention_heads = num_heads
        self.attention_head_size = hidden_size // num_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.shape[:-1] + (
            self.num_attention_heads, self.attention_head_size,
        )
        return x.view(new_shape).transpose(1, 2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        scale = 1.0 / math.sqrt(self.attention_head_size)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) * scale
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_probs, value_layer)
        context = context.transpose(1, 2).reshape(
            hidden_states.shape[0], -1, self.all_head_size
        )
        return context


class ViTSelfOutput(nn.Module):
    def __init__(self, hidden_size: int = 768) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.dense(hidden_states)


class ViTIntermediate(nn.Module):
    def __init__(self, hidden_size: int = 768, intermediate_size: int = 3072) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.dense(hidden_states))


class ViTOutputLayer(nn.Module):
    def __init__(self, hidden_size: int = 768, intermediate_size: int = 3072) -> None:
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.dense(hidden_states)


class ViTLayer(nn.Module):
    """Pre-norm transformer block with the transformers-4.x parameter names."""

    def __init__(self, hidden_size: int = 768, num_heads: int = 12,
                 intermediate_size: int = 3072, layer_norm_eps: float = 1e-12) -> None:
        super().__init__()
        self.layernorm_before = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attention = nn.Module()
        self.attention.attention = ViTAttention(hidden_size, num_heads)
        self.attention.output = ViTSelfOutput(hidden_size)
        self.layernorm_after = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.intermediate = ViTIntermediate(hidden_size, intermediate_size)
        self.output = ViTOutputLayer(hidden_size, intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.attention.attention(
            self.layernorm_before(hidden_states)
        )
        hidden_states = self.attention.output(hidden_states) + residual
        residual = hidden_states
        hidden_states = self.output(self.intermediate(
            self.layernorm_after(hidden_states)
        ))
        return hidden_states + residual


class ViTModel(nn.Module):
    """ViT-B/16 encoder matching the original ``transformers`` parameter layout."""

    def __init__(self, hidden_size: int = 768, num_hidden_layers: int = 12,
                 num_attention_heads: int = 12, intermediate_size: int = 3072,
                 image_size: int = 224, patch_size: int = 16,
                 num_channels: int = 3, layer_norm_eps: float = 1e-12,
                 add_pooling_layer: bool = True) -> None:
        super().__init__()
        self.embeddings = ViTEmbeddings(
            num_channels, image_size, patch_size, hidden_size
        )
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([
            ViTLayer(hidden_size, num_attention_heads, intermediate_size,
                     layer_norm_eps)
            for _ in range(num_hidden_layers)
        ])
        self.layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.pooler = None
        if add_pooling_layer:
            self.pooler = nn.Module()
            self.pooler.dense = nn.Linear(hidden_size, hidden_size)

    def forward(self, pixel_values: torch.Tensor,
                interpolate_pos_encoding: bool = False) -> ViTOutput:
        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding)
        for layer in self.encoder.layer:
            hidden_states = layer(hidden_states)
        sequence_output = self.layernorm(hidden_states)
        pooled = None
        if self.pooler is not None:
            pooled = torch.tanh(self.pooler.dense(sequence_output[:, 0]))
        return ViTOutput(last_hidden_state=sequence_output, pooler_output=pooled)


def dino_vit_b16() -> ViTModel:
    """The architecture TripoSR's image tokenizer was trained on."""
    return ViTModel(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        image_size=224,
        patch_size=16,
        num_channels=3,
        layer_norm_eps=1e-12,
    )