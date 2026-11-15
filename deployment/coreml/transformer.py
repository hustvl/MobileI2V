# Copyright 2024 The HuggingFace Team. All rights reserved.
# Copyright 2026 MobileI2V contributors.
# SPDX-License-Identifier: Apache-2.0
"""Inference-only extraction of DiffusersKit's LTXSanaTransformer3DModel.

State-dict names are preserved for the existing Diffusers-format checkpoints.
No custom Diffusers installation or training dependencies are required.
"""

import torch
from torch import nn
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import PatchEmbed, PixArtAlphaTextProjection
from diffusers.models.normalization import AdaLayerNormSingle, RMSNorm

from .attention import HybridAttentionProcessor, RotaryEmbedding3D


class GLUMBConv(nn.Module):
    def __init__(self, dim, ratio):
        super().__init__()
        expanded = int(dim * ratio)
        self.conv_inverted = nn.Conv2d(dim, expanded * 2, 1)
        self.conv_depth = nn.Conv2d(expanded * 2, expanded * 2, 3, padding=1, groups=expanded * 2)
        self.conv_point = nn.Conv2d(expanded, dim, 1, bias=False)
        self.nonlinearity = nn.SiLU()

    def forward(self, x):
        x = self.conv_depth(self.nonlinearity(self.conv_inverted(x)))
        x, gate = x.chunk(2, dim=1)
        return self.conv_point(x * self.nonlinearity(gate))


class HybridBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, ratio, norm_eps, linear, rotary, optimized):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=norm_eps)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=norm_eps)
        self.attn1 = Attention(
            query_dim=dim, heads=heads, kv_heads=heads, dim_head=head_dim,
            bias=not linear, qk_norm="rms_norm_across_heads" if linear else None,
            processor=HybridAttentionProcessor(rotary, linear, optimized),
        )
        self.ff = GLUMBConv(dim, ratio)
        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim ** 0.5)

    def forward(self, x, timestep, latent_shape):
        batch, tokens, dim = x.shape
        shift, scale, gate, ff_shift, ff_scale, ff_gate = (
            self.scale_shift_table[None] + timestep.reshape(batch, tokens, 6, dim)
        ).unbind(dim=2)
        x = x + gate * self.attn1(self.norm1(x) * (1 + scale) + shift)
        frames, height, width = latent_shape
        hidden = self.norm2(x) * (1 + ff_scale) + ff_shift
        hidden = hidden.reshape(batch * frames, height, width, dim).permute(0, 3, 1, 2)
        hidden = self.ff(hidden).flatten(2).transpose(1, 2).reshape(batch, tokens, dim)
        return x + ff_gate * hidden


class OutputNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, embedded_timestep, scale_shift_table):
        shift, scale = (scale_shift_table[None] + embedded_timestep[:, :, None]).unbind(dim=2)
        return self.norm(x) * (1 + scale) + shift


class MobileI2VTransformer(nn.Module):
    def __init__(self, config, latent_shape, interpolation, optimized=True):
        super().__init__()
        # Legacy checkpoints count only the 14 linear blocks in num_layers.
        if config.get("num_layers", 14) != 14 or config.get("patch_size", 1) != 1:
            raise ValueError("Only the 14-linear + 2-global, patch_size=1 hybrid is supported")
        if config.get("norm_elementwise_affine", False) or config.get("dropout", 0) != 0:
            raise ValueError("Only the existing inference architecture is supported")
        self.latent_shape = tuple(latent_shape)
        self.in_channels = config.get("in_channels", 128)
        self.out_channels = config.get("out_channels", self.in_channels)
        heads = config.get("num_attention_heads", 16)
        head_dim = config.get("attention_head_dim", 72)
        dim = heads * head_dim
        self.flow = config.get("flow", True)
        self.patch_embed = PatchEmbed(
            height=config.get("sample_size", 16), width=config.get("sample_size", 16),
            patch_size=1, in_channels=self.in_channels, embed_dim=dim, pos_embed_type=None,
        )
        self.time_embed = AdaLayerNormSingle(dim)
        self.flow_embed = AdaLayerNormSingle(dim)
        # These tensors exist in legacy checkpoints, although the hybrid has
        # no cross-attention and its output does not depend on text embeddings.
        self.caption_projection = PixArtAlphaTextProjection(config.get("caption_channels", 896), dim)
        self.caption_norm = RMSNorm(dim, eps=1e-5, elementwise_affine=True)
        self.rotary = RotaryEmbedding3D(head_dim, self.latent_shape, interpolation)
        self.transformer_blocks = nn.ModuleList([
            HybridBlock(dim, heads, head_dim, config.get("mlp_ratio", 2.5),
                        config.get("norm_eps", 1e-6), index % 8 != 7, self.rotary, optimized)
            for index in range(16)
        ])
        self.scale_shift_table = nn.Parameter(torch.randn(2, dim) / dim ** 0.5)
        self.norm_out = OutputNorm(dim)
        self.proj_out = nn.Linear(dim, self.out_channels)

    def forward(self, hidden_states, timestep, cond_mask, flow_score):
        batch = hidden_states.shape[0]
        frames, height, width = self.latent_shape
        hidden = hidden_states.permute(0, 2, 1, 3, 4).reshape(batch * frames, self.in_channels, height, width)
        hidden = self.patch_embed(hidden).reshape(batch, frames * height * width, -1)
        token_timestep = torch.minimum(timestep[:, None], (1 - cond_mask) * 1000) / 1000
        # Public Diffusers accepts flat timesteps. Flattening here replaces the
        # local fork's extension to its shared sinusoidal embedding function.
        modulation, embedded = self.time_embed(token_timestep.flatten(), hidden_dtype=hidden.dtype)
        modulation = modulation.reshape(batch, frames * height * width, -1)
        embedded = embedded.reshape(batch, frames * height * width, -1)
        if self.flow:
            flow = flow_score[:, None].expand_as(token_timestep)
            flow_modulation, _ = self.flow_embed(flow.flatten(), hidden_dtype=hidden.dtype)
            modulation = modulation + flow_modulation.reshape_as(modulation)
        for block in self.transformer_blocks:
            hidden = block(hidden, modulation, self.latent_shape)
        hidden = self.proj_out(self.norm_out(hidden, embedded, self.scale_shift_table))
        return hidden.reshape(batch, frames, height, width, self.out_channels).permute(0, 4, 1, 2, 3)
