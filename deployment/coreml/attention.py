# Copyright 2024 The HuggingFace Team. All rights reserved.
# Copyright 2026 MobileI2V contributors.
# SPDX-License-Identifier: Apache-2.0
"""3D RoPE and channels-first, per-head attention from DiffusersKit.

Only self-attention is needed by the released 7-linear/1-global hybrid.
The reference path uses batched matmuls / SDPA; the export path uses 4D
per-head einsums. Both use the same projections, QK normalization and RoPE.
"""

import torch
from torch import nn
from torch.nn import functional as F


class RotaryEmbedding3D(nn.Module):
    def __init__(self, head_dim, latent_shape, interpolation):
        super().__init__()
        if head_dim % 6:
            raise ValueError("3D RoPE requires head_dim divisible by 6")
        axis_dim = head_dim // 3
        positions = torch.meshgrid(
            *(torch.arange(n, dtype=torch.float32) for n in latent_shape), indexing="ij"
        )
        frequency = 1.0 / (10000 ** (torch.arange(0, axis_dim, 2).float() / axis_dim))
        phases = []
        for position, scale in zip(positions, interpolation):
            phase = position.reshape(-1, 1) / scale * frequency
            phases.append(torch.cat((phase, phase), dim=-1))
        phase = torch.cat(phases, dim=-1)[None, None]
        self.register_buffer("cos", phase.cos(), persistent=False)
        self.register_buffer("sin", phase.sin(), persistent=False)

    def forward(self, tokens):
        rotated = []
        for axis in tokens.chunk(3, dim=-1):
            left, right = axis.chunk(2, dim=-1)
            rotated.append(torch.cat((-right, left), dim=-1))
        return tokens * self.cos + torch.cat(rotated, dim=-1) * self.sin


class HybridAttentionProcessor:
    def __init__(self, rotary, linear, optimized=True):
        self.rotary = rotary
        self.linear = linear
        self.optimized = optimized

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        if encoder_hidden_states is not None or attention_mask is not None:
            raise ValueError("This hybrid processor supports unmasked self-attention only")
        batch, length, dim = hidden_states.shape
        head_dim = dim // attn.heads
        query, key, value = (projection(hidden_states) for projection in (attn.to_q, attn.to_k, attn.to_v))
        if attn.norm_q is not None:
            query, key = attn.norm_q(query), attn.norm_k(key)
        query, key, value = (
            tensor.reshape(batch, length, attn.heads, head_dim).transpose(1, 2)
            for tensor in (query, key, value)
        )
        query, key = self.rotary(query), self.rotary(key)

        if self.linear:
            query, key, value = F.relu(query).float(), F.relu(key).float(), value.float()
            if self.optimized:
                outputs = []
                for head in range(attn.heads):
                    q = query[:, head].transpose(1, 2).unsqueeze(2)
                    k = key[:, head].unsqueeze(2)
                    v = value[:, head].transpose(1, 2).unsqueeze(2)
                    ones = torch.ones_like(v[:, :1])
                    scores = torch.einsum("bchw,bwhd->bchd", torch.cat((v, ones), dim=1), k)
                    output = torch.einsum("bchd,bdhq->bchq", scores, q)
                    outputs.append(output[:, :-1] / (output[:, -1:] + 1e-15))
                output = torch.cat(outputs, dim=1).squeeze(2).transpose(1, 2)
            else:
                value = F.pad(value.transpose(-1, -2), (0, 0, 0, 1), value=1.0)
                scores = value @ key
                output = scores @ query.transpose(-1, -2)
                output = output[:, :, :-1] / (output[:, :, -1:] + 1e-15)
                output = output.flatten(1, 2).transpose(1, 2)
        elif self.optimized:
            outputs = []
            for head in range(attn.heads):
                q = query[:, head].transpose(1, 2).unsqueeze(2)
                k = key[:, head].unsqueeze(2)
                v = value[:, head].transpose(1, 2).unsqueeze(2)
                weights = torch.einsum("bchq,bkhc->bkhq", q, k) * head_dim ** -0.5
                outputs.append(torch.einsum("bkhq,bchk->bchq", weights.softmax(dim=1), v))
            output = torch.cat(outputs, dim=1).squeeze(2).transpose(1, 2)
        else:
            output = F.scaled_dot_product_attention(query, key, value)
            output = output.transpose(1, 2).reshape(batch, length, dim)

        output = output.to(hidden_states.dtype)
        return attn.to_out[1](attn.to_out[0](output))
