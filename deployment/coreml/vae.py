# Copyright 2024 The HuggingFace Team. All rights reserved.
# Copyright 2026 MobileI2V contributors.
# SPDX-License-Identifier: Apache-2.0
"""Inference-only Turbo-VAED-LTX decoder extracted from DiffusersKit.

Preserves the local decoder's activations, channel normalization, temporal
boundary padding, upsampling order and parameter names. This is not the
standard LTX-Video decoder and must be paired with Turbo-VAED weights.
"""

import torch
from torch import nn
from torch.nn import functional as F
from diffusers.models.normalization import RMSNorm


class CausalConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, causal=False, depthwise=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.causal = causal
        self.depthwise = depthwise
        if depthwise:
            self.depthwise_conv = nn.Conv3d(
                in_channels, in_channels, kernel_size, padding=(0, kernel_size // 2, kernel_size // 2),
                groups=in_channels,
            )
            self.pointwise_conv = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                                  padding=(0, kernel_size // 2, kernel_size // 2))

    def forward(self, x):
        # The local depthwise variant uses symmetric temporal padding.
        left = self.kernel_size - 1 if self.causal and not self.depthwise else (self.kernel_size - 1) // 2
        right = 0 if self.causal and not self.depthwise else (self.kernel_size - 1) // 2
        if left:
            x = torch.cat((x[:, :, :1].repeat(1, 1, left, 1, 1), x), dim=2)
        if right:
            x = torch.cat((x, x[:, :, -1:].repeat(1, 1, right, 1, 1)), dim=2)
        if self.depthwise:
            return self.pointwise_conv(self.depthwise_conv(x))
        return self.conv(x)


class ResnetBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, causal, depthwise, kernel_size, upsampled=False):
        super().__init__()
        self.norm1 = RMSNorm(in_channels, eps=1e-8, elementwise_affine=False)
        self.norm2 = RMSNorm(out_channels, eps=1e-8, elementwise_affine=False)
        kernel = kernel_size if depthwise else 3
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel, causal, depthwise)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel, causal, depthwise)
        self.nonlinearity = nn.SiLU()
        self.first_activation = nn.ReLU() if upsampled else nn.SiLU()
        self.norm3 = None
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.norm3 = nn.LayerNorm(in_channels, eps=1e-6)
            self.conv_shortcut = CausalConv3d(in_channels, out_channels, 1, causal, depthwise)

    def forward(self, x):
        hidden = self.norm1(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        hidden = self.conv1(self.first_activation(hidden))
        hidden = self.norm2(hidden.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        hidden = self.conv2(self.nonlinearity(hidden))
        if self.norm3 is not None:
            x = self.norm3(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
            x = self.conv_shortcut(x)
        return hidden + x


class Upsampler3d(nn.Module):
    def __init__(self, channels, causal):
        super().__init__()
        self.conv = CausalConv3d(channels, channels * 8, causal=causal)

    def forward(self, x):
        batch, channels, frames, height, width = x.shape
        x = self.conv(x).reshape(batch, channels, 2, 2, 2, frames, height, width)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return x[:, :, 1:]


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layers, causal, depthwise, kernel, upsample=False):
        super().__init__()
        self.conv_in = None
        if in_channels != out_channels:
            self.conv_in = ResnetBlock3d(in_channels, out_channels, causal, depthwise, kernel)
        self.upsamplers = nn.ModuleList([Upsampler3d(out_channels, causal)]) if upsample else None
        self.resnets = nn.ModuleList([
            ResnetBlock3d(out_channels, out_channels, causal, depthwise, kernel, upsample)
            for _ in range(layers)
        ])

    def forward(self, x):
        if self.conv_in is not None:
            x = self.conv_in(x)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                x = upsampler(x)
        for block in self.resnets:
            x = block(x)
        return x


class SpatialUpsampler(nn.Module):
    def __init__(self, channels, framewise):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.framewise = framewise

    def forward(self, x):
        if self.framewise:
            return torch.stack([F.pixel_shuffle(self.conv(x[:, :, t]), 2) for t in range(x.shape[2])], dim=2)
        batch, channels, frames, height, width = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        x = F.pixel_shuffle(self.conv(x), 2)
        return x.reshape(batch, frames, channels // 4, height * 2, width * 2).permute(0, 2, 1, 3, 4)


class TurboVAEDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Do not silently accept unsupported architecture changes.
        unsupported = {
            "timestep_conditioning": False, "decoder_inject_noise": [False] * 5,
            "upsample_residual": [False] * 4, "upsample_factor": [1] * 4,
            "decoder_spatio_only": [False] * 4, "patch_size": 4, "patch_size_t": 1,
        }
        for key, default in unsupported.items():
            if config.get(key, default) != default:
                raise ValueError(f"Unsupported Turbo-VAED setting: {key}")
        channels = list(reversed(config["decoder_block_out_channels"]))
        layers = list(reversed(config["decoder_layers_per_block"]))
        depthwise = list(reversed(config.get("decoder_is_dw_conv", [False] * 5)))
        scaling = list(reversed(config.get("decoder_spatio_temporal_scaling", [True, True, True, False])))
        if len(channels) != 4 or len(layers) != 5 or len(depthwise) != 5 or scaling != [False, True, True, True]:
            raise ValueError("Expected four Turbo-VAED blocks with 8x temporal and 32x spatial scaling")
        self.latent_channels = config.get("latent_channels", 128)
        self.out_channels = config.get("out_channels", 3)
        causal = config.get("decoder_causal", False)
        kernel = config.get("decoder_dw_kernel_size", 3)
        self.conv_in = CausalConv3d(self.latent_channels, channels[0], causal=causal)
        self.mid_block = DecoderBlock(channels[0], channels[0], layers[0], causal, depthwise[0], kernel)
        self.up_blocks = nn.ModuleList([
            DecoderBlock(channels[max(i - 1, 0)], channels[i], layers[i + 1],
                         causal, depthwise[i + 1], kernel, scaling[i])
            for i in range(4)
        ])
        last = channels[-1]
        self.norm_up_1 = RMSNorm(last, eps=1e-8, elementwise_affine=False)
        self.upsampler2d_1 = SpatialUpsampler(last, framewise=True)
        self.norm_up_2 = RMSNorm(last // 4, eps=1e-8, elementwise_affine=False)
        self.upsampler2d_2 = SpatialUpsampler(last // 4, framewise=False)
        self.conv_act = nn.SiLU()
        self.conv_out = CausalConv3d(last // 16, self.out_channels, causal=causal)

    def forward(self, latents):
        x = self.mid_block(self.conv_in(latents))
        for block in self.up_blocks:
            x = block(x)
        x = self.norm_up_1(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        x = self.upsampler2d_1(self.conv_act(x))
        x = self.norm_up_2(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        x = self.upsampler2d_2(self.conv_act(x))
        # This is variance normalization, not RMS normalization, in the source.
        x = x * torch.rsqrt(x.var(dim=1, unbiased=False, keepdim=True) + 1e-8)
        # Stable Core ML output contract: [batch, channels, frames, height, width].
        return self.conv_out(self.conv_act(x))
