import os

import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from einops import rearrange

from diffusion.model.builder import MODELS
from diffusion.model.nets.basic_modules import DWMlp, GLUMBConv, MBConvPreGLU, Mlp
from diffusion.model.nets.sana_blocks import (
    Attention,
    CaptionEmbedder,
    FlashAttention,
    LiteLA,
    MultiHeadCrossAttention,
    PatchEmbed,
    PatchEmbed_3D,
    T2IFinalLayer,
    TimestepEmbedder,
    t2i_modulate,
)
from diffusion.model.norms import RMSNorm
from diffusion.model.utils import auto_grad_checkpoint, to_2tuple
from diffusion.utils.dist_utils import get_rank
from diffusion.utils.import_utils import is_triton_module_available
from diffusion.utils.logger import get_root_logger

_triton_modules_available = False
if is_triton_module_available():
    from diffusion.model.nets.fastlinear.modules import TritonLiteMLA, TritonMBConvPreGLU

    _triton_modules_available = True


class SanaBlock(nn.Module):
    """
    A Sana block with global shared adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0,
        input_size=None,
        qk_norm=False,
        attn_type="flash",
        ffn_type="mlp",
        mlp_acts=("silu", "silu", None),
        linear_head_dim=32,
        **block_kwargs,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if attn_type == "flash":
            # flash self attention
            self.attn = FlashAttention(
                hidden_size,
                num_heads=num_heads,
                qkv_bias=True,
                qk_norm=qk_norm,
                **block_kwargs,
            )
        elif attn_type == "linear":
            # linear self attention
            # TODO: Here the num_heads set to 36 for tmp used
            linear_head_dim = 72
            self_num_heads = hidden_size // linear_head_dim
            self.attn = LiteLA(hidden_size, hidden_size, heads=self_num_heads, eps=1e-8, qk_norm=qk_norm)
        elif attn_type == "triton_linear":
            if not _triton_modules_available:
                raise ValueError(
                    f"{attn_type} type is not available due to _triton_modules_available={_triton_modules_available}."
                )
            # linear self attention with triton kernel fusion
            # TODO: Here the num_heads set to 36 for tmp used
            self_num_heads = hidden_size // linear_head_dim
            self.attn = TritonLiteMLA(hidden_size, num_heads=self_num_heads, eps=1e-8)
        elif attn_type == "vanilla":
            # vanilla self attention
            self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        else:
            raise ValueError(f"{attn_type} type is not defined.")

        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, qk_norm=qk_norm,**block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # to be compatible with lower version pytorch
        if ffn_type == "dwmlp":
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = DWMlp(
                in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
            )
        elif ffn_type == "glumbconv":
            self.mlp = GLUMBConv(
                in_features=hidden_size,
                hidden_features=int(hidden_size * mlp_ratio),
                use_bias=(True, True, False),
                norm=(None, None, None),
                act=mlp_acts,
            )
        elif ffn_type == "glumbconv_dilate":
            self.mlp = GLUMBConv(
                in_features=hidden_size,
                hidden_features=int(hidden_size * mlp_ratio),
                use_bias=(True, True, False),
                norm=(None, None, None),
                act=mlp_acts,
                dilation=2,
            )
        elif ffn_type == "mbconvpreglu":
            self.mlp = MBConvPreGLU(
                in_dim=hidden_size,
                out_dim=hidden_size,
                mid_dim=int(hidden_size * mlp_ratio),
                use_bias=(True, True, False),
                norm=None,
                act=("silu", "silu", None),
            )
        elif ffn_type == "triton_mbconvpreglu":
            if not _triton_modules_available:
                raise ValueError(
                    f"{ffn_type} type is not available due to _triton_modules_available={_triton_modules_available}."
                )
            self.mlp = TritonMBConvPreGLU(
                in_dim=hidden_size,
                out_dim=hidden_size,
                mid_dim=int(hidden_size * mlp_ratio),
                use_bias=(True, True, False),
                norm=None,
                act=("silu", "silu", None),
            )
        elif ffn_type == "mlp":
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(
                in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
            )
        else:
            raise ValueError(f"{ffn_type} type is not defined.")
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        N = 768
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)
        #self.scale_shift_table = nn.Parameter(torch.randn(6, N, hidden_size) / hidden_size**0.5)

    def forward(self, x, y, t, flow_score, mask=None, H=None, W=None,T=None, S=None,**kwargs):
        B, N, C = x.shape

        #self.scale_shift_table = self.scale_shift_table.unsqueeze(1).repeat(1,N,1)
        #print("scale_shift_table", self.scale_shift_table.shape)
        #print(t.shape)
        #print(flow_score.shape)
        

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, N, 6, -1) + flow_score.reshape(B, N, 6, -1)
        ).chunk(6, dim=2)
        #print(shift_msa.shape,scale_msa.shape)
        #print(x.shape)
        shift_msa = shift_msa.squeeze(2)
        scale_msa = scale_msa.squeeze(2)
        gate_msa = gate_msa.squeeze(2)
        shift_mlp = shift_mlp.squeeze(2)
        scale_mlp = scale_mlp.squeeze(2)
        gate_mlp = gate_mlp.squeeze(2)

        #x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)).reshape(B, N, C))
        x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa)
        x_s = self.attn(x_m)
        x_s = gate_msa * x_s
        x = x + self.drop_path(x_s)
        x = x + self.cross_attn(x, y, mask)
        
        x_m = t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)
        x_m = rearrange(x_m, "B (T S) C -> (B T) S C", T=T, S=S)
        x_mlp = self.mlp(x_m, H,W)
        x_mlp = rearrange(x_mlp, "(B T) S C -> B (T S) C", T=T, S=S)
        x_mlp = gate_mlp * x_mlp 
        x = x + self.drop_path(x_mlp)
        #x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))

        return x



class SanaBlock_cross(nn.Module):
    """
    A Sana block with global shared adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0,
        input_size=None,
        qk_norm=False,
        attn_type="flash",
        ffn_type="mlp",
        mlp_acts=("silu", "silu", None),
        linear_head_dim=32,
        **block_kwargs,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if attn_type == "flash":
            # flash self attention
            self.attn = FlashAttention(
                hidden_size,
                num_heads=num_heads,
                qkv_bias=True,
                qk_norm=qk_norm,
                **block_kwargs,
            )
        elif attn_type == "linear":
            # linear self attention
            # TODO: Here the num_heads set to 36 for tmp used
            linear_head_dim = 72
            self_num_heads = hidden_size // linear_head_dim
            self.attn = LiteLA(hidden_size, hidden_size, heads=self_num_heads, eps=1e-8, qk_norm=qk_norm)
        elif attn_type == "triton_linear":
            if not _triton_modules_available:
                raise ValueError(
                    f"{attn_type} type is not available due to _triton_modules_available={_triton_modules_available}."
                )
            # linear self attention with triton kernel fusion
            # TODO: Here the num_heads set to 36 for tmp used
            self_num_heads = hidden_size // linear_head_dim
            self.attn = TritonLiteMLA(hidden_size, num_heads=self_num_heads, eps=1e-8)
        elif attn_type == "vanilla":
            # vanilla self attention
            self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        else:
            raise ValueError(f"{attn_type} type is not defined.")

        #self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # to be compatible with lower version pytorch
        if ffn_type == "dwmlp":
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = DWMlp(
                in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
            )
        elif ffn_type == "glumbconv":
            self.mlp = GLUMBConv(
                in_features=hidden_size,
                hidden_features=int(hidden_size * mlp_ratio),
                use_bias=(True, True, False),
                norm=(None, None, None),
                act=mlp_acts,
            )
        elif ffn_type == "glumbconv_dilate":
            self.mlp = GLUMBConv(
                in_features=hidden_size,
                hidden_features=int(hidden_size * mlp_ratio),
                use_bias=(True, True, False),
                norm=(None, None, None),
                act=mlp_acts,
                dilation=2,
            )
        elif ffn_type == "mbconvpreglu":
            self.mlp = MBConvPreGLU(
                in_dim=hidden_size,
                out_dim=hidden_size,
                mid_dim=int(hidden_size * mlp_ratio),
                use_bias=(True, True, False),
                norm=None,
                act=("silu", "silu", None),
            )
        elif ffn_type == "triton_mbconvpreglu":
            if not _triton_modules_available:
                raise ValueError(
                    f"{ffn_type} type is not available due to _triton_modules_available={_triton_modules_available}."
                )
            self.mlp = TritonMBConvPreGLU(
                in_dim=hidden_size,
                out_dim=hidden_size,
                mid_dim=int(hidden_size * mlp_ratio),
                use_bias=(True, True, False),
                norm=None,
                act=("silu", "silu", None),
            )
        elif ffn_type == "mlp":
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(
                in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
            )
        else:
            raise ValueError(f"{ffn_type} type is not defined.")
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        N = 768
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)
        #self.scale_shift_table = nn.Parameter(torch.randn(6, N, hidden_size) / hidden_size**0.5)

    def forward(self, x, y, t, flow_score, mask=None, H=None, W=None,T=None, S=None,**kwargs):
        B, N, C = x.shape

        #self.scale_shift_table = self.scale_shift_table.unsqueeze(1).repeat(1,N,1)
        #print("scale_shift_table", self.scale_shift_table.shape)
        #print(t.shape)
        #print(flow_score.shape)
        

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, N, 6, -1) + flow_score.reshape(B, N, 6, -1)
        ).chunk(6, dim=2)
        #print(shift_msa.shape,scale_msa.shape)
        #print(x.shape)
        shift_msa = shift_msa.squeeze(2)
        scale_msa = scale_msa.squeeze(2)
        gate_msa = gate_msa.squeeze(2)
        shift_mlp = shift_mlp.squeeze(2)
        scale_mlp = scale_mlp.squeeze(2)
        gate_mlp = gate_mlp.squeeze(2)

        #x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)).reshape(B, N, C))
        x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa)
        x_s = self.attn(x_m)
        x_s = gate_msa * x_s
        x = x + self.drop_path(x_s)
        #x = x + self.cross_attn(x, y, mask)
        
        x_m = t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)
        x_m = rearrange(x_m, "B (T S) C -> (B T) S C", T=T, S=S)
        x_mlp = self.mlp(x_m, H, W)
        x_mlp = rearrange(x_mlp, "(B T) S C -> B (T S) C", T=T, S=S)
        x_mlp = gate_mlp * x_mlp 
        x = x + self.drop_path(x_mlp)
        #x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))

        return x


class SanaBlock_vanila(nn.Module):
    """
    A Sana block with global shared adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0,
        input_size=None,
        qk_norm=False,
        attn_type="flash",
        ffn_type="mlp",
        mlp_acts=("silu", "silu", None),
        linear_head_dim=32,
        **block_kwargs,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True,qk_norm=qk_norm)

        #self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # to be compatible with lower version pytorch
        if ffn_type == "dwmlp":
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = DWMlp(
                in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
            )
        elif ffn_type == "glumbconv":
            self.mlp = GLUMBConv(
                in_features=hidden_size,
                hidden_features=int(hidden_size * mlp_ratio),
                use_bias=(True, True, False),
                norm=(None, None, None),
                act=mlp_acts,
            )
        elif ffn_type == "glumbconv_dilate":
            self.mlp = GLUMBConv(
                in_features=hidden_size,
                hidden_features=int(hidden_size * mlp_ratio),
                use_bias=(True, True, False),
                norm=(None, None, None),
                act=mlp_acts,
                dilation=2,
            )
        elif ffn_type == "mbconvpreglu":
            self.mlp = MBConvPreGLU(
                in_dim=hidden_size,
                out_dim=hidden_size,
                mid_dim=int(hidden_size * mlp_ratio),
                use_bias=(True, True, False),
                norm=None,
                act=("silu", "silu", None),
            )
        elif ffn_type == "triton_mbconvpreglu":
            if not _triton_modules_available:
                raise ValueError(
                    f"{ffn_type} type is not available due to _triton_modules_available={_triton_modules_available}."
                )
            self.mlp = TritonMBConvPreGLU(
                in_dim=hidden_size,
                out_dim=hidden_size,
                mid_dim=int(hidden_size * mlp_ratio),
                use_bias=(True, True, False),
                norm=None,
                act=("silu", "silu", None),
            )
        elif ffn_type == "mlp":
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(
                in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
            )
        else:
            raise ValueError(f"{ffn_type} type is not defined.")
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        N = 768
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)
        #self.scale_shift_table = nn.Parameter(torch.randn(6, N, hidden_size) / hidden_size**0.5)

    def forward(self, x, y, t, flow_score, mask=None, H=None, W=None,T=None, S=None,**kwargs):
        B, N, C = x.shape

        #self.scale_shift_table = self.scale_shift_table.unsqueeze(1).repeat(1,N,1)
        #print("scale_shift_table", self.scale_shift_table.shape)
        #print(t.shape)
        #print(flow_score.shape)
        

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, N, 6, -1) + flow_score.reshape(B, N, 6, -1)
        ).chunk(6, dim=2)
        #print(shift_msa.shape,scale_msa.shape)
        #print(x.shape)
        shift_msa = shift_msa.squeeze(2)
        scale_msa = scale_msa.squeeze(2)
        gate_msa = gate_msa.squeeze(2)
        shift_mlp = shift_mlp.squeeze(2)
        scale_mlp = scale_mlp.squeeze(2)
        gate_mlp = gate_mlp.squeeze(2)

        #x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)).reshape(B, N, C))
        x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa)
        x_s = self.attn(x_m)
        x_s = gate_msa * x_s
        x = x + self.drop_path(x_s)
        #x = x + self.cross_attn(x, y, mask)
        
        x_m = t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)
        x_m = rearrange(x_m, "B (T S) C -> (B T) S C", T=T, S=S)
        x_mlp = self.mlp(x_m, H,W)
        x_mlp = rearrange(x_mlp, "(B T) S C -> B (T S) C", T=T, S=S)
        x_mlp = gate_mlp * x_mlp 
        x = x + self.drop_path(x_mlp)
        #x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))

        return x


#############################################################################
#                                 Core Sana Model                                #
#################################################################################
@MODELS.register_module()
class Sana(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        #input_size=32,   #DC-AE 32
        input_height=32,
        input_width=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        pred_sigma=True,
        drop_path: float = 0.0,
        caption_channels=2304,
        pe_interpolation=1.0,
        config=None,
        model_max_length=120,
        qk_norm=False,
        y_norm=False,
        norm_eps=1e-5,
        attn_type="flash",
        ffn_type="mlp",
        use_pe=True,
        y_norm_scale_factor=1.0,
        patch_embed_kernel=None,
        mlp_acts=("silu", "silu", None),
        linear_head_dim=32,
        **kwargs,
    ):
        super().__init__()
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pe_interpolation = pe_interpolation
        self.depth = depth
        self.use_pe = use_pe
        self.y_norm = y_norm
        self.fp32_attention = kwargs.get("use_fp32_attention", False)
        self.input_size = (17, input_height, input_width)
        self.hidden_size = hidden_size
        
        num_patches = np.prod([self.input_size[i] // 1 for i in range(3)])
        self.num_patches = num_patches
        self.num_temporal = self.input_size[0] // 1
        self.num_spatial = num_patches // self.num_temporal

        kernel_size = patch_embed_kernel or patch_size
        
        #print("in_channels",in_channels)
        self.x_embedder = PatchEmbed(
            input_height, input_width, patch_size, in_channels, hidden_size, kernel_size=kernel_size, bias=True
        )
        
        #self.x_embedder = PatchEmbed_3D(input_size, patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.flow_embedder = TimestepEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches
        self.base_size = (input_height // self.patch_size, input_width // self.patch_size)
        # Will use fixed sin-cos embedding:
        #self.register_buffer("pos_embed", torch.zeros(1, num_patches, hidden_size))

        self.register_buffer("pos_embed", self.get_spatial_pos_embed())
        self.register_buffer("pos_embed_temporal", self.get_temporal_pos_embed())

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.flow_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
            uncond_prob=class_dropout_prob,
            act_layer=approx_gelu,
            token_num=model_max_length,
        )
        if self.y_norm:
            self.attention_y_norm = RMSNorm(hidden_size, scale_factor=y_norm_scale_factor, eps=norm_eps)
        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth+2)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                SanaBlock_cross(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path[i],
                    input_size=(input_height // patch_size, input_width // patch_size),
                    qk_norm=qk_norm,
                    attn_type=attn_type,
                    ffn_type=ffn_type,
                    mlp_acts=mlp_acts,
                    linear_head_dim=linear_head_dim,
                )
                for i in range(7)
            ]
        )

        #print(drop_path)
        for i in range(1):
            self.blocks.append(
                    SanaBlock_vanila(
                        hidden_size,
                        num_heads,
                        mlp_ratio=mlp_ratio,
                        drop_path=drop_path[14],
                        input_size=(input_height // patch_size, input_width // patch_size),
                        # qk_norm=qk_norm,
                        attn_type=attn_type,
                        ffn_type=ffn_type,
                        mlp_acts=mlp_acts,
                        linear_head_dim=linear_head_dim,
                    )
                )

        
        for i in range(7):
            self.blocks.append(
                    SanaBlock_cross(
                        hidden_size,
                        num_heads,
                        mlp_ratio=mlp_ratio,
                        drop_path=drop_path[i],
                        input_size=(input_height // patch_size, input_width // patch_size),
                        qk_norm=qk_norm,
                        attn_type=attn_type,
                        ffn_type=ffn_type,
                        mlp_acts=mlp_acts,
                        linear_head_dim=linear_head_dim,
                    )
                )
        
        for i in range(1):
            self.blocks.append(
                    SanaBlock_vanila(
                        hidden_size,
                        num_heads,
                        mlp_ratio=mlp_ratio,
                        drop_path=drop_path[14],
                        input_size=(input_height // patch_size, input_width // patch_size),
                        # qk_norm=qk_norm,
                        attn_type=attn_type,
                        ffn_type=ffn_type,
                        mlp_acts=mlp_acts,
                        linear_head_dim=linear_head_dim,
                    )
                )
        

        self.final_layer = T2IFinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

        if config:
            logger = get_root_logger(os.path.join(config.work_dir, "train_log.log"))
            logger = logger.warning
        else:
            logger = print
        if get_rank() == 0:
            logger(
                f"use pe: {use_pe}, position embed interpolation: {self.pe_interpolation}, base size: {self.base_size}"
            )
            logger(
                f"attention type: {attn_type}; ffn type: {ffn_type}; "
                f"autocast linear attn: {os.environ.get('AUTOCAST_LINEAR_ATTN', False)}"
            )



    def get_dynamic_size(self, x):
        _, _, T, H, W = x.size()
        if T % self.patch_size != 0:
            T += self.patch_size - T % self.patch_size
        if H % self.patch_size != 0:
            H += self.patch_size - H % self.patch_size
        if W % self.patch_size != 0:
            W += self.patch_size - W % self.patch_size
        T = T // self.patch_size
        H = H // self.patch_size
        W = W // self.patch_size
        return (T, H, W)

    def forward(self, x, timestep, guide_image, y, cond_mask , flow_score, mask=None, data_info=None, **kwargs):
        """
        Forward pass of Sana.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """

        B = x.shape[0]
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)
        _, _, Tx, Hx, Wx = x.size()
        T, H, W = self.get_dynamic_size(x)
        S = H * W
        pos_embed = self.pos_embed.to(self.dtype)
        self.h, self.w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        #x = rearrange(x, "b t c h w -> b t (c h) w")   # (1, 32, 17, 16, 16)
        pos_embed = self.get_spatial_pos_embed((x.shape[-2], x.shape[-1])).to(x.dtype)
        '''
        if self.use_pe:
            x = self.x_embedder(x) + pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        else:
            x = self.x_embedder(x)
        '''
        #num_temporal=17
        x = rearrange(x, "B C T H W -> (B T) C H W")
        x = self.x_embedder(x)  # (B, N, D)
        x = rearrange(x, "(B T) S C -> B (T S) C", B=B, T=T, S = S)

        timestep = timestep.unsqueeze(1).repeat(1, 2760)  #768
        timestep = torch.min(timestep, (1.0 -cond_mask)*1000)
        timestep = timestep/1000
        t = self.t_embedder(timestep.to(x.dtype))  # (N, D)
        t0 = self.t_block(t)

        flow_score = flow_score.unsqueeze(1).repeat(1, 2760)  #768
        flow_score = self.flow_embedder(flow_score.to(x.dtype))  # (N, D)
        flow_score = self.flow_block(flow_score)

        y = self.y_embedder(y, self.training)  # (N, 1, L, D)
        
        
        if self.y_norm:
            y = self.attention_y_norm(y)
        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])
        
        
        for block in self.blocks:
            x = auto_grad_checkpoint(block, x, y, t0, flow_score, y_lens, H, W ,T, S)  # (N, T, D) #support grad checkpoint
        

        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)

        x = self.unpatchify(x, T, H, W, Tx, Hx, Wx)
        return x

    def __call__(self, *args, **kwargs):
        """
        This method allows the object to be called like a function.
        It simply calls the forward method.
        """
        return self.forward(*args, **kwargs)

    def forward_with_dpmsolver(self, x, timestep, guide_image, y, mask=None, **kwargs):
        """
        dpm solver donnot need variance prediction
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb 
        
        model_out = self.forward(x, timestep, guide_image, y, mask)
        return model_out.chunk(2, dim=1)[0] if self.pred_sigma else model_out
    
    '''
    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    '''

    def unpatchify(self, x, N_t, N_h, N_w, R_t, R_h, R_w):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]

        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """

        # N_t, N_h, N_w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        T_p = H_p = W_p = self.patch_size
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t,
            N_h=N_h,
            N_w=N_w,
            T_p=T_p,
            H_p=H_p,
            W_p=W_p,
            C_out=self.out_channels,
        )
        # unpad
        x = x[:, :, :R_t, :R_h, :R_w]
        return x
    
    def unpatchify_3D(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        t = 17
        h = w = int((x.shape[1]/t) ** 0.5)
        #print(h,w)
        #assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], t, h, w, p, p, c))
        #print(x.shape)
        x = torch.einsum("nthwpqc->ncthpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, t, h * p, w * p))
        #print(x.shape)
        return imgs


    def get_spatial_pos_embed(self, grid_size=None):
        if grid_size is None:
            grid_size = self.input_size[1:]
        pos_embed = get_2d_sincos_pos_embed(
            self.hidden_size,
            (grid_size[0] // self.patch_size, grid_size[1] // self.patch_size),
            pe_interpolation=1,
            base_size=self.base_size,
        )
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        return pos_embed

    def get_temporal_pos_embed(self):
        pos_embed = get_1d_sincos_pos_embed(
            self.hidden_size,
            self.input_size[0] // self.patch_size,
            scale=1.0,
        )
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        return pos_embed

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        if self.use_pe:
            # Initialize (and freeze) pos_embed by sin-cos embedding:
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1],
                int(self.x_embedder.num_patches**0.5),
                pe_interpolation=self.pe_interpolation,
                base_size=self.base_size,
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.flow_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.flow_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.flow_block[1].weight, std=0.02)

        # Initialize caption embedding MLP:
        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

    @property
    def dtype(self):
        return next(self.parameters()).dtype


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, pe_interpolation=1.0, base_size=(16,16)):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_size = to_2tuple(grid_size)
    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size[0]) / pe_interpolation
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size[1]) / pe_interpolation
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_1d_sincos_pos_embed(embed_dim, length, scale=1.0):
    pos = np.arange(0, length)[..., None] / scale
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   Sana Configs                              #
#################################################################################
@MODELS.register_module()
def Sana_600M_P1_D28(**kwargs):
    return Sana(depth=28, hidden_size=1152, patch_size=1, num_heads=16, **kwargs)


@MODELS.register_module()
def Sana_1600M_P1_D20(**kwargs):
    # 20 layers, 1648.48M
    return Sana(depth=20, hidden_size=2240, patch_size=1, num_heads=20, **kwargs)

@MODELS.register_module()
def Sana_300M_P1_D14(**kwargs):
    return Sana(depth=14, hidden_size=1152, patch_size=1, num_heads=16, **kwargs)


@MODELS.register_module()
def Sana_300M_P1_D16(**kwargs):
    return Sana(depth=16, hidden_size=1152, patch_size=1, num_heads=16, **kwargs)

@MODELS.register_module()
def Sana_300M_P1_D20(**kwargs):
    return Sana(depth=20, hidden_size=1152, patch_size=1, num_heads=16, **kwargs)