"""This file contains some base implementation for discrminators.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 

TODO: Add reference to Mark Weber's tech report on the improved discriminator architecture.
"""
import functools
import math
from typing import Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import SpectralNorm
from torchvision.transforms import RandomCrop, Normalize
import timm

from .maskgit_vqgan import Conv2dSame
from .diffAug import DiffAugment
from .dino_discrim_utils import forward_vit,make_vit_backbone,ResidualBlock,FullyConnectedLayer


class BlurBlock(torch.nn.Module):
    def __init__(self,
                 kernel: Tuple[int] = (1, 3, 3, 1)
                 ):
        super().__init__()

        kernel = torch.tensor(kernel, dtype=torch.float32, requires_grad=False)
        kernel = kernel[None, :] * kernel[:, None]
        kernel /= kernel.sum()
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        self.register_buffer("kernel", kernel)

    def calc_same_pad(self, i: int, k: int, s: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ic, ih, iw = x.size()[-3:]
        pad_h = self.calc_same_pad(i=ih, k=4, s=2)
        pad_w = self.calc_same_pad(i=iw, k=4, s=2)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])

        weight = self.kernel.expand(ic, -1, -1, -1)

        out = F.conv2d(input=x, weight=weight, stride=2, groups=x.shape[1])
        return out

# traditional PATCHGAN discriminator
class NLayerDiscriminator(torch.nn.Module):
    def __init__(
        self,
        num_channels: int = 3,
        hidden_channels: int = 128,
        num_stages: int = 3,
        blur_resample: bool = True,
        blur_kernel_size: int = 4
    ):
        """ Initializes the NLayerDiscriminator.

        Args:
            num_channels -> int: The number of input channels.
            hidden_channels -> int: The number of hidden channels.
            num_stages -> int: The number of stages.
            blur_resample -> bool: Whether to use blur resampling.
            blur_kernel_size -> int: The blur kernel size.
        """
        super().__init__()
        assert num_stages > 0, "Discriminator cannot have 0 stages"
        assert (not blur_resample) or (blur_kernel_size >= 3 and blur_kernel_size <= 5), "Blur kernel size must be in [3,5] when sampling]"

        in_channel_mult = (1,) + tuple(map(lambda t: 2**t, range(num_stages)))
        init_kernel_size = 5
        activation = functools.partial(torch.nn.LeakyReLU, negative_slope=0.1)

        self.block_in = torch.nn.Sequential(
            Conv2dSame(
                num_channels,
                hidden_channels,
                kernel_size=init_kernel_size
            ),
            activation(),
        )

        BLUR_KERNEL_MAP = {
            3: (1,2,1),
            4: (1,3,3,1),
            5: (1,4,6,4,1),
        }

        discriminator_blocks = []
        for i_level in range(num_stages):
            in_channels = hidden_channels * in_channel_mult[i_level]
            out_channels = hidden_channels * in_channel_mult[i_level + 1]
            block = torch.nn.Sequential(
                Conv2dSame(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                ),
                torch.nn.AvgPool2d(kernel_size=2, stride=2) if not blur_resample else BlurBlock(BLUR_KERNEL_MAP[blur_kernel_size]),
                torch.nn.GroupNorm(32, out_channels),
                activation(),
            )
            discriminator_blocks.append(block)

        self.blocks = torch.nn.ModuleList(discriminator_blocks)

        self.pool = torch.nn.AdaptiveMaxPool2d((16, 16))

        self.to_logits = torch.nn.Sequential(
            Conv2dSame(out_channels, out_channels, 1),
            activation(),
            Conv2dSame(out_channels, 1, kernel_size=5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass.

        Args:
            x -> torch.Tensor: The input tensor.

        Returns:
            output -> torch.Tensor: The output tensor.
        """
        hidden_states = self.block_in(x)
        for block in self.blocks:
            hidden_states = block(hidden_states)

        hidden_states = self.pool(hidden_states)

        return self.to_logits(hidden_states)

# frozen dinov2 discriminator
class SpectralConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SpectralNorm.apply(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12)


class BatchNormLocal(nn.Module):
    def __init__(self, num_features: int, affine: bool = True, virtual_bs: int = 8, eps: float = 1e-5):
        super().__init__()
        self.virtual_bs = virtual_bs
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()

        # Reshape batch into groups.
        G = np.ceil(x.size(0) / self.virtual_bs).astype(int)
        x = x.view(G, -1, x.size(-2), x.size(-1))

        # Calculate stats.
        mean = x.mean([1, 3], keepdim=True)
        var = x.var([1, 3], keepdim=True, unbiased=False)
        x = (x - mean) / (torch.sqrt(var + self.eps))

        if self.affine:
            x = x * self.weight[None, :, None] + self.bias[None, :, None]

        return x.view(shape)


def make_block(channels: int, kernel_size: int) -> nn.Module:
    return nn.Sequential(
        SpectralConv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode='circular',
        ),
        BatchNormLocal(channels),
        nn.LeakyReLU(0.2, True),
    )


class DiscHead(nn.Module):
    def __init__(self, channels: int, c_dim: int, cmap_dim: int = 64):
        super().__init__()
        self.channels = channels
        self.c_dim = c_dim
        self.cmap_dim = cmap_dim

        self.main = nn.Sequential(
            make_block(channels, kernel_size=1),
            ResidualBlock(make_block(channels, kernel_size=9))
        )

        if self.c_dim > 0:
            self.cmapper = FullyConnectedLayer(self.c_dim, cmap_dim)
            self.cls = SpectralConv1d(channels, cmap_dim, kernel_size=1, padding=0)
        else:
            self.cls = SpectralConv1d(channels, 1, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.main(x)
        out = self.cls(h)

        if self.c_dim > 0:
            cmap = self.cmapper(c).unsqueeze(-1)
            out = (out * cmap).sum(1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        return out


# -------------------------
# DINOv2 backbone (frozen)
# -------------------------

class DINOv2Backbone(nn.Module):
    """
    Wrap a timm DINOv2 ViT and expose intermediate features via vit_utils hooks.
    Default: vit_small_patch14_dinov2.lvd142m  (patch=14, pretrain img_size ~518)
    """
    def __init__(
        self,
        hooks: list[int] = [2, 5, 8, 11],
        hook_patch: bool = True,
        dinov2_name: str = 'vit_small_patch14_dinov2.lvd142m',
    ):
        super().__init__()
        self.n_hooks = len(hooks) + int(hook_patch)

        # Create timm model (pretrained) and wrap as ViT backbone with hooks.
        timm_model = timm.create_model(dinov2_name, pretrained=True, num_classes=0, global_pool='')
        self.model = make_vit_backbone(
            timm_model,
            patch_size=[14, 14],   # DINOv2 = patch14
            hooks=hooks,
            hook_patch=hook_patch,
        )
        self.model = self.model.eval().requires_grad_(False)

        # Use model's default config for input size & normalization to avoid mismatch.
        cfg = getattr(timm_model, 'default_cfg', {}) or {}
        mean = cfg.get('mean', (0.485, 0.456, 0.406))
        std = cfg.get('std', (0.229, 0.224, 0.225))
        input_size = cfg.get('input_size', (3, 518, 518))
        self.img_resolution = int(input_size[-1])
        self.embed_dim = getattr(timm_model, 'embed_dim', 384)
        self.norm = Normalize(mean, std)

    # @torch.no_grad()
    def forward(self, x: torch.Tensor) -> dict:
        """ input: x in [0,1]; output: dict of activations (tokens as sequence) """
        # Resize to native resolution expected by DINOv2 weights.
        x = F.interpolate(x, self.img_resolution, mode='area')
        x = self.norm(x)
        feats = forward_vit(self.model, x)
        return feats


# -------------------------
# Projected Discriminator (DINOv2)
# -------------------------

class DINODiscriminator(nn.Module):
    """
    Projected-GAN discriminator with a frozen DINOv2 ViT feature network.
    - Multi-head over multiple transformer blocks (+ optional patch tokens).
    - c_dim can be 0 (unconditional) for ImageTokenization use.
    """
    def __init__(self, c_dim: int = 0, diffaug: bool = True, p_crop: float = 0.5,
                 hooks: list[int] = [2, 5, 8, 11], hook_patch: bool = True,
                 dinov2_name: str = 'vit_small_patch14_dinov2.lvd142m'):
        super().__init__()
        self.c_dim = c_dim
        self.diffaug = diffaug
        self.p_crop = p_crop

        # DINOv2 backbone (frozen)
        self.dino = DINOv2Backbone(hooks=hooks, hook_patch=hook_patch, dinov2_name=dinov2_name)

        # Per-scale heads (share channel dim = embed_dim)
        heads = []
        for i in range(self.dino.n_hooks):
            heads += [str(i), DiscHead(self.dino.embed_dim, c_dim)],
        self.heads = nn.ModuleDict(heads)

    def train(self, mode: bool = True):
        # Keep backbone frozen; only heads train.
        self.dino = self.dino.train(False)
        self.heads = self.heads.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def forward(self, x: torch.Tensor, c: torch.Tensor = None) -> torch.Tensor:
        # DiffAug (expects [-1,1]).
        if self.diffaug:
            x = DiffAugment(x, policy='color,translation,cutout')

        # Map to [0,1].
        x = x.add(1).div(2)

        # Optional crop: if larger than backbone's native res, randomly crop with prob p_crop.
        if x.size(-1) > self.dino.img_resolution and np.random.random() < self.p_crop:
            x = RandomCrop(self.dino.img_resolution)(x)

        # Forward DINOv2 backbone (frozen).
        features = self.dino(x)

        # Heads over each hooked scale/token set.
        if c is None and self.c_dim > 0:
            # If a conditional dim is configured but c is not provided, use zeros.
            # (Safe default; callers can pass real condition later.)
            c = x.new_zeros(x.size(0), self.c_dim)

        logits = []
        for k, head in self.heads.items():
            logits.append(head(features[k], c if self.c_dim > 0 else None).view(x.size(0), -1))
        logits = torch.cat(logits, dim=1)
        return logits # [B,L]
