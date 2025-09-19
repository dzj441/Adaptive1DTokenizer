

import types
import math
from typing import Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def _apply_activation(x: torch.Tensor, act: str, alpha: float = 0.2):
    if act is None or act == 'linear':
        return x
    act = act.lower()
    if act in ('relu',):
        return F.relu(x)
    if act in ('lrelu', 'leakyrelu', 'leaky_relu'):
        return F.leaky_relu(x, negative_slope=alpha)
    if act in ('silu', 'swish'):
        return F.silu(x)
    if act in ('gelu',):
        return F.gelu(x)
    if act in ('tanh',):
        return torch.tanh(x)
    if act in ('sigmoid',):
        return torch.sigmoid(x)
    if act in ('softplus',):
        return F.softplus(x)
    raise ValueError(f'Unsupported activation: {act}')

@torch.no_grad()
def _reshape_bias_for_dim(b: torch.Tensor, x: torch.Tensor, dim: int) -> torch.Tensor:
    if b is None or b.numel() == 0:
        return None
    if dim < 0:
        dim += x.ndim
    shape = [1] * x.ndim
    shape[dim] = -1
    return b.view(shape).to(dtype=x.dtype, device=x.device)

def bias_act(x: torch.Tensor,
             b: torch.Tensor = None,
             dim: int = 1,
             act: str = 'linear',
             alpha: float = 0.2,
             gain: float = 1.0,
             clamp: float = None) -> torch.Tensor:

    if b is not None and b.numel() > 0:
        b = _reshape_bias_for_dim(b, x, dim)
        x = x + b
    x = _apply_activation(x, act, alpha)
    if gain is not None and gain != 1.0:
        x = x * gain
    if clamp is not None:
        x = torch.clamp(x, -clamp, clamp)
    return x

class AddReadout(nn.Module):
    def __init__(self, start_index: bool = 1):
        super(AddReadout, self).__init__()
        self.start_index = start_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.start_index == 2:
            readout = (x[:, 0] + x[:, 1]) / 2
        else:
            readout = x[:, 0]
        return x[:, self.start_index:] + readout.unsqueeze(1)


class Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(self.dim0, self.dim1)
        return x.contiguous()

class ResidualBlock(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.fn(x) + x) / np.sqrt(2)


class FullyConnectedLayer(nn.Module):
    def __init__(
        self,
        in_features: int,              # Number of input features.
        out_features: int,             # Number of output features.
        bias: bool  = True,            # Apply additive bias before the activation function?
        activation: str   = 'linear',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier: float = 1.0,    # Learning rate multiplier.
        weight_init: float = 1.0,      # Initial standard deviation of the weight tensor.
        bias_init: float = 0.0,        # Initial value for the additive bias.
    ):

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) * (weight_init / lr_multiplier))
        bias_init = np.broadcast_to(np.asarray(bias_init, dtype=np.float32), [out_features])
        self.bias = torch.nn.Parameter(torch.from_numpy(bias_init / lr_multiplier)) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self) -> str:
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'

# DINO hooks for middle states
def forward_vit(pretrained: nn.Module, x: torch.Tensor) -> dict:
    _, _, H, W = x.size()
    _ = pretrained.model.forward_flex(x)
    return {k: pretrained.rearrange(v) for k, v in activations.items()}


def _resize_pos_embed(self, posemb: torch.Tensor, gs_h: int, gs_w: int) -> torch.Tensor:
    posemb_tok, posemb_grid = (
        posemb[:, : self.start_index],
        posemb[0, self.start_index :],
    )

    gs_old = int(math.sqrt(len(posemb_grid)))

    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear", align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)

    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)

    return posemb


def forward_flex(self, x: torch.Tensor) -> torch.Tensor:
    # patch proj and dynamically resize
    B, C, H, W = x.size()
    x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
    pos_embed = self._resize_pos_embed(
        self.pos_embed, H // self.patch_size[1], W // self.patch_size[0]
    )

    # add cls token
    cls_tokens = self.cls_token.expand(
        x.size(0), -1, -1
    )
    x = torch.cat((cls_tokens, x), dim=1)

    # forward pass
    x = x + pos_embed
    x = self.pos_drop(x)

    for blk in self.blocks:
        x = blk(x)

    x = self.norm(x)
    return x


activations = {}


def get_activation(name: str):
    def hook(model, input, output):
        activations[name] = output
    return hook


def make_vit_backbone(
    model: nn.Module,
    patch_size: list[int] = [16, 16],
    hooks: list[int] = [2, 5, 8, 11],
    hook_patch: bool = True,
    start_index: list[int] = 1,
):
    assert len(hooks) == 4

    pretrained = nn.Module()
    pretrained.model = model

    # add hooks
    pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation('0'))
    pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation('1'))
    pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation('2'))
    pretrained.model.blocks[hooks[3]].register_forward_hook(get_activation('3'))
    if hook_patch:
        pretrained.model.pos_drop.register_forward_hook(get_activation('4'))

    # configure readout
    pretrained.rearrange = nn.Sequential(AddReadout(start_index), Transpose(1, 2))
    pretrained.model.start_index = start_index
    pretrained.model.patch_size = patch_size

    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)
    pretrained.model._resize_pos_embed = types.MethodType(
        _resize_pos_embed, pretrained.model
    )

    return pretrained