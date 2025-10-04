# -*- coding: utf-8 -*-
"""
Unified semantic encoder wrapper that reuses your existing load_encoders()
and adds DINOv3 support. The wrapper performs preprocessing inside
forward_features and normalizes all outputs to a common dict API.

modified from https://github.com/Martinser/REG/blob/main/train.py
"""

import os
import torch
import torch.nn as nn
import numpy as np

from torchvision.transforms import Normalize
import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .semantic_encoders import mocov3_vit

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD  = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = IMAGENET_DEFAULT_MEAN
IMAGENET_STD  = IMAGENET_DEFAULT_STD

def fix_mocov3_state_dict(state_dict):
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.base_encoder'):
            # fix naming bug in checkpoint
            new_k = k[len("module.base_encoder."):]
            if "blocks.13.norm13" in new_k:
                new_k = new_k.replace("norm13", "norm1")
            if "blocks.13.mlp.fc13" in k:
                new_k = new_k.replace("fc13", "fc1")
            if "blocks.14.norm14" in k:
                new_k = new_k.replace("norm14", "norm2")
            if "blocks.14.mlp.fc14" in k:
                new_k = new_k.replace("fc14", "fc2")
            # remove prefix
            if 'head' not in new_k and new_k.split('.')[0] != 'fc':
                state_dict[new_k] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    if 'pos_embed' in state_dict.keys():
        state_dict['pos_embed'] = timm.layers.pos_embed.resample_abs_pos_embed(
            state_dict['pos_embed'], [16, 16],
        )
    return state_dict

def preprocess_raw_image(x: torch.Tensor, enc_type: str) -> torch.Tensor:
    """
    Input:
        x: uint8 or float tensor of shape [B, 3, H, W], where H=W is 256 or 512.
    Output:
        x': float tensor normalized and resized to 224*(H//256)
    Notes:
        - For dinov2/dinov3/jepa: ImageNet mean/std + bicubic resize.
        - For clip: CLIP mean/std + bicubic resize.
        - For mocov3/mae/dinov1: ImageNet mean/std (no resize unless specified).
    """
    resolution = x.shape[-1]
    if x.dtype == torch.uint8:
        x = x.float() / 255.0

    if 'clip' in enc_type:
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic', align_corners=False)
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)

    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = Normalize(IMAGENET_MEAN, IMAGENET_STD)(x)

    elif 'dinov2' in enc_type or 'dinov3' in enc_type:
        x = Normalize(IMAGENET_MEAN, IMAGENET_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic', align_corners=False)

    elif 'dinov1' in enc_type:
        x = Normalize(IMAGENET_MEAN, IMAGENET_STD)(x)

    elif 'jepa' in enc_type:
        x = Normalize(IMAGENET_MEAN, IMAGENET_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic', align_corners=False)

    else:
        raise NotImplementedError(f"Unknown encoder type in preprocess: {enc_type}")

    return x


@torch.no_grad() 
def load_encoders(enc_type, resolution=256,local = False,model_path = './dinov2'):
    """
    Return:
        encoders: List[nn.Module], each moved to device and eval()'ed.
                  For DINOv2 hub models, forward_features() already exists.
                  For DINOv3 HF models, we'll adapt in the wrapper.
        encoder_types: List[str]
        architectures: List[str]
    """
    assert (resolution == 256) or (resolution == 512)
    
    enc_names = enc_type.split(',')
    encoders, architectures, encoder_types = [], [], []
    for enc_name in enc_names:
        encoder_type, architecture, model_config = enc_name.split('-')

        # Allow 512x512 for DINOv2 and DINOv3 (keep previous restriction for others)
        if resolution == 512 and encoder_type not in ('dinov2', 'dinov3'):
            raise NotImplementedError(
                "Currently, we only support 512x512 experiments with DINOv2/DINOv3 encoders."
            )

        architectures.append(architecture)
        encoder_types.append(encoder_type)

        if encoder_type == 'mocov3':
            if architecture == 'vit':
                if model_config == 's':
                    encoder = mocov3_vit.vit_small()
                elif model_config == 'b':
                    encoder = mocov3_vit.vit_base()
                elif model_config == 'l':
                    encoder = mocov3_vit.vit_large()
                ckpt = torch.load(f'./ckpts/mocov3_vit{model_config}.pth')
                state_dict = fix_mocov3_state_dict(ckpt['state_dict'])
                del encoder.head
                encoder.load_state_dict(state_dict, strict=True)
                encoder.head = torch.nn.Identity()
            elif architecture == 'resnet':
                raise NotImplementedError()
 
            encoder.eval()


        elif 'dinov2' in encoder_type:
            if local and os.path.exists(model_path):
                from transformers import Dinov2Config,Dinov2Model
                from safetensors.torch import load_file                
                config = Dinov2Config.from_pretrained(model_path)
                encoder = Dinov2Model(config)
                weights_path_safetensors = os.path.join(model_path, "model.safetensors")
                weights_path_bin = os.path.join(model_path, "pytorch_model.bin")
                
                if os.path.exists(weights_path_safetensors):
                    state_dict = load_file(weights_path_safetensors, device="cpu")
                elif os.path.exists(weights_path_bin):
                    state_dict = torch.load(weights_path_bin, map_location="cpu")
                else:
                    raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin found in {model_path}")
                encoder.load_state_dict(state_dict, strict=False) 
                print(f"[load_encoders] Successfully loaded DINOv2 from local Hugging Face directory: {model_path}")
                
            else: # remote
                if 'reg' in encoder_type:
                    try:
                        encoder = torch.hub.load('~/.cache/torch/hub/facebookresearch_dinov2_main',
                                                 f'dinov2_vit{model_config}14_reg', source='local')
                    except Exception:
                        encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_config}14_reg')
                else:
                    try:
                        encoder = torch.hub.load('~/.cache/torch/hub/facebookresearch_dinov2_main',
                                                 f'dinov2_vit{model_config}14', source='local')
                    except Exception:
                        encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_config}14')

                del encoder.head
                patch_resolution = 16 * (resolution // 256)  # 256->16, 512->32
                encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
                    encoder.pos_embed.data, [patch_resolution, patch_resolution],
                )
                encoder.head = torch.nn.Identity()
            print(f"[load_encoders] Using {enc_name} as aligning model (DINOv2)")
            encoder.eval()

        elif 'dinov3' in encoder_type:
            # NEW: DINOv3 via HuggingFace Transformers (ViT-*-16, with register tokens)
            from transformers import AutoModel
            name_map = {
                's':  'facebook/dinov3-vits16-pretrain-lvd1689m',
                'b':  'facebook/dinov3-vitb16-pretrain-lvd1689m',  # size ~ dinov2-vit-b
                'l':  'facebook/dinov3-vitl16-pretrain-lvd1689m',
                '7b': 'facebook/dinov3-vit7b16-pretrain-lvd1689m',
            }
            if model_config not in name_map:
                raise NotImplementedError(f"Unsupported dinov3 size: {model_config}")
            hf_name = name_map[model_config]
            encoder = AutoModel.from_pretrained(hf_name).eval()
            # Note: No forward_features() on HF model, will be unified in wrapper.

            print(f"[load_encoders] Using {enc_name} as aligning model (DINOv3)")

        elif 'dinov1' == encoder_type: # currently not supported
            raise NotImplementedError(f"Currently not supporting dino v1")
            from semantic_encoders import dinov1
            encoder = dinov1.vit_base()
            ckpt =  torch.load(f'./ckpts/dinov1_vit{model_config}.pth') 
            if 'pos_embed' in ckpt.keys():
                ckpt['pos_embed'] = timm.layers.pos_embed.resample_abs_pos_embed(
                    ckpt['pos_embed'], [16, 16],
                )
            del encoder.head
            encoder.head = torch.nn.Identity()
            encoder.load_state_dict(ckpt, strict=True)
            encoder.forward_features = encoder.forward
            encoder.eval()

        elif encoder_type == 'clip':
            import clip
            from semantic_encoders.clip_vit import UpdatedVisionTransformer
            encoder_ = clip.load(f"ViT-{model_config}/14", device='cpu')[0].visual
            encoder = UpdatedVisionTransformer(encoder_)
            encoder.embed_dim = encoder.model.transformer.width
            encoder.forward_features = encoder.forward
            encoder.eval()
        
        elif encoder_type == 'mae':
            from semantic_encoders.mae_vit import vit_large_patch16
            kwargs = dict(img_size=256)
            encoder = vit_large_patch16(**kwargs)
            with open(f"ckpts/mae_vit{model_config}.pth", "rb") as f:
                state_dict = torch.load(f)
            if 'pos_embed' in state_dict["model"].keys():
                state_dict["model"]['pos_embed'] = timm.layers.pos_embed.resample_abs_pos_embed(
                    state_dict["model"]['pos_embed'], [16, 16],
                )
            encoder.load_state_dict(state_dict["model"])

            encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
                encoder.pos_embed.data, [16, 16],
            )
            encoder.eval()

        elif encoder_type == 'jepa':
            from semantic_encoders.jepa import vit_huge
            kwargs = dict(img_size=[224, 224], patch_size=14)
            encoder = vit_huge(**kwargs)
            with open(f"ckpts/ijepa_vit{model_config}.pth", "rb") as f:
                state_dict = torch.load(f, map_location='cpu')
            new_state_dict = dict()
            for key, value in state_dict['encoder'].items():
                new_state_dict[key[7:]] = value
            encoder.load_state_dict(new_state_dict)
            encoder.forward_features = encoder.forward
            encoder.eval()

        else:
            raise NotImplementedError(f"Unknown encoder type: {encoder_type}")

        encoders.append(encoder)
    
    return encoders, encoder_types, architectures



class PretrainedSemanticEncoder(nn.Module):
    """
    A thin adapter that:
      - Reuses your load_encoders() to create the underlying model.
      - Applies preprocess_raw_image() internally inside forward_features.
      - Normalizes outputs to a common dict API:
            'x_norm_clstoken':      [B, D]
            'x_norm_patchtokens':   [B, N, D]
            'x_norm_registertokens':[B, R, D] (only for DINOv3 if R>0)
      - Exposes .embed_dim for compatibility with your training code.
    """
    def __init__(self, enc_type: str, resolution: int = 256, local: bool = True):
        super().__init__()
        self.enc_type_str = enc_type
        self.resolution = resolution
        self.local = local

        # Reuse your loader; we expect exactly one encoder here.
        encoders, encoder_types, architectures = load_encoders(enc_type, resolution, local = self.local)
        assert len(encoders) == 1, f"PretrainedSemanticEncoder expects a single enc_type; got {len(encoders)}"

        self.base = encoders[0]              # underlying nn.Module
        self.encoder_type = encoder_types[0] # e.g., 'dinov2' / 'dinov3' / ...
        self.architecture = architectures[0] # usually 'vit'

        if 'dinov2' in self.encoder_type:
            if hasattr(self.base, 'embed_dim'):
                # torch.hub / timm 
                self.embed_dim = int(self.base.embed_dim)
            elif self.local:
                # Hugging Face 
                self.embed_dim = int(self.base.config.hidden_size)
            else:
                raise AttributeError(f"Cannot determine embed_dim for encoder type {self.encoder_type}")

        elif 'dinov3' in self.encoder_type:
            self.embed_dim = int(self.base.config.hidden_size)
            self.num_register = int(getattr(self.base.config, "num_register_tokens", 0))
        else:
            # For other encoders (if they set embed_dim already), try to read, else raise
            if hasattr(self.base, 'embed_dim'):
                self.embed_dim = int(getattr(self.base, 'embed_dim'))
            else:
                raise AttributeError("Underlying encoder does not expose 'embed_dim'")

        for p in self.base.parameters():
            p.requires_grad_(False)
        self.base.eval()

    @torch.no_grad()
    def forward_features(self, raw_images: torch.Tensor):
        """
        Accept raw uint8/float images of shape [B, 3, H, W] with H=W in {256, 512}.
        This function:
          1) applies training-identical preprocessing (resize+normalize),
          2) forwards through the underlying model,
          3) returns a unified dict of tokens.
        """
        x = preprocess_raw_image(raw_images, self.encoder_type)

        # 2) model-specific forwarding and unification of outputs
        if 'dinov2' in self.encoder_type:
            # torch.hub dinov2 supports .forward_features(), returns dict with the same keys
            if self.local:
                out = self.base(pixel_values=x)
                h = out.last_hidden_state  # [B, 1(+R)+N, D]
                cls = h[:, 0, :]
                patches = h[:,1:,:]
                return {
                    'x_norm_clstoken': cls,
                    'x_norm_patchtokens': patches,
                }                
            else:
                feats = self.base.forward_features(x)
                assert 'x_norm_clstoken' in feats and 'x_norm_patchtokens' in feats
            return feats

        elif 'dinov3' in self.encoder_type:
            out = self.base(pixel_values=x)
            h = out.last_hidden_state  # [B, 1(+R)+N, D]
            cls = h[:, 0, :]
            if getattr(self, 'num_register', 0) > 0:
                regs = h[:, 1:1+self.num_register, :]
                patches = h[:, 1+self.num_register:, :]
                return {
                    'x_norm_clstoken': cls,
                    'x_norm_registertokens': regs,
                    'x_norm_patchtokens': patches,
                }
            else:
                patches = h[:, 1:, :]
                return {
                    'x_norm_clstoken': cls,
                    'x_norm_patchtokens': patches,
                }

        else:
            if hasattr(self.base, 'forward_features'):
                return self.base.forward_features(x)
            raise NotImplementedError(f"forward_features unification not implemented for {self.encoder_type}")
    
    def eval(self):
        super().eval()
        self.base.eval()
        return self


if __name__ == "__main__":
    from accelerate import Accelerator
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    accelerator = Accelerator(mixed_precision="bf16", device_placement=True)

    # 1) DINOv2 ViT-L/14
    enc2 = PretrainedSemanticEncoder(
        enc_type="dinov2-vit-l",
        resolution=256,
    )
    enc2 = accelerator.prepare(enc2)
    enc2.eval() 

    dummy = torch.randint(0, 256, (1, 3, 256, 256), dtype=torch.uint8, device=accelerator.device)

    with torch.no_grad(), accelerator.autocast():
        f2 = enc2.forward_features(dummy)

    print("[DINOv2] device:", accelerator.device)
    print("[DINOv2] cls:", f2["x_norm_clstoken"].shape,
          "patches:", f2["x_norm_patchtokens"].shape)

    # 2) DINOv3 ViT-L/16
    enc3 = PretrainedSemanticEncoder(
        enc_type="dinov3-vit-l",
        resolution=256,
    )
    enc3 = accelerator.prepare(enc3)
    enc3.eval()

    with torch.no_grad(), accelerator.autocast():
        f3 = enc3.forward_features(dummy)

    print("[DINOv3] device:", accelerator.device)
    print("[DINOv3] cls:", f3["x_norm_clstoken"].shape,
          "regs:", (f3["x_norm_registertokens"].shape if "x_norm_registertokens" in f3 else None),
          "patches:", f3["x_norm_patchtokens"].shape)