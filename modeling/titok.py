"""This file contains the model definition of TiTok.

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
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from modeling.modules.base_model import BaseModel
from modeling.modules.blocks import TiTokEncoder, TiTokDecoder
from modeling.quantizer.quantizer import VectorQuantizer, DiagonalGaussianDistribution,simVQ
from modeling.modules.maskgit_vqgan import Encoder as Pixel_Eecoder
from modeling.modules.maskgit_vqgan import Decoder as Pixel_Decoder
from modeling.modules.maskgit_vqgan import VectorQuantizer as Pixel_Quantizer
from modeling.modules.gptc import GPTC_models
import json
from omegaconf import OmegaConf
from pathlib import Path

from huggingface_hub import PyTorchModelHubMixin


class PretrainedTokenizer(nn.Module):
    def __init__(self, pretrained_weight):
        super().__init__()
        conf = OmegaConf.create(
            {"channel_mult": [1, 1, 2, 2, 4],
            "num_resolutions": 5,
            "dropout": 0.0,
            "hidden_channels": 128,
            "num_channels": 3,
            "num_res_blocks": 2,
            "resolution": 256,
            "z_channels": 256})
        self.encoder = Pixel_Eecoder(conf)
        self.decoder = Pixel_Decoder(conf)
        self.quantize = Pixel_Quantizer(
            num_embeddings=1024, embedding_dim=256, commitment_cost=0.25)
        # Load pretrained weights
        self.load_state_dict(torch.load(pretrained_weight, map_location=torch.device("cpu")), strict=True)
        
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def encode(self, x):
        hidden_states = self.encoder(x)
        quantized_states, codebook_indices, codebook_loss = self.quantize(hidden_states)
        return codebook_indices.detach()
    
    @torch.no_grad()
    def decode(self, codes):
        quantized_states = self.quantize.get_codebook_entry(codes)
        rec_images = self.decoder(quantized_states)
        rec_images = torch.clamp(rec_images, 0.0, 1.0)
        return rec_images.detach()
    
    @torch.no_grad()
    def decode_tokens(self, codes):
        return self.decode(codes)


class TiTok(BaseModel, PyTorchModelHubMixin):
    def __init__(self, config):

        if isinstance(config, dict):
            config = OmegaConf.create(config)

        super().__init__()
        self.config = config
        # This should be False for stage1 and True for stage2.
        self.finetune_decoder = config.model.vq_model.get("finetune_decoder", True)
        self.use_semantic_guidance = config.model.vq_model.get("use_semantic_guidance", False)
        self.use_prior_model = config.model.vq_model.get("use_prior_model", False)
        self.quantize_mode = config.model.vq_model.get("quantize_mode", "vq")
        if self.quantize_mode not in ["vq", "vae"]:
            raise ValueError(f"Unsupported quantize mode {self.quantize_mode}.")
        
        if self.finetune_decoder and self.quantize_mode not in ["vq"]:
            raise ValueError("Only supprot finetune_decoder with vq quantization for now.")

        self.encoder = TiTokEncoder(config)
        self.decoder = TiTokDecoder(config)
        
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        scale = self.encoder.width ** -0.5
        self.latent_tokens = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.encoder.width))
        
        self.num_semantic_latent_tokens = config.model.vq_model.get("num_semantic_latent_tokens", 0)
        if self.num_semantic_latent_tokens > 0:
            self.semantic_latent = nn.Parameter(
                scale * torch.randn(self.num_semantic_latent_tokens, self.encoder.width))
        else:
            self.semantic_latent = None

        # Note: we init the prior model in the same way as the encoder and decoder
        self.apply(self._init_weights)

        if self.quantize_mode == "vq":
            self.quantize = simVQ(
                codebook_size=config.model.vq_model.codebook_size,
                token_size=config.model.vq_model.token_size,
                commitment_cost=config.model.vq_model.commitment_cost,
                use_l2_norm=config.model.vq_model.use_l2_norm,
                clustering_vq=config.model.vq_model.clustering_vq)
        elif self.quantize_mode == "vae":
            self.quantize = DiagonalGaussianDistribution
        else:
            raise NotImplementedError
        
        
    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights and config to a local directory."""
        # Assume 'self.config' is your DictConfig object
        # Convert to a regular dictionary
        dict_config = OmegaConf.to_container(self.config)
        # Save as JSON
        file_path = Path(save_directory) / "config.json"
        with open(file_path, 'w') as json_file:
            json.dump(dict_config, json_file, indent=4)
        super()._save_pretrained(save_directory)

    def _init_weights(self, module):
        """ Initialize the weights.
            :param:
                module -> torch.nn.Module: module to initialize
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
    
    def encode(self, x, semantic_token_dict):
        z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens , semantic_latent = self.semantic_latent, semantic_token_dict = semantic_token_dict)
        if self.quantize_mode == "vq":
            z_quantized, result_dict = self.quantize(z)
        elif self.quantize_mode == "vae":
            posteriors = self.quantize(z)
            z_quantized = posteriors.sample()
            result_dict = posteriors

        return z_quantized, result_dict
    
    def decode(self, z_quantized):
        decoded,cls_reconsturcted = self.decoder(z_quantized)
        return decoded,cls_reconsturcted
    
    def decode_tokens(self, tokens):
        if self.quantize_mode == "vq":
            tokens = tokens.squeeze(1)
            batch, seq_len = tokens.shape # B x N
            z_quantized = self.quantize.get_codebook_entry(
                tokens.reshape(-1)).reshape(batch, 1, seq_len, -1)
            z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()
        elif self.quantize_mode == "vae":
            z_quantized = tokens
        decoded,cls_reconsturcted = self.decode(z_quantized)
        return decoded,cls_reconsturcted
    
    def forward(self, x, semantic_token_dict = None, **kwargs):
        '''
        encode-quantize-decode & calculate prior loss
        pass the following during training if using prior loss:
            global_step
            max_steps
        '''
        # encoding
        z_quantized, result_dict = self.encode(x,semantic_token_dict) # z_quantized : [B,D,1,N]
        # decoding
        decoded,cls_reconsturcted = self.decode(z_quantized)
        result_dict["cls_recon"] = cls_reconsturcted
        return decoded, result_dict


if __name__ == "__main__":
    # Minimal self-test for TiTok 
    import sys
    from pathlib import Path
    import torch
    from omegaconf import OmegaConf

    def pretty_kv(name, val):
        """Pretty-print a key/value with shapes for tensors."""
        if torch.is_tensor(val):
            if val.ndim == 0:
                print(f"{name}: {val.item()} (scalar Tensor, dtype={val.dtype}, device={val.device})")
            else:
                print(f"{name}: Tensor(shape={tuple(val.shape)}, dtype={val.dtype}, device={val.device})")
        elif isinstance(val, dict):
            print(f"{name}: dict(len={len(val)})")
            for k, v in val.items():
                pretty_kv(f"  {name}.{k}", v)
        else:
            print(f"{name}: {val} ({type(val).__name__})")

    # 1) Load config from YAML
    cfg_path_str = r"configs/training/adaptive1DTokenzier/titok_bl32_vq.yaml"
    if not (cfg_path_str.lower().endswith(".yml") or cfg_path_str.lower().endswith(".yaml")):
        print(f"[Error] Not a YAML file: {cfg_path_str}")
        sys.exit(1)

    cfg_path = Path(cfg_path_str)
    if not cfg_path.exists():
        print(f"[Skip] Config not found: {cfg_path}. Please run from repo root or fix the path.")
        sys.exit(0)

    try:
        cfg = OmegaConf.load(str(cfg_path))
    except Exception as e:
        print(f"[Error] Failed to load YAML via OmegaConf: {e}")
        sys.exit(1)

    # 2) Build model
    try:
        model = TiTok(cfg)
    except Exception as e:
        print(f"[Error] Failed to construct TiTok: {e}")
        sys.exit(1)

    model.eval()
    device = torch.device("cpu")
    model.to(device)

    # 3) Random input [2, 3, 256, 256]
    torch.manual_seed(0)
    x = torch.randn(2, 3, 256, 256, device=device)

    # 4) Forward
    with torch.no_grad():
        # Pass default steps to satisfy possible SS scheduling
        decoded, result_dict = model(x, global_step=0, max_steps=1)

    # 5) Assert output shape equals input shape
    if not torch.is_tensor(decoded):
        print("[Error] Model output is not a Tensor.")
        sys.exit(1)

    print(f"Input shape : {tuple(x.shape)}")
    print(f"Output shape: {tuple(decoded.shape)}")
    assert decoded.shape == x.shape, f"Output shape {tuple(decoded.shape)} != input shape {tuple(x.shape)}"
    print("[OK] Output shape matches input shape.")

    # 6) Print all result_dict entries
    if not isinstance(result_dict, dict):
        print("[Warn] result_dict is not a dict; got:", type(result_dict).__name__)
    else:
        print("\n=== result_dict entries ===")
        for k, v in result_dict.items():
            pretty_kv(k, v)

    print("\n[Done] TiTok __main__ self-test completed.")