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
from modeling.quantizer.quantizer import VectorQuantizer, DiagonalGaussianDistribution
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
        

        if self.use_prior_model:
            self.prior_model_type = config.model.prior_model.get("prior_model_type", "gptc-S")
            prior_model_builder = GPTC_models[self.prior_model_type]
            prior_config = config.model.prior_model

            no_dropout = bool(prior_config.get("no_dropout", False))
            embd_pdrop = 0.0 if no_dropout else float(prior_config.get("embd_pdrop", 0.1))
            resid_pdrop = 0.0 if no_dropout else float(prior_config.get("resid_pdrop", 0.1))
            attn_pdrop  = 0.0 if no_dropout else float(prior_config.get("attn_pdrop", 0.1))
            if no_dropout:
                print(f"Warning: prior_loss is using no dropout")

            # kwargs building build prior model 
            prior_kwargs = dict(
                n_ind=config.model.vq_model.token_size,         # input dim of prior
                n_classes=config.model.vq_model.codebook_size,      # codebook size for prior head
                # structural/behavioral knobs (with sane defaults)
                l2_normalized=bool(prior_config.get("l2_normalized", True)),
                max_seq_len=config.model.vq_model.num_latent_tokens,
                detach_x=bool(prior_config.get("detach_x", False)),
                detach_target=bool(prior_config.get("detach_target", True)),
                fully_separated=bool(prior_config.get("fully_separated", False)),
                embd_pdrop=embd_pdrop,
                resid_pdrop=resid_pdrop,
                attn_pdrop=attn_pdrop,
            )
            self.prior_model = prior_model_builder(**prior_kwargs)

            # prior loss settings
            self.token_dim = config.model.vq_model.token_size
            self.codebook_size = config.model.vq_model.codebook_size
            self.prior_n_rounds = int(prior_config.get("n_rounds", 1))
            self.prior_no_grad_before_last_round = bool(prior_config.get("no_grad_before_last_round", False))
            self.prior_avg_loss_over_rounds = bool(prior_config.get("avg_loss_over_rounds", True))
            self.use_mix_ss = bool(prior_config.get("use_mix_ss", False))
            self.mix_ss_max_ratio = float(prior_config.get("mix_ss_max_ratio", 0.5))
            self.mix_ss_peak_steps_ratio = float(prior_config.get("mix_ss_peak_steps_ratio", 0.3))
            self.prior_latent_ce_temperature = float(prior_config.get("latent_ce_temperature", 1.0))
            self.prior_token_sampling_policy = str(prior_config.get("prior_token_sampling_policy","greedy")).lower()

        # Note: we init the prior model in the same way as the encoder and decoder
        self.apply(self._init_weights)

        if self.quantize_mode == "vq":
            self.quantize = VectorQuantizer(
                codebook_size=config.model.vq_model.codebook_size,
                token_size=config.model.vq_model.token_size,
                commitment_cost=config.model.vq_model.commitment_cost,
                use_l2_norm=config.model.vq_model.use_l2_norm,
                clustering_vq=config.model.vq_model.clustering_vq)
        elif self.quantize_mode == "vae":
            self.quantize = DiagonalGaussianDistribution
        else:
            raise NotImplementedError
        
        if self.finetune_decoder:
            # Freeze encoder/quantizer/latent tokens
            self.latent_tokens.requires_grad_(False)
            self.encoder.eval()
            self.encoder.requires_grad_(False)
            self.quantize.eval()
            self.quantize.requires_grad_(False)

            # Include MaskGiT-VQGAN's quantizer and decoder
            self.pixel_quantize = Pixel_Quantizer(
                num_embeddings=1024, embedding_dim=256, commitment_cost=0.25)
            self.pixel_decoder = Pixel_Decoder(OmegaConf.create(
                {"channel_mult": [1, 1, 2, 2, 4],
                "num_resolutions": 5,
                "dropout": 0.0,
                "hidden_channels": 128,
                "num_channels": 3,
                "num_res_blocks": 2,
                "resolution": 256,
                "z_channels": 256}))
        
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
    
    def encode(self, x):
        if self.finetune_decoder:
            with torch.no_grad():
                self.encoder.eval()
                self.quantize.eval()
                z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens)
                z_quantized, result_dict = self.quantize(z)
                result_dict["quantizer_loss"] *= 0
                result_dict["commitment_loss"] *= 0
                result_dict["codebook_loss"] *= 0
        else:
            z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens)
            if self.quantize_mode == "vq":
                z_quantized, result_dict = self.quantize(z)
            elif self.quantize_mode == "vae":
                posteriors = self.quantize(z)
                z_quantized = posteriors.sample()
                result_dict = posteriors

        return z_quantized, result_dict
    
    def decode(self, z_quantized):
        decoded = self.decoder(z_quantized)
        if self.finetune_decoder:
            quantized_states = torch.einsum(
                'nchw,cd->ndhw', decoded.softmax(1),
                self.pixel_quantize.embedding.weight)
            decoded = self.pixel_decoder(quantized_states)
        return decoded
    
    def decode_tokens(self, tokens):
        if self.quantize_mode == "vq":
            tokens = tokens.squeeze(1)
            batch, seq_len = tokens.shape # B x N
            z_quantized = self.quantize.get_codebook_entry(
                tokens.reshape(-1)).reshape(batch, 1, seq_len, -1)
            z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()
        elif self.quantize_mode == "vae":
            z_quantized = tokens
        decoded = self.decode(z_quantized)
        return decoded
    
    def forward(self, x, **kwargs):
        '''
        encode-quantize-decode & calculate prior loss
        pass the following during training:
            global_step
            max_steps
        '''
        # encoding
        z_quantized, result_dict = self.encode(x) # z_quantized : [B,D,1,N]
        # decoding
        decoded = self.decode(z_quantized)
        # prepare for prior loss
        if self.training and self.quantize_mode == 'vq' and self.use_prior_model:
            min_encoding_indices = result_dict["min_encoding_indices"] # [B,1,N] # assert VQ
            B, D, H, W = z_quantized.shape  # H==1, W==N
            assert z_quantized.dim() == 4, "z_quantized must be 4D (B, D, H=1, W=N)" # fake 2D shape
            assert z_quantized.shape[2] == 1, f"Expected H==1, but got H={z_quantized.shape[2]}"
            assert min_encoding_indices.dim() == 3 and min_encoding_indices.shape[1] == 1, f"min_encoding_indices must be (B, 1, N); got {tuple(min_encoding_indices.shape)}"            
            quantized_z = z_quantized.squeeze(2).transpose(1, 2).contiguous() # [B,N,D] ;for prior loss
            encoded_indices = min_encoding_indices.squeeze(1)  # [B,N] ;for prior loss
            result_dict["quantized_z_for_prior"] = quantized_z
            result_dict["encoded_indices_for_prior"] = encoded_indices
            # calculate prior loss
            prior_dict = self.calculate_prior_loss_with_pred(result_dict, **kwargs) # steps & global_steps in kwargs
            result_dict.update(prior_dict)
        return decoded, result_dict

    def get_emb(self):
        if self.quantize_mode == 'vq':
            emb = self.quantize.get_emb() 
            emb = emb.detach()
            return emb
        else:
            return None

    def calculate_prior_loss_with_pred(self, tokenizer_output, **kwargs):
        return_dict = {}
        B = tokenizer_output['quantized_z_for_prior'].size(0) 
        ar_input = tokenizer_output['quantized_z_for_prior'] # (b, n, d= latent_dim) normalized  VQ 后的 embedding 不是 ID 被 project out 后的 因此 dim = 16；

        labels = tokenizer_output['encoded_indices_for_prior'][:, 1:].contiguous() # (b, n - 1) # VQ 后的 码本 id next token prediction 只要后 n-1个
        logits_all_rounds, ar_pred_cont, regularized_z_ss = self.prior_ar_predict_n_rounds_ss(ar_input, **kwargs) # regularized_z_ss: (b, n, d)
        labels_all_rounds = labels.unsqueeze(0).expand(logits_all_rounds.size(0), -1, -1).contiguous() # (n_rounds or 1, b, n - 1) # 每一轮的未归一化的分数
        
        loss_latent_ce = F.cross_entropy(logits_all_rounds.view(-1, self.codebook_size), labels_all_rounds.view(-1))
        return_dict['loss_latent_ce'] = loss_latent_ce
        topk_accuracies = calculate_topk_accuracy(logits_all_rounds[0], labels, topk=(1, 5), prepend='prior_') # add top as top is predicted only with learned latent
        return_dict.update(topk_accuracies)

        return return_dict


    def logits_to_token_embedding_with_ss(self, logits, ar_input_staring_from_idx_1, mask=None, **kwargs):
        # logits: (b, n - 1, codebook_size), sequence index from 1 to n-1 (inclusive)
        # ar_input_staring_from_idx_1: (b, n - 1, d=16), requires_grad=True
        if mask is None: # 重新sample mask 如果 没有sample
            b, n_minus_1, _ = logits.size()
            if self.use_mix_ss:
                ss_ratio = (kwargs['global_step'] / (kwargs['max_steps'] * self.mix_ss_peak_steps_ratio )) * self.mix_ss_max_ratio
                ss_ratio = min(ss_ratio, self.mix_ss_max_ratio)
            else:
                ss_ratio = 1.0

            mask = torch.rand(b, n_minus_1, 1, device=self.device) < ss_ratio
            mask = mask.expand(-1, -1, self.token_dim) # (b, n - 1, d=16)

        with torch.autocast(device_type='cuda', enabled=False):
            logits = logits.float()
            if self.prior_token_sampling_policy == "greedy":
                # Hard nearest-neighbor: argmax over negative L2 logits (equiv. argmin distance)
                indices = logits.argmax(dim=-1)  # (B, N-1), dtype=long                
            elif self.prior_token_sampling_policy == "boltzmann":
                # Boltzmann sampling over negative distances (soft exploration)                
                probs = F.softmax(logits, dim=-1) # (b, n - 1, codebook_size)
                indices = torch.multinomial(probs.view(-1, self.codebook_size), 1).view(*probs.size()[:-1]) # (b, n - 1) 多项分布采样 不 argmax
        token_embedding = F.embedding(indices, self.get_emb()) # (b, n - 1, d=16) 从码本取出 id 对应的 embedding
        token_embedding = torch.where(mask, token_embedding, ar_input_staring_from_idx_1) # ss 混合 把 mask的部分替换为 自己要的

        return token_embedding

    def calculate_logits_and_ar_pred_cont(self, prior_model_output):
        ar_pred_cont = prior_model_output # (b, n, d=16)
        y = prior_model_output
        E = self.get_emb()
        y2 = (y ** 2).sum(-1, keepdim=True)         # (B, N, 1)
        E2 = (E ** 2).sum(-1)                        # (codebook_size,)
        logits = -(y2 + E2[None, None, :] - 2.0 * (y @ E.t()))  # (B, N, codebook_size)   negL2 logits
        logits = logits[:, 1:].mul_(1.0 / self.prior_latent_ce_temperature).contiguous() # (b, n - 1, codebook_size)  排除掉 第0个
        return logits, ar_pred_cont

    def prior_ar_predict_n_rounds_ss(self, ar_input, **kwargs):
        prior_model = self.prior_model
        n_rounds = self.prior_n_rounds
        no_grad_before_last_round = self.prior_no_grad_before_last_round

        b, n, _ = ar_input.size()
        n_minus_1 = n - 1
        if self.use_mix_ss:
            global_step = kwargs['global_step'] # 记录step 全局 step 和 总的 step 由trainer 传进来
            max_steps = kwargs['max_steps']
            peak_steps_ratio = torch.tensor(self.mix_ss_peak_steps_ratio, dtype=torch.float32) # ss 调度时候 用多长时间  涨到 max_ratio 相对于 整个训练的max_steps 而言
            max_ratio = torch.tensor(self.mix_ss_max_ratio, dtype=torch.float32) # 最多喂多少自己的token 0。5

            ss_ratio = (global_step / (max_steps * peak_steps_ratio)) * max_ratio
            ss_ratio = torch.min(ss_ratio, max_ratio)
        else:
            ss_ratio = torch.tensor(1.0, dtype=torch.float32)

        mask_ss = torch.rand(b, n_minus_1, 1, device=self.device) < ss_ratio # sample 出来 被 ss 自喂的token位置
        mask_ss = mask_ss.expand(-1, -1, self.token_dim) # (b, n - 1, d=16) # 把 mask 应用到 embedding的每个位置

        logits_all_rounds = []
        next_ar_input = ar_input # (b, n, d=16)
        for i in range(n_rounds): # 一轮正常的 一轮 ss
            if no_grad_before_last_round and i < n_rounds - 1:
                # we can not use "with torch.no_grad()" here due to a pytorch's bug!
                # https://github.com/pytorch/pytorch/issues/112583
                prior_model.requires_grad_(False)
                with torch.amp.autocast("cuda", enabled=False):
                    prior_model_output = prior_model.ar_predict(next_ar_input.detach().float())
                    logits, ar_pred_cont = self.calculate_logits_and_ar_pred_cont(prior_model_output.float())
                prior_model.requires_grad_(True)
            else:
                with torch.amp.autocast("cuda", enabled=False):
                    prior_model_output = prior_model.ar_predict(next_ar_input.float()) # (b, n - 1, codebook_size) or (b, n, d=16) # 第0个
                    logits, ar_pred_cont = self.calculate_logits_and_ar_pred_cont(prior_model_output.float()) # logits；对码本的未归一化打分，表示改选哪个code softmax后变成概率
                    logits_all_rounds.append(logits)


            if i < n_rounds - 1:
                token_embedding = self.logits_to_token_embedding_with_ss(logits, ar_input[:, 1:], mask=mask_ss, **kwargs) # (b, n - 1, d=16) ar_input 是 gt的 不是自喂的
                next_ar_input = torch.cat([ar_input[:, :1], token_embedding], dim=1) # (b, n, d=16) 第一轮结束 把当前的ss后的结果 和 gt 的第0个 混在一起 送进去下一轮forward

        if self.prior_avg_loss_over_rounds:
            logits_all_rounds = torch.stack(logits_all_rounds, dim=0) # (n_rounds, b, n - 1, codebook_size)

        else:
            logits_all_rounds = torch.stack([logits_all_rounds[-1]], dim=0) # (1, b, n - 1, codebook_size)

        return logits_all_rounds, ar_pred_cont, next_ar_input # here the next_ar_input is actually the last round's ar_input

    def prior_parameters(self, require_grad_only: bool = True):
        """return prior model parameters"""
        pm = getattr(self, "prior_model", None)
        if pm is None:
            return []
        params = list(pm.parameters())
        if require_grad_only:
            params = [p for p in params if p.requires_grad]
        return params

    def non_prior_parameters(self, require_grad_only: bool = True):
        """return parameters other than prior_model"""
        params = []
        for name, p in self.named_parameters():
            if not name.startswith("prior_model."):
                if (not require_grad_only) or p.requires_grad:
                    params.append(p)
        return params
    
@torch.no_grad()
@torch.autocast(device_type='cuda', enabled=False)
def calculate_topk_accuracy(logits, targets, topk=(1, 5), prepend=''):
    """
    Computes the top-k accuracy for the specified values of k.

    Args:
    logits (torch.Tensor): The predicted logits (unnormalized scores) with shape (batch_size, num_tokens, num_classes).
    targets (torch.Tensor): The true labels with shape (batch_size, num_tokens).
    topk (tuple): A tuple of integers specifying the top-k accuracies to compute.

    Returns:
    dict: A dictionary with keys 'top1' and 'top5' containing the respective accuracies.
    """
    logits = logits.float()
    logits = logits.view(-1, logits.size(-1))
    targets = targets.view(-1)

    maxk = max(topk)
    batch_size = targets.size(0)

    # Get the top maxk predictions and their indices
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()  # Transpose to shape (maxk, batch_size)
    correct = pred.eq(targets.view(1, -1).expand_as(pred))  # Shape (maxk, batch_size)

    topk_accuracies = {}
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        topk_accuracies[f'{prepend}top{k}_acc'] = correct_k.mul_(100.0 / batch_size)

    return topk_accuracies

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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