"""Vector quantizer.

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

Reference: 
    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py
    https://github.com/google-research/magvit/blob/main/videogvt/models/vqvae.py
    https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/distributions/distributions.py
    https://github.com/lyndonzheng/CVQ-VAE/blob/main/quantise.py
"""
from typing import Mapping, Text, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from accelerate.utils.operations import gather
from accelerate.state import AcceleratorState
from torch.cuda.amp import autocast


def _safe_gather(x: torch.Tensor) -> torch.Tensor:
    try:
        state = AcceleratorState()
        if state.num_processes == 1:
            return x
    except Exception:
        return x
    return gather(x)

def _l2norm(t: torch.Tensor) -> torch.Tensor:
    return F.normalize(t, p=2, dim=-1)

def _entropy_loss(affinity: torch.Tensor, temperature: float = 0.01, mode: str = "softmax"):
    """
    Compute E[H(p)] - H(E[p]).
    - affinity: [..., K]
    - temperature: p = softmax(affinity / T)
    - mode: 'softmax' or 'argmax' (straight-through onehot)
    """
    flat = affinity.view(-1, affinity.shape[-1]) / temperature
    probs = F.softmax(flat, dim=-1)
    log_probs = F.log_softmax(flat, dim=-1)

    if mode == "softmax":
        target_probs = probs
    elif mode == "argmax":
        codes = torch.argmax(flat, dim=-1)
        onehots = F.one_hot(codes, num_classes=flat.shape[-1]).to(probs.dtype)
        target_probs = probs - (probs - onehots).detach()
    else:
        raise ValueError(f"Unsupported mode={mode}")

    avg_probs = target_probs.mean(dim=0)
    avg_entropy = -(avg_probs * torch.log(avg_probs + 1e-5)).sum()
    sample_entropy = -(target_probs * log_probs).sum(dim=-1).mean()
    loss = sample_entropy - avg_entropy
    return loss, sample_entropy, avg_entropy


class simVQ(nn.Module):
    def __init__(
        self,
        codebook_size: int = 1024,
        token_size: int = 256,
        commitment_cost: float = 0.25,
        use_l2_norm: bool = False,
        clustering_vq: bool = False,
        simvq: bool = True,  # 新增：SimVQ 默认开启
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.token_size = token_size
        self.commitment_cost = commitment_cost

        self.embedding = torch.nn.Embedding(codebook_size, token_size)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=self.token_size ** -0.5)

        # 保留原参数以兼容，但在 simvq=True 时不使用
        self.use_l2_norm = use_l2_norm
        self.clustering_vq = clustering_vq

        # ----- SimVQ: freeze C, learn W -----
        self.simvq = simvq
        if self.simvq:
            # freeze C
            for p in self.embedding.parameters():
                p.requires_grad = False
            # linear
            # self.embedding_proj = nn.Linear(token_size, token_size)
            self.embedding_proj = nn.Linear(token_size, token_size, bias=False)
            with torch.no_grad():
                self.embedding_proj.weight.copy_(torch.eye(token_size))

            if self.clustering_vq:
                raise ValueError("SimVQ don't support clustering_vq=True")
        else:
            if self.clustering_vq:
                self.decay = 0.99
                self.register_buffer("embed_prob", torch.zeros(codebook_size))
                self.register_buffer("ema_embedding", self.embedding.weight.detach().clone())

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """
        输入 z: 形状 [B, C, H, W]（也兼容 H=1, W=L 的一维场景）
        返回:
          z_quantized: [B, C, H, W]
          result_dict: dict(quantizer_loss, commitment_loss, codebook_loss, min_encoding_indices[B,H,W])
        """
        z = z.float()
        # 重排到 [B, H, W, C] 再展平
        z_hw_c = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = rearrange(z_hw_c, 'b h w c -> (b h w) c')  # [N, D]

        # 选择量化码本的基底
        if self.simvq:
            # SimVQ: 使用投影后的码本 C @ W
            codebook_projected = self.embedding_proj(self.embedding.weight)  # [K, D]
        else:
            # 非 SimVQ 路径（保留兼容）
            emb_source = self.ema_embedding if self.clustering_vq else self.embedding.weight
            if self.use_l2_norm:
                z_flattened = F.normalize(z_flattened, dim=-1)
                emb_source = F.normalize(emb_source, dim=-1)
            codebook_projected = emb_source  # [K, D]


        if self.use_l2_norm:
            z_for_dist  = F.normalize(z_flattened,dim=-1,eps=1e-6)
            codebook_for_dist = F.normalize(codebook_projected, dim=-1,eps=1e-6)
        else:
            z_for_dist  = z_flattened
            codebook_for_dist = codebook_projected

        # 距离矩阵 d: [N, K]，使用 ||z||^2 + ||q||^2 - 2 z·q
        d = torch.sum(z_for_dist**2, dim=1, keepdim=True) + \
            torch.sum(codebook_for_dist**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_for_dist, codebook_for_dist.T)
        

        # 最近邻索引
        min_encoding_indices = torch.argmin(d, dim=1)  # [N]

        # 查表得到量化向量（使用与计算距离一致的码本）
        z_quantized_hw_c = self.get_codebook_entry(min_encoding_indices).view(z_hw_c.shape)
        if self.use_l2_norm:
            z_hw_c = torch.nn.functional.normalize(z_hw_c, dim=-1)
        
        # 计算损失（保持你原公式与标量命名）
        commitment_loss = self.commitment_cost * torch.mean((z_quantized_hw_c.detach() - z_hw_c) ** 2)
        codebook_loss = torch.mean((z_quantized_hw_c - z_hw_c.detach()) ** 2)
        loss = commitment_loss + codebook_loss

        # 直通估计（STE）：前向替换为量化值，反向对编码器等价恒等
        z_quantized_hw_c = z_hw_c + (z_quantized_hw_c - z_hw_c).detach()

        # 还原到 [B, C, H, W]
        z_quantized = rearrange(z_quantized_hw_c, 'b h w c -> b c h w').contiguous()

        # 索引形状回到 [B, H, W]
        B, C, H, W = z.shape
        min_encoding_indices = min_encoding_indices.view(B, H, W)

        result_dict = dict(
            quantizer_loss=loss,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            min_encoding_indices=min_encoding_indices,
            entropy_loss=torch.zeros((), device=z.device, dtype=z.dtype).detach(),
            n_reactivate=torch.zeros((), device=z.device, dtype=z.dtype).detach(),
            threshold_count=torch.zeros((), device=z.device, dtype=z.dtype).detach(),
        )

        return z_quantized, result_dict

    def get_codebook_entry(self, indices):
        weight = self.embedding_proj(self.embedding.weight)  # [K, D]
        if self.use_l2_norm:
           weight = F.normalize(weight, dim=-1, eps=1e-6)
        
        if len(indices.shape) == 1:
            # 离散索引查表
            z_quantized = F.embedding(indices, weight)
        elif len(indices.shape) == 2:
            # soft one-hot * codebook
            z_quantized = torch.einsum('bd,dn->bn', indices, weight.T)
        else:
            raise NotImplementedError("indices must be 1D or 2D")

        return z_quantized

    @torch.autocast(device_type='cuda', enabled=False)
    def get_emb(self):
        emb = self.embedding.weight
        assert emb.dtype == torch.float32, f"Embedding weight dtype is {emb.dtype}, expected float32"
        weight = self.embedding_proj(self.embedding.weight)  # [K, D]
        if self.use_l2_norm:
            weight = F.normalize(weight, dim=-1, eps=1e-6)
        return weight
class VectorQuantizer(nn.Module):
    def __init__(
        self,
        codebook_size: int = 1024,
        token_size: int = 256,
        commitment_cost: float = 0.25,
        use_l2_norm: bool = True,
        clustering_vq: bool = False,

        use_reinit: bool = True,
        reinit_decay: float = 0.99,
        reinit_threshold_base: float = 0.0125, 
        reset_boost: float = 1.1,
        reactivate_after: int = 10000,
        reactivate_every: int = 1,
        max_react_frac: float = 0.001,       # e.g. 0.1% of codebook per step
        max_react_cap: int | None = None,    # absolute cap; if not None, min(fraction_cap, cap)

        # Optional entropy loss (kept off by default)
        use_entropy_loss: bool = False,
        entropy_temperature: float = 0.01,
        entropy_mode: str = "softmax",
        entropy_weight: float = 0.0,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.token_size = token_size
        self.commitment_cost = commitment_cost
        self.use_l2_norm = use_l2_norm

        self.embedding = nn.Embedding(codebook_size, token_size)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

        self.clustering_vq = clustering_vq
        if clustering_vq:
            self.decay = 0.99
            self.register_buffer("embed_prob", torch.zeros(codebook_size))
            self.register_buffer("ema_embedding", self.embedding.weight.detach().clone())

        self.use_reinit = use_reinit
        self.reinit_decay = float(reinit_decay)
        self.reinit_threshold_base = float(reinit_threshold_base)
        self.reset_boost = float(reset_boost)
        self.reactivate_after = int(reactivate_after)
        self.reactivate_every = int(reactivate_every)

        self.max_react_frac = float(max_react_frac)
        self.max_react_cap = max_react_cap if max_react_cap is None else int(max_react_cap)

        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("threshold_count", torch.tensor(0.0))
        self._threshold_inited = False

        self.register_buffer("steps", torch.zeros((), dtype=torch.long))

        # entropy loss
        self.use_entropy_loss = use_entropy_loss
        self.entropy_temperature = float(entropy_temperature)
        self.entropy_mode = entropy_mode
        self.entropy_weight = float(entropy_weight)

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        z = z.float()
        z_bhwc = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flat = rearrange(z_bhwc, 'b h w c -> (b h w) c')
        unnormed_z_flattened = z_flat
        z_flat_for_dist = _l2norm(z_flat) if self.use_l2_norm else z_flat

        emb_source = self.ema_embedding if self.clustering_vq else self.embedding.weight
        emb_for_dist = _l2norm(emb_source) if self.use_l2_norm else emb_source

        d = (
            torch.sum(z_flat_for_dist**2, dim=1, keepdim=True)
            + torch.sum(emb_for_dist**2, dim=1)
            - 2 * torch.einsum('nd,kd->nk', z_flat_for_dist, emb_for_dist)
        )

        min_encoding_indices = torch.argmin(d, dim=1)  # [N]
        z_q = self.get_codebook_entry(min_encoding_indices).view_as(z_bhwc)

        z_for_commit = _l2norm(z_bhwc) if self.use_l2_norm else z_bhwc
        commitment_loss = self.commitment_cost * torch.mean((z_q.detach() - z_for_commit) ** 2)
        codebook_loss = torch.mean((z_q - z_for_commit.detach()) ** 2)

        if self.clustering_vq and self.training:
            with torch.no_grad():
                # usage update
                encoding_indices = _safe_gather(min_encoding_indices)
                if len(min_encoding_indices.shape) != 1:
                    raise ValueError(f"min_encoding_indices in a wrong shape, {min_encoding_indices.shape}")
                # Compute and update the usage of each entry in the codebook.
                encodings = torch.zeros(encoding_indices.shape[0], self.codebook_size, device=z.device)
                encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
                avg_probs = torch.mean(encodings, dim=0)
                self.embed_prob.mul_(self.decay).add_(avg_probs, alpha=1-self.decay)
                # codebook update
                all_d = _safe_gather(d)
                all_unnormed_z_flattened = _safe_gather(unnormed_z_flattened).detach()
                if all_d.shape[0] != all_unnormed_z_flattened.shape[0]:
                    raise ValueError(
                        "all_d and all_unnormed_z_flattened have different length" + 
                        f"{all_d.shape}, {all_unnormed_z_flattened.shape}")
                indices = torch.argmin(all_d, dim=0)
                random_feat = all_unnormed_z_flattened[indices]

                decay = torch.exp(
                    -(self.embed_prob * self.codebook_size * 10)
                    / (1 - self.decay) - 1e-3).view(-1, 1).expand(-1, self.token_size)

                # update EMA embedding safely
                self.ema_embedding.copy_(
                    self.ema_embedding * (1 - decay) + random_feat * decay
                )
        total_loss = commitment_loss + codebook_loss

        # Dead-code reactivation 
        n_reactivate = 0
        if self.training and self.use_reinit:
            with torch.no_grad():
                self.steps += 1
                g_idx = _safe_gather(min_encoding_indices)          # [N_global]
                g_feats = _safe_gather(z_flat)                       # [N_global, D] (raw pool)

                # init threshold in count-space on first fwd
                if not self._threshold_inited:
                    N_global = g_idx.shape[0]
                    ratio = N_global / float(self.codebook_size)     # expected hits per code per step
                    thr = self.reinit_threshold_base * ratio
                    self.threshold_count.data.copy_(torch.tensor(thr, device=self.threshold_count.device))
                    self._threshold_inited = True

                # EMA update on cluster_size
                bins = torch.bincount(g_idx, minlength=self.codebook_size).to(self.cluster_size.dtype)
                self.cluster_size.mul_(self.reinit_decay).add_(bins, alpha=1 - self.reinit_decay)

                # gated by warmup and frequency
                do_react = (self.steps.item() >= self.reactivate_after) and \
                           ((self.steps.item() - self.reactivate_after) % max(1, self.reactivate_every) == 0)

                if do_react:
                    dead_mask = self.cluster_size < self.threshold_count
                    if dead_mask.any() and g_feats.numel() > 0:

                        cap_frac = max(1, int(self.codebook_size * self.max_react_frac))
                        cap = cap_frac if self.max_react_cap is None else min(cap_frac, int(self.max_react_cap))

                        # pick the least-used dead codes (top-k smallest cluster_size among dead)
                        num_dead_total = int(dead_mask.sum().item())
                        n_pick = min(num_dead_total, cap)

                        if n_pick > 0:
                            cs = self.cluster_size.clone()
                            cs[~dead_mask] = float('inf')       # mask out non-dead
                            # smallest n_pick
                            _, dead_idx = torch.topk(cs, k=n_pick, largest=False, sorted=False)

                            # sample replacement features for exactly n_pick codes
                            M = g_feats.shape[0]
                            if M >= n_pick:
                                sel = torch.randperm(M, device=g_feats.device)[:n_pick]
                            else:
                                sel = torch.randint(0, M, (n_pick,), device=g_feats.device)
                            new_codes = g_feats[sel]
                            if self.use_l2_norm:
                                new_codes = _l2norm(new_codes)

                            # write into embedding used by quantization
                            if self.clustering_vq:
                                self.ema_embedding.data[dead_idx] = new_codes
                                self.embedding.weight.data[dead_idx] = new_codes
                            else:
                                self.embedding.weight.data[dead_idx] = new_codes

                            # lift counts so they don't immediately die again
                            safe_val = float(self.reset_boost) * float(self.threshold_count.item())
                            self.cluster_size.data[dead_idx] = safe_val
                            n_reactivate = int(n_pick)

        # straight-through
        z_out = z_for_commit + (z_q - z_for_commit).detach()
        z_out = rearrange(z_out, 'b h w c -> b c h w').contiguous()

        # optional entropy loss
        ent_loss = sample_entropy = avg_entropy = 0.0
        if self.use_entropy_loss:
            ent_loss, sample_entropy, avg_entropy = _entropy_loss(-d, temperature=self.entropy_temperature,
                                                                  mode=self.entropy_mode)
            if self.entropy_weight != 0.0:
                total_loss = total_loss + self.entropy_weight * ent_loss

        result = dict(
            quantizer_loss=total_loss,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            min_encoding_indices=min_encoding_indices.view(z_out.shape[0], z_out.shape[2], z_out.shape[3]),
            entropy_loss=ent_loss,
            n_reactivate=n_reactivate,
            threshold_count=float(self.threshold_count.item()),
        )
        return z_out, result

    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.ndim == 1:
            zq = self.embedding(indices)
        elif indices.ndim == 2:
            zq = torch.einsum('nk,kd->nd', indices, self.embedding.weight)
        else:
            raise NotImplementedError
        return _l2norm(zq) if self.use_l2_norm else zq

    @torch.autocast(device_type='cuda', enabled=False)
    def get_emb(self):
        emb = self.embedding.weight
        return _l2norm(emb) if self.use_l2_norm else emb



class DiagonalGaussianDistribution(object):
    @torch.autocast(device_type="cuda",enabled=False)
    def __init__(self, parameters, deterministic=False):
        """Initializes a Gaussian distribution instance given the parameters.

        Args:
            parameters (torch.Tensor): The parameters for the Gaussian distribution. It is expected
                to be in shape [B, 2 * C, *], where B is batch size, and C is the embedding dimension.
                First C channels are used for mean and last C are used for logvar in the Gaussian distribution.
            deterministic (bool): Whether to use deterministic sampling. When it is true, the sampling results
                is purely based on mean (i.e., std = 0).
        """
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters.float(), 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    @torch.autocast(device_type="cuda",enabled=False)
    def sample(self):
        x = self.mean.float() + self.std.float() * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    @torch.autocast(device_type="cuda",enabled=False)
    def mode(self):
        return self.mean

    @torch.autocast(device_type="cuda",enabled=False)
    def kl(self):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            return 0.5 * torch.sum(torch.pow(self.mean.float(), 2)
                                    + self.var.float() - 1.0 - self.logvar.float(),
                                    dim=[1, 2])
