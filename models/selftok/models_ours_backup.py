import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from mimogpt.models.selftok.models import DiT, DiTBlock, get_2d_sincos_pos_embed, modulate, TimestepEmbedder, FinalLayer
import torch.nn.functional as F
#from mimogpt.tokenizer.selftok import VectorQuantize
from mimogpt.tokenizer.selftok import LFQ
# import xformers.ops
from mimogpt.models.selftok.quantizer import VectorQuantizer_L2norm as VectorQuantize


class ViTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            c_dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(c_dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, c: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        B, N, C = x.shape
        kv = self.kv(c).reshape(B, c.shape[1], 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q = self.to_q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q, k = self.q_norm(q), self.k_norm(k)
        
        if mask is not None:
            attn_mask = torch.zeros([B, self.num_heads, q.shape[2], k.shape[2]], dtype=q.dtype, device=q.device)
            attn_mask.masked_fill_(mask.unsqueeze(1).unsqueeze(1).repeat(1, self.num_heads, q.shape[2], 1) == 0, float('-inf'))
            # inftensor = torch.tensor(float('-inf')).expand_as(attn_mask[0]).to(attn_mask.device)
            # for idx in range(B):
            #     if torch.equal(attn_mask[idx], inftensor):
            #         print(idx,'eq')
            #         attn_mask[idx, ...] = 0
        else:
            attn_mask = None
        
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if attn_mask is not None:
                attn += attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        # x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_mask)
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DiTCrossAttnBlock(DiTBlock):
    def __init__(self, hidden_size, encoder_hidden_size, num_heads, mlp_ratio=4.0, cross_modulate=True, **block_kwargs):
        super().__init__(hidden_size, num_heads, mlp_ratio, **block_kwargs)
        self.cross_attn = CrossAttention(hidden_size, encoder_hidden_size, num_heads, qkv_bias=True, **block_kwargs)
        if cross_modulate:
            self.cross_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 3 * hidden_size, bias=True)
            )
        else:
            self.cross_modulation = None
        self.cross_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
    def forward(self, x, c, encoder_hidden_states, mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        if self.cross_modulation is not None:
            gate_mca, shift_mca, scale_mca = self.cross_modulation(c).chunk(3, dim=1)
            x = x + gate_mca.unsqueeze(1) * self.cross_attn(modulate(self.cross_norm(x), shift_mca, scale_mca), encoder_hidden_states, mask)
        else:
            x = x + self.cross_attn(self.cross_norm(x), encoder_hidden_states)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class Encoder(nn.Module):
    def __init__(
        self, K, input_size=32, encoder_hidden_size=256, patch_size=8, in_channels=4,
        hidden_size=256, depth=None, num_heads=4, mlp_ratio=4.0, w_diversity=0.0, w_commit=1.0,
        n_e = 8192, n_e_cfg=None, share_codebook=False, reverse_code=False, code_dim=32,
        quantize_kmeans_init=True, decay=0.99, dead_code_threshold=0.0, quantizer_temp=10.0,
        use_cosine_sim=True, pre_norm=False, post_norm=True, lfq=False,
        k_embed=False, code_shrink_rate=1, ema_update=True, if_force_sync=True
    ):
        super().__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        depth = depth or self.K
        self.depth = depth
        self.hidden_size = hidden_size
        self.n_e = n_e
        self.pre_norm = pre_norm
        self.post_norm = post_norm
        self.w_diversity = w_diversity
        self.w_commit = w_commit
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.n_tokens = K * (input_size // patch_size) ** 2
        
        self.blocks = nn.ModuleList([
            ViTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer_norm = nn.LayerNorm(hidden_size // code_shrink_rate, eps=1e-6)  
        self.final_layer_norm2 = nn.LayerNorm(code_dim, eps=1e-6)


        self.project_out = nn.Linear(code_dim, encoder_hidden_size)


        self.final_layer_norm3 = nn.LayerNorm(encoder_hidden_size, eps=1e-6)  
        
        self.share_codebook = share_codebook
        self.reverse_code = reverse_code
        self.code_dim = code_dim
        self.lfq = lfq
        self.quantizer_temp = quantizer_temp
        

        self.encoder_hidden_size = encoder_hidden_size
        self.n_e = n_e
        self.beta = 0.99


        self.keep_mask = None

        # 2. k embed
        self.n_tokens_cfg = [input_size // patch_size] * self.K
        if k_embed:
            self.k_embed = nn.Embedding(self.K, encoder_hidden_size)
            nn.init.constant_(self.k_embed.weight.data, 0)
            d = torch.cat([torch.full((n*n,), i) for i, n in enumerate(self.n_tokens_cfg)]).view(1, self.n_tokens, 1)
            dT = d.transpose(1, 2)
            k_embed_indices = dT[:, 0].contiguous()
            self.register_buffer('k_embed_indices', k_embed_indices)
        else:
            self.k_embed = None

        if share_codebook:
            if not lfq:
                self.quantizer = VectorQuantize(
                    dim = hidden_size // code_shrink_rate,
                    output_dim=encoder_hidden_size,
                    codebook_dim = code_dim,
                    codebook_size = n_e,     # codebook size
                    decay = decay,
                    diversity_weight=self.w_diversity,
                    commitment_weight = self.w_commit,   # the weight on the commitment loss
                    use_cosine_sim = use_cosine_sim,
                    threshold_ema_dead_code = dead_code_threshold,
                    kmeans_init = quantize_kmeans_init,   # set to True
                    kmeans_iters = 10,
                    ema_update = ema_update,
                    if_force_sync=if_force_sync
                )
            else:
                self.quantizer = LFQ(
                    dim=hidden_size // code_shrink_rate,
                    output_dim=encoder_hidden_size,
                    codebook_size=2**(code_dim//2),
                    entropy_loss_weight=self.w_diversity,
                    commitment_loss_weight=self.w_commit,
                    diversity_gamma=10.,
                    num_codebooks=2,
                    ema_update = ema_update
                )
        else:
            n_e_list = [self.n_e]*self.K if not n_e_cfg else n_e_cfg
            self.n_e_list = n_e_list
            if not lfq:
                self.quantizer = nn.ModuleList([
                    VectorQuantize(
                    n_e,
                    code_dim,
                    beta=0.25,
                    latent_dim=hidden_size // code_shrink_rate
                    )  for k in range(self.K)]
                )
            else:
                self.quantizer = nn.ModuleList([
                    LFQ(
                        dim=hidden_size // code_shrink_rate,
                        output_dim=encoder_hidden_size,
                        codebook_size=2**(code_dim//2),
                        entropy_loss_weight=self.w_diversity,
                        commitment_loss_weight=self.w_commit,
                        diversity_gamma=10.,
                        num_codebooks=2,
                        ema_update = ema_update
                    ) for k in range(self.K)]
                )
        self.initialize_weights()
        
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
            
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
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    
    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward
    
    def remove_tokens(self, token_idx_list):
        if len(token_idx_list) == 0:
            self.keep_mask = None
        else:
            self.keep_mask = torch.ones(self.n_tokens).reshape(self.K, -1)
            self.keep_mask[token_idx_list] = 0
            self.keep_mask = self.keep_mask.flatten()

    def forward_quantizer(self, quantizer, x):
        if not self.lfq:
            log_dict = {
                'num_active_codes': 1,
                'perplexity': 0,
                'num_reactivate': 0,
                'commitment_loss': 0,
                'deterministic_entropy': 0,
                'diversity_entropy': 0,
            }

            outs_q, loss, info = quantizer(x)

            indices = info[2]
            avg_probs = info[5]
            cluster_size = info[1]
            total_codes = info[6]
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            # num_active_codes = (log_dict['avg_probs_accu'] > len(indices) / self.n_e / 100).sum().item()
            num_active_codes = (cluster_size > (total_codes / self.n_e / 5.0)).sum().item()
            diversity_entropy, deterministic_entropy = info[3], info[4]

            log_dict['num_active_codes'] = num_active_codes
            log_dict['perplexity'] = perplexity.item()
            log_dict['commitment_loss'] = loss.item()
            log_dict['diversity_entropy'] = diversity_entropy
            log_dict['deterministic_entropy'] = deterministic_entropy

        else:
            ret, losses = quantizer(x, self.quantizer_temp, return_loss_breakdown=True)
            outs_q, indices, loss = ret.quantized, ret.indices, ret.entropy_aux_loss
            log_dict = {
                'num_active_codes': 1,
                'num_reactivate': 0,
                'perplexity': 0,
                'commitment_loss': losses.commitment.item(),
                'deterministic_entropy': losses.per_sample_entropy.item(),
                'diversity_entropy': losses.batch_entropy.item(),
            }
        return outs_q, indices, loss, log_dict
    
    def get_encoder_outs(self, x):
        outs = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i >= self.depth - self.K:
                outs.append(x)
        if self.reverse_code:
            outs.reverse()
        assert len(outs) == self.K
        outs = torch.cat(outs, dim=1)
        return outs
    
    def forward_enc(self, x=None, hidden_states=None, d=None):
        """
        Forward pass of feature encoder.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        d: N, the depth for each sample
        """
        x = self.x_embedder(x) + self.pos_embed
        if hidden_states is None:
            outs = self.get_encoder_outs(x)
            if self.pre_norm:
                outs = self.final_layer_norm(outs)
            # outs = self.final_layer_norm(outs)
            to_quantizer_features = outs
            # to_quantizer_features = self.final_layer_norm2(to_quantizer_features)
            perplexity_list = []
            deterministic_list = []
            if self.share_codebook:
                outs_q, indices, loss, log_dict = \
                    self.forward_quantizer(self.quantizer, to_quantizer_features)
                perplexity_list.append(log_dict['perplexity'])
                deterministic_list.append(log_dict["deterministic_entropy"])
                num_active_codes = log_dict['num_active_codes']
                num_reactivate = log_dict["num_reactivate"]
                commitment_loss = log_dict["commitment_loss"]
                diversity_entropy = log_dict["diversity_entropy"]
                perplexity = log_dict["perplexity"]
            else:
                token_num = int(outs.shape[1] / self.K)
                outs_q, loss, perplexity, num_active_codes, num_reactivate, all_indicies = [], 0.0, 0.0, 0.0, 0.0, []
                commitment_loss, diversity_entropy = 0.0, 0.0
                for i, cur_quantizer in enumerate(self.quantizer):
                    curfeat = to_quantizer_features[:, i*token_num:i*token_num+token_num, :].contiguous()
                    cur_outs_q, indices, cur_loss, log_dict = \
                        self.forward_quantizer(cur_quantizer, curfeat)
                    all_indicies.append(indices)
                    perplexity_list.append(log_dict['perplexity'])
                    deterministic_list.append(log_dict["deterministic_entropy"])
                    # outs_q.append(cur_outs_q)
                    B = indices.shape[0]
                    cur_indices = indices.view(-1,1)
                    cur_outs_q = cur_quantizer.get_output_from_indices(cur_indices)
                    cur_outs_q = cur_outs_q.view(B, -1, cur_outs_q.shape[-1])
                    outs_q.append(cur_outs_q)

                    loss += cur_loss
                    perplexity += log_dict['perplexity']
                    num_active_codes += log_dict['num_active_codes']
                    num_reactivate += log_dict["num_reactivate"]
                    commitment_loss += log_dict["commitment_loss"]
                    diversity_entropy += log_dict["diversity_entropy"]
                outs_q = torch.cat(outs_q, 1)
                loss /= self.K
                perplexity /= self.K
                num_active_codes /= self.K
                num_reactivate /= self.K
                commitment_loss /= self.K
                diversity_entropy /= self.K
            log_dict = {
                "num_active_codes": num_active_codes,
                "num_reactivate": num_reactivate,
                "perplexity": perplexity,
                "perplexity_list": perplexity_list,
                "deterministic_list": deterministic_list,
                "deterministic_entropy": np.array(deterministic_list).mean(),
                "commitment_loss": commitment_loss,
                "diversity_entropy": diversity_entropy,
            }
            
            if self.k_embed is not None:
                outs_q = outs_q + self.k_embed(self.k_embed_indices.expand(len(x), -1))



            if self.post_norm:
                outs_q = self.final_layer_norm3(outs_q)
        else:
            outs_q = hidden_states
            loss = 0.0
            log_dict = {}
        
        if d is None:
            return outs_q, torch.cat(all_indicies, 1)
        
        zero_cond_idx = torch.nonzero(d==-1).squeeze(1).cpu().tolist()
        B, N, P = outs_q.shape[0], outs_q.shape[1], x.shape[1]
        enc_mask = torch.arange(self.K).repeat_interleave(P)[None, ...].expand(B,N).to(d.device)

        enc_mask = (enc_mask <= d.unsqueeze(1))
        attn_mask = enc_mask
        mask_v = enc_mask[..., None].expand_as(outs_q)

        encoder_hidden_states = outs_q * mask_v

        if self.keep_mask is not None:
            keep_mask = self.keep_mask.to(x.device)
            encoder_hidden_states = encoder_hidden_states * keep_mask[...,None][None,...]
        return encoder_hidden_states, outs_q, attn_mask, loss, log_dict

    def forward(self, x=None, hidden_states=None, d=None):
        """
        Forward pass of feature encoder.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        d: N, the depth for each sample
        """
        x = self.x_embedder(x) + self.pos_embed
        if hidden_states is None:
            outs = self.get_encoder_outs(x)
            if self.pre_norm:
                outs = self.final_layer_norm(outs)
            # outs = self.final_layer_norm(outs)
            to_quantizer_features = outs
            # to_quantizer_features = self.final_layer_norm2(to_quantizer_features)
            perplexity_list = []
            deterministic_list = []
            if self.share_codebook:
                outs_q, indices, loss, log_dict = \
                    self.forward_quantizer(self.quantizer, to_quantizer_features)
                perplexity_list.append(log_dict['perplexity'])
                deterministic_list.append(log_dict["deterministic_entropy"])
                num_active_codes = log_dict['num_active_codes']
                num_reactivate = log_dict["num_reactivate"]
                commitment_loss = log_dict["commitment_loss"]
                diversity_entropy = log_dict["diversity_entropy"]
                perplexity = log_dict["perplexity"]
            else:
                token_num = int(outs.shape[1] / self.K)
                outs_q, loss, perplexity, num_active_codes, num_reactivate, all_indicies = [], 0.0, 0.0, 0.0, 0.0, []
                commitment_loss, diversity_entropy = 0.0, 0.0
                for i, cur_quantizer in enumerate(self.quantizer):
                    curfeat = to_quantizer_features[:, i*token_num:i*token_num+token_num, :].contiguous()
                    cur_outs_q, indices, cur_loss, log_dict = \
                        self.forward_quantizer(cur_quantizer, curfeat)
                    all_indicies.append(indices)
                    perplexity_list.append(log_dict['perplexity'])
                    deterministic_list.append(log_dict["deterministic_entropy"])
                    outs_q.append(cur_outs_q)
                    loss += cur_loss
                    perplexity += log_dict['perplexity']
                    num_active_codes += log_dict['num_active_codes']
                    num_reactivate += log_dict["num_reactivate"]
                    commitment_loss += log_dict["commitment_loss"]
                    diversity_entropy += log_dict["diversity_entropy"]
                outs_q = torch.cat(outs_q, 1)
                loss /= self.K
                perplexity /= self.K
                num_active_codes /= self.K
                num_reactivate /= self.K
                commitment_loss /= self.K
                diversity_entropy /= self.K

            log_dict = {
                "num_active_codes": num_active_codes,
                "num_reactivate": num_reactivate,
                "perplexity": perplexity,
                "perplexity_list": perplexity_list,
                "deterministic_list": deterministic_list,
                "deterministic_entropy": np.array(deterministic_list).mean(),
                "commitment_loss": commitment_loss,
                "diversity_entropy": diversity_entropy,
            }

            # outs_q = self.decode_task_layer(outs_q)
            if self.k_embed is not None:
                outs_q = outs_q + self.k_embed(self.k_embed_indices.expand(len(x), -1))

            
            outs_q = self.project_out(outs_q)

            
            if self.post_norm:
                outs_q = self.final_layer_norm3(outs_q)
        else:
            outs_q = hidden_states
            loss = 0.0
            log_dict = {}
        
        if d is None:
            return outs_q, torch.cat(all_indicies, 1)
        
        zero_cond_idx = torch.nonzero(d==-1).squeeze(1).cpu().tolist()
        B, N, P = outs_q.shape[0], outs_q.shape[1], x.shape[1]
        enc_mask = torch.arange(self.K).repeat_interleave(P)[None, ...].expand(B,N).to(d.device)
        # mask_nograd = (mask < d.unsqueeze(1))
        # mask_grad = (mask == d.unsqueeze(1))
        # mask_nograd = mask_nograd[..., None].expand_as(outs)
        # mask_grad = mask_grad[..., None].expand_as(outs)
        # encoder_hidden_states = outs * mask_grad + outs.detach() * mask_nograd
        enc_mask = (enc_mask <= d.unsqueeze(1))
        attn_mask = enc_mask
        mask_v = enc_mask[..., None].expand_as(outs_q)

        # encoder_hidden_states = self.final_layer_norm(outs_q * mask)
        encoder_hidden_states = outs_q * mask_v
        
        # final_attn_mask = torch.ones(attn_mask.shape, dtype=attn_mask.dtype, device=attn_mask.device)
        # for idx in range(attn_mask.shape[0]):
        #     if idx in zero_cond_idx:
        #         final_attn_mask[idx,...] = zero_mask
        #     else:
        #         final_attn_mask[idx,...] = attn_mask[idx,...]
        
        if self.keep_mask is not None:
            keep_mask = self.keep_mask.to(x.device)
            encoder_hidden_states = encoder_hidden_states * keep_mask[...,None][None,...]
            # final_attn_mask = (final_attn_mask * keep_mask[None, ...]).bool()
        return encoder_hidden_states, outs_q, attn_mask, loss, log_dict
    
class DiTiEncoder(Encoder):
    def __init__(
        self, K, input_size=32, encoder_hidden_size=256, patch_size=8, in_channels=4,
        hidden_size=256, depth=None, num_heads=4, mlp_ratio=4.0, w_diversity=0.0, w_commit=1.0,
        n_e = 8192, n_e_cfg=None, share_codebook=False, reverse_code=False, code_dim=32,
        quantize_kmeans_init=True, decay=0.99, dead_code_threshold=0.0, quantizer_temp=10.,
        use_cosine_sim=True, pre_norm=False, post_norm=True, lfq=False, k_embed=False
    ):
        assert hidden_size % K == 0
        super().__init__(
            K, input_size, encoder_hidden_size, patch_size, in_channels, hidden_size, depth, num_heads,
            mlp_ratio, w_diversity, w_commit, n_e, n_e_cfg, share_codebook, reverse_code, code_dim,
            quantize_kmeans_init, decay, dead_code_threshold, quantizer_temp, use_cosine_sim,
            pre_norm, post_norm, lfq, k_embed=k_embed, code_shrink_rate=K
        )
    
    def get_encoder_outs(self, x):
        for block in self.blocks:
            x = block(x)
        outs = torch.cat(x.chunk(self.K, dim=2), dim=1)
        return outs
    

class SelfCondDiT(nn.Module):
    """
    Condition on encoded feature to reconstruct
    """
    def __init__(
        self,
        input_size=64,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.0,
        learn_sigma=True,
        encoder_hidden_size=256,
        cross_modulate=False,
        train_filter=["cross_attn", "cross_modulation", "cross_norm"]
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.class_dropout_prob = class_dropout_prob
        self.train_filter = train_filter
        
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        
        self.blocks = nn.ModuleList([
            DiTCrossAttnBlock(
                hidden_size, encoder_hidden_size, num_heads,
                mlp_ratio=mlp_ratio, cross_modulate=cross_modulate
            ) for _ in range(depth)
        ])
        # print(f"Encoder Parameters: {sum(p.numel() for p in self.encoder.parameters()):,}")
        self.initialize_weights()

    def freeze(self):
        for name, param in self.named_parameters():
            if self.train_filter is not None and len(self.train_filter) > 0:
                if any(item in name for item in self.train_filter) and 't_embedder.mlp' not in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
    def parameters(self, recurse=True):
        """
        Override the parameters() function to yield only parameters
        whose names contain 'cross_modulation'.

        :param recurse: Whether to recurse into submodules (default: True)
        :param self.train_filter: The string to filter parameter names by (default: 'cross_modulation')
        :return: An iterator over module parameters matching the filter
        """
        for name, param in self.named_parameters(recurse=recurse):
            if self.train_filter is not None and len(self.train_filter) > 0:
                if any(item in name for item in self.train_filter) and 't_embedder.mlp' not in name:
                    yield param
            else:
                yield param
                 
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            if block.cross_modulation is not None:
                nn.init.constant_(block.cross_modulation[-1].weight, 0)
                nn.init.constant_(block.cross_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
            
    def drop_cond(self, encoder_hidden_states):
        drop_ids = torch.rand(
            encoder_hidden_states.shape[0], device=encoder_hidden_states.device
        ) < self.class_dropout_prob
        encoder_hidden_states[drop_ids, :, :] = 0
        return encoder_hidden_states
    
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
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    
    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward
    
    def forward(self, x, t, encoder_hidden_states, mask=None):
        if self.training:
            encoder_hidden_states = self.drop_cond(encoder_hidden_states)
        x = self.x_embedder(x) + self.pos_embed        # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                         # (N, D)
        for i, block in enumerate(self.blocks):
            x = block(x, t, encoder_hidden_states, mask)  # (N, T, D)
            # x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c)
        x = self.final_layer(x, t)                     # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                         # (N, out_channels, H, W)
        return x
    
def SCDiT_XL2(**kwargs):
    return SelfCondDiT(
        depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs
    )
    
def Enc_Tiny_8(**kwargs):
    return Encoder(patch_size=8, hidden_size=256, num_heads=4, **kwargs)

def Enc_Base_8(**kwargs):
    return Encoder(patch_size=8, hidden_size=768, num_heads=12, **kwargs)

def Enc_Base_16(**kwargs):
    return Encoder(patch_size=16, hidden_size=256, num_heads=4, **kwargs)

def Enc_L_8(**kwargs):
    assert kwargs["K"] <= 24, "Enc-L/8 supports K up to 24."
    return Encoder(patch_size=8, hidden_size=768, num_heads=16, depth=24, **kwargs)

def Enc_H_8(**kwargs):
    assert kwargs["K"] <= 32, "Enc-H/8 supports K up to 32."
    return Encoder(patch_size=8, hidden_size=768, num_heads=16, depth=32, **kwargs)

def Enc_H_8_XS(**kwargs):
    assert kwargs["K"] <= 32, "Enc-H/8 supports K up to 32."
    return Encoder(patch_size=8, hidden_size=256, num_heads=16, depth=32, **kwargs)

def Enc_H_8_XS_24(**kwargs):
    assert kwargs["K"] <= 32, "Enc-H/8 supports K up to 32."
    return Encoder(patch_size=8, hidden_size=256, num_heads=16, depth=24, **kwargs)

def Enc_H2_8_XS(**kwargs):
    assert kwargs["K"] <= 40, "Enc-H/8 supports K up to 40."
    return Encoder(patch_size=8, hidden_size=256, num_heads=16, depth=40, **kwargs)

def Enc_H3_8_XS(**kwargs):
    assert kwargs["K"] <= 48, "Enc-H/8 supports K up to 48."
    return Encoder(patch_size=8, hidden_size=256, num_heads=16, depth=48, **kwargs)

def Enc_B_8_XS(**kwargs):
    assert kwargs["K"] <= 16, "Enc-B/8 supports K up to 16."
    return Encoder(patch_size=8, hidden_size=256, num_heads=16, depth=16, **kwargs)

def Enc_H_4_XS(**kwargs):
    assert kwargs["K"] <= 32, "Enc-H/4 supports K up to 32."
    return Encoder(patch_size=4, hidden_size=64, num_heads=8, depth=32, **kwargs)

def Enc_B_4_XS(**kwargs):
    assert kwargs["K"] <= 16, "Enc-B/4 supports K up to 16."
    return Encoder(patch_size=4, hidden_size=64, num_heads=8, depth=16, **kwargs)

def Enc_H_8_XXS(**kwargs):
    assert kwargs["K"] <= 32, "Enc-H/8 supports K up to 32."
    return Encoder(patch_size=8, hidden_size=128, num_heads=8, depth=32, **kwargs)

# NOTE: DiTiEnc models need to specify depth
def DiTiEnc_Base_8(**kwargs):
    num_heads = 16
    assert num_heads % kwargs["K"] == 0    # K can be 1, 2, 4, 8, 16
    return DiTiEncoder(patch_size=8, hidden_size=768, num_heads=num_heads, depth=16, **kwargs)

def DiTiEnc_B_8_XS(**kwargs):
    num_heads = 16
    assert num_heads % kwargs["K"] == 0    # K can be 1, 2, 4, 8, 16
    return DiTiEncoder(patch_size=8, hidden_size=256, num_heads=num_heads, depth=16, **kwargs)

DiT_models = {
    'SCDiT-XL/2': SCDiT_XL2
}

Enc_models = {
    'Enc-Tiny/8': Enc_Tiny_8,
    'Enc-Base/8': Enc_Base_8,
    'Enc-L/8': Enc_L_8,
    'Enc-H/8': Enc_H_8,
    'Enc-H/8-XS': Enc_H_8_XS,
    'Enc-H/8-XS-24': Enc_H_8_XS_24,
    'Enc-H2/8-XS': Enc_H2_8_XS,
    'Enc-H3/8-XS': Enc_H3_8_XS,
    'Enc-B/8-XS': Enc_B_8_XS,
    'Enc-H/4-XS': Enc_H_4_XS,
    'Enc-B/4-XS': Enc_B_4_XS,
    'Enc-H/8-XXS': Enc_H_8_XXS,
    'Enc-Base/16': Enc_Base_16,
    'DiTiEnc-B/8': DiTiEnc_Base_8,
    'DiTiEnc-B/8-XS': DiTiEnc_B_8_XS
}

