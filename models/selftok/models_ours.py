import torch
import torch.nn as nn
import numpy as np
import math
from .sd3.mmdit import PatchEmbed
from mimogpt.models.selftok.models import DiT, DiTBlock, get_2d_sincos_pos_embed, modulate, TimestepEmbedder, FinalLayer
import torch.nn.functional as F
from mimogpt.tokenizer.selftok import LFQ
from .quantizer import construct_quantizer
from .modules import DiTCrossAttnBlock, ViTBlock, QFormer, DualBlock, ConcatBlock, DiTDualBlock
from einops import rearrange
# import xformers.ops

try:
    from torch.utils.checkpoint import checkpoint
    print("Using gradient checkpointing...")
except:
    print("Disabling gradient checkpointing...")
    assert False

from torch.utils.checkpoint import checkpoint
def ckpt_wrapper(module):
    def ckpt_forward(*inputs):
        outputs = module(*inputs)
        return outputs
    return ckpt_forward

class Encoder(nn.Module):
    def __init__(
        self, K, input_size=32, encoder_hidden_size=256, patch_size=8, in_channels=4,
        hidden_size=256, depth=None, num_heads=4, mlp_ratio=4.0, w_diversity=0.0, w_commit=1.0,
        n_e = 8192, n_e_cfg=None, share_codebook=False, reverse_code=False, code_dim=32,
        quantize_kmeans_init=True, decay=0.99, dead_code_threshold=0.0, quantizer_temp=10.0,
        use_cosine_sim=True, pre_norm=False, post_norm=True, lfq=False,
        k_embed=False, encoder_out_dim=None, ema_update=True, gradient_checkpointing=False, 
        pos_embed_max_size=None, smart_react=False, **kwargs
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
        self.pos_embed_max_size = pos_embed_max_size
        encoder_out_dim = encoder_out_dim or hidden_size
        self.gradient_checkpointing = gradient_checkpointing
        self.x_embedder = PatchEmbed(img_size=input_size,patch_size=patch_size, in_chans=in_channels, embed_dim=hidden_size, bias=True)
        if pos_embed_max_size is not None:
            num_patches = pos_embed_max_size * pos_embed_max_size
            self.x_embedder.num_patches = pos_embed_max_size * pos_embed_max_size
        else:
            num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        if num_patches is not None:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        else:
            self.pos_embed = None

        self.n_tokens = K * (input_size // patch_size) ** 2
        
        self.blocks = nn.ModuleList([
            ViTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer_norm = nn.LayerNorm(encoder_out_dim, eps=1e-6)  
        self.final_layer_norm2 = nn.LayerNorm(code_dim, eps=1e-6)
        self.final_layer_norm3 = nn.LayerNorm(encoder_hidden_size, eps=1e-6)  
        
        self.share_codebook = share_codebook
        self.reverse_code = reverse_code
        self.code_dim = code_dim
        self.lfq = lfq
        self.quantizer_temp = quantizer_temp
        
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
                smart_re_K = K if smart_react else 0
                self.quantizer = construct_quantizer(
                    latent_dim = encoder_out_dim,
                    output_dim = encoder_hidden_size,
                    code_dim = code_dim,
                    codebook_size = n_e,     # codebook size
                    ema_decay = decay,
                    diversity_weight = self.w_diversity,
                    commitment_weight = self.w_commit,   # the weight on the commitment loss
                    dead_code_threshold = dead_code_threshold,
                    use_ema = ema_update,
                    smart_re_K=smart_re_K
                )
            else:
                self.quantizer = LFQ(
                    dim=encoder_out_dim,
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
                    construct_quantizer(
                        latent_dim =encoder_out_dim,
                        output_dim = encoder_hidden_size,
                        code_dim = code_dim,
                        codebook_size = n_e_list[k],     # codebook size
                        ema_decay = decay,
                        diversity_weight = self.w_diversity,
                        commitment_weight = self.w_commit,   # the weight on the commitment loss
                        dead_code_threshold = dead_code_threshold,
                        use_ema = ema_update
                    ) for k in range(self.K)]
                )
            else:
                self.quantizer = nn.ModuleList([
                    LFQ(
                        dim=encoder_out_dim,
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
        if self.pos_embed is not None:
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1],
                int(self.x_embedder.num_patches ** 0.5)
            )
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
    
    def remove_tokens(self, token_idx_list):
        if len(token_idx_list) == 0:
            self.keep_mask = None
        else:
            self.keep_mask = torch.ones(self.n_tokens).reshape(self.K, -1)
            self.keep_mask[token_idx_list] = 0
            self.keep_mask = self.keep_mask.flatten()

    def forward_quantizer(self, quantizer, x):
        if not self.lfq:
            outs_q, indices, loss, log_dict = quantizer(x)
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
            if self.gradient_checkpointing:
                x = checkpoint(ckpt_wrapper(block), x, use_reentrant=False)
            else:
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
        if self.pos_embed_max_size is not None:
            hw = x.shape[-2:]
            x = self.x_embedder(x) + self.cropped_pos_embed(hw)
        else:
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

    def get_encoder_mask(self, x, outs_q, d):
        B, N, P = outs_q.shape[0], outs_q.shape[1], x.shape[1]
        enc_mask = torch.arange(self.K).repeat_interleave(P)[None, ...].expand(B,N).to(d.device)
        return (enc_mask <= d.unsqueeze(1))
    
    def calc_entropy(self, p):
        ap = p.mean(dim=0)
        p_log_p = ap * torch.log(ap)
        entropy_to_max = -p_log_p.sum(dim=-1)
        # E(H(p))
        p_log_p = p * torch.log(p)
        entropy_to_min = -p_log_p.sum(dim=-1)
        entropy_to_min = entropy_to_min
        return entropy_to_min
    
    def get_perplexity_list(self, indices, chunks=50):
        probs = F.one_hot(indices, num_classes=self.n_e).float().mean(dim=0)
        chunk_probs = torch.stack([t.mean(dim=0) for t in probs.tensor_split(chunks, dim=0)],dim=0).float()
        if not hasattr(self, 'tracker_per_k'):
            self.tracker_per_k = torch.zeros_like(chunk_probs)
            self.tracker_per_k += 1.0 / self.n_e
        if self.training:
            self.tracker_per_k.mul_(0.99).add_(chunk_probs * 0.01)
        ap = self.tracker_per_k
        perplexity_list = torch.exp(-torch.sum(ap * torch.log(ap + 1e-10), dim=1)).tolist()
        deterministic_list = self.calc_entropy(ap).tolist()
        return perplexity_list, deterministic_list

    def cropped_pos_embed(self, hw):
        assert self.pos_embed_max_size is not None
        p = self.x_embedder.patch_size[0]
        h, w = hw
        # patched size
        h = h // p
        w = w // p
        assert h <= self.pos_embed_max_size, (h, self.pos_embed_max_size)
        assert w <= self.pos_embed_max_size, (w, self.pos_embed_max_size)
        top = (self.pos_embed_max_size - h) // 2
        left = (self.pos_embed_max_size - w) // 2
        spatial_pos_embed = rearrange(
            self.pos_embed,
            "1 (h w) c -> 1 h w c",
            h=self.pos_embed_max_size,
            w=self.pos_embed_max_size,
        )
        spatial_pos_embed = spatial_pos_embed[:, top : top + h, left : left + w, :]
        spatial_pos_embed = rearrange(spatial_pos_embed, "1 h w c -> 1 (h w) c")
        return spatial_pos_embed
    
    def forward(self, x=None, hidden_states=None, d=None):
        """
        Forward pass of feature encoder.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        d: N, the depth for each sample
        """
        if self.pos_embed_max_size is not None:
            hw = x.shape[-2:]
            x = self.x_embedder(x)
            x = x + self.cropped_pos_embed(hw)
        else:
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
                perplexity_list, deterministic_list = self.get_perplexity_list(indices)
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

            if self.post_norm:
                outs_q = self.final_layer_norm3(outs_q)
        else:
            outs_q = hidden_states
            loss = 0.0
            log_dict = {}
        
        if d is None:
            return outs_q, torch.cat(all_indicies, 1)
        
        # zero_cond_idx = torch.nonzero(d==-1).squeeze(1).cpu().tolist()
        # mask_nograd = (mask < d.unsqueeze(1))
        # mask_grad = (mask == d.unsqueeze(1))
        # mask_nograd = mask_nograd[..., None].expand_as(outs)
        # mask_grad = mask_grad[..., None].expand_as(outs)
        # encoder_hidden_states = outs * mask_grad + outs.detach() * mask_nograd
        enc_mask = self.get_encoder_mask(x, outs_q, d)
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
        use_cosine_sim=True, pre_norm=False, post_norm=True, lfq=False,
        k_embed=False, ema_update=True, **kwargs
    ):
        assert hidden_size % K == 0
        super().__init__(
            K, input_size, encoder_hidden_size, patch_size, in_channels, hidden_size, depth, num_heads,
            mlp_ratio, w_diversity, w_commit, n_e, n_e_cfg, share_codebook, reverse_code, code_dim,
            quantize_kmeans_init, decay, dead_code_threshold, quantizer_temp, use_cosine_sim,
            pre_norm, post_norm, lfq, k_embed=k_embed, encoder_out_dim=hidden_size//K, ema_update=ema_update
        )
    
    def get_encoder_outs(self, x):
        for block in self.blocks:
            x = block(x)
        outs = torch.cat(x.chunk(self.K, dim=2), dim=1)
        return outs


'''
    Encoder with a special input: mode, which can be either
        - 'qformer' with cross attention interaction between query and latent
        - 'concat' with self attention interaction between query and latent
        - 'dual-xx' with self attention interaction between query and latent, but query has its own transformer
            - xx='cross': query as q, latent as kv
            - xx='self': [query,latent] into self-attention
'''
class QformerEncoder(Encoder):
    def __init__(
        self, K, input_size=32, encoder_hidden_size=256, patch_size=8, in_channels=4,
        hidden_size=256, depth=None, num_heads=4, mlp_ratio=4.0, w_diversity=0.0, w_commit=1.0,
        n_e = 8192, n_e_cfg=None, share_codebook=False, reverse_code=False, code_dim=32,
        quantize_kmeans_init=True, decay=0.99, dead_code_threshold=0.0, quantizer_temp=10.0,
        use_cosine_sim=True, pre_norm=False, post_norm=True, lfq=False, qformer_mode='qformer',
        k_embed=False, ema_update=True, gradient_checkpointing=False, pos_embed_max_size=None,
        xavier_init=False, smart_react=False, **kwargs
    ):
        super().__init__(
            K, input_size, encoder_hidden_size, patch_size, in_channels, hidden_size, depth, num_heads,
            mlp_ratio, w_diversity, w_commit, n_e, n_e_cfg, share_codebook, reverse_code, code_dim,
            quantize_kmeans_init, decay, dead_code_threshold, quantizer_temp, use_cosine_sim,
            pre_norm, post_norm, lfq, k_embed=k_embed, encoder_out_dim=kwargs['query_dim'],
            ema_update=ema_update, gradient_checkpointing=gradient_checkpointing, 
            pos_embed_max_size=pos_embed_max_size, smart_react=smart_react, **kwargs
        )
        qformer_depth = depth
        self.num_query_token = K # num_query_token
        query_dim = kwargs['query_dim']
        self.query_tokens = nn.Parameter(torch.zeros(1, self.num_query_token, query_dim))
        self.query_tokens.data.normal_(mean=0.0, std=0.02)
        self.mode = qformer_mode
        if self.mode == 'qformer':
            self.qformer = QFormer(
                self.num_query_token, hidden_size, query_dim, num_heads, qformer_depth, mlp_ratio=mlp_ratio
            )
            self.blocks = nn.Identity()
        elif self.mode == 'dual':
            self.blocks = nn.ModuleList([
                DualBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **kwargs) for _ in range(depth)
            ])
        elif self.mode == 'concat':
            self.blocks = nn.ModuleList([
                ConcatBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **kwargs) for _ in range(depth)
            ])
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        if xavier_init:
            self.apply(_basic_init)

    def get_encoder_outs(self, x):
        query_tokens = self.query_tokens.expand(x.shape[0], -1, -1)
        if self.mode == 'qformer':
            query_tokens = self.qformer(x, query_tokens) # [B, L, C]
        elif self.mode == 'concat':
            # cat_x = torch.cat([query_tokens, x], 1)
            # for i, block in enumerate(self.blocks):
            #     cat_x = block(cat_x)
            # out = cat_x[:, :query_tokens.shape[1], :]
            for i, block in enumerate(self.blocks):
                if self.gradient_checkpointing:
                    x, query_tokens = checkpoint(ckpt_wrapper(block), x, query_tokens, use_reentrant=False)
                else:
                    x, query_tokens = block(x, query_tokens)
        elif self.mode == 'dual':
            for i, block in enumerate(self.blocks):
                x, query_tokens = block(x, query_tokens)
        else:
            raise ValueError("Unknown mode to QFormerEncoder.")
        return query_tokens
    
    def get_encoder_mask(self, x, outs_q, d):
        # no spatial token, so num patches is essentially 1
        B, N = outs_q.shape[0], outs_q.shape[1]
        enc_mask = torch.arange(self.K).repeat_interleave(1)[None, ...].expand(B,N).to(d.device)
        return (enc_mask <= d.unsqueeze(1))


class SelfCondDiT(nn.Module):
    """
    Condition on encoded feature to reconstruct
    dit_attention:
        - ca (default): cross attn
        - bi: bidirection
        - uni: uni-direction (token=>latent)
        - uni-0: zero-init with uni-direction (token=>latent)
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
        train_filter=["cross_attn", "cross_modulation", "cross_norm"],
        freeze_filter=[],
        dit_attention='ca',
        gradient_checkpointing=False,
        K=None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.class_dropout_prob = class_dropout_prob
        self.train_filter = train_filter
        self.freeze_filter = freeze_filter
        self.gradient_checkpointing = gradient_checkpointing
        self.x_embedder = PatchEmbed(img_size=input_size,patch_size=patch_size, in_chans=in_channels, embed_dim=hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        
        self.dit_attention = dit_attention
        if self.dit_attention=='ca':
            self.blocks = nn.ModuleList([
                DiTCrossAttnBlock(
                    hidden_size, encoder_hidden_size, num_heads,
                    mlp_ratio=mlp_ratio, cross_modulate=cross_modulate
                ) for _ in range(depth)
            ])
            self.bidirectional = None       # not applicable
            self.zero_init = None           # not applicable
        else:
            if dit_attention != 'bi':
                self.t_embedder_q = TimestepEmbedder(encoder_hidden_size)
                self.query_proj = nn.Identity()
                q_size = encoder_hidden_size
            else:
                self.t_embedder_q = nn.Identity()
                self.query_proj = nn.Linear(encoder_hidden_size, hidden_size, bias=True) \
                    if encoder_hidden_size != hidden_size else nn.Identity()
                q_size = hidden_size
                
            self.blocks = nn.ModuleList([
                DiTDualBlock(
                    hidden_size, q_size, num_heads, query_heads=num_heads,
                    mlp_ratio=mlp_ratio, dit_attention=dit_attention
                ) for _ in range(depth)
            ])
        # print(f"Encoder Parameters: {sum(p.numel() for p in self.encoder.parameters()):,}")
        self.initialize_weights()

    def freeze(self):
        freezed_param_names = []
        train_param_names = []
        for name, param in self.named_parameters():
            if self.train_filter is not None:
                # if not train all
                if any(item in name for item in self.train_filter) and \
                    not any(item in name for item in self.freeze_filter):
                    param.requires_grad = True
                    train_param_names.append(name)
                else:
                    param.requires_grad = False
                    freezed_param_names.append(name)
            elif not any(item in name for item in self.freeze_filter):
                param.requires_grad = True
                train_param_names.append(name)
            else:
                param.requires_grad = False
                freezed_param_names.append(name)
        return train_param_names, freezed_param_names
        
    def parameters(self, recurse=True):
        """
        Override the parameters() function to yield only parameters
        whose names contain 'cross_modulation'.

        :param recurse: Whether to recurse into submodules (default: True)
        :param self.train_filter: The string to filter parameter names by (default: 'cross_modulation')
        :return: An iterator over module parameters matching the filter
        """
        for name, param in self.named_parameters(recurse=recurse):
            if self.train_filter is not None:
                if any(item in name for item in self.train_filter) and \
                    not any(item in name for item in self.freeze_filter):
                # 't_embedder.mlp' not in name:
                    yield param
            elif not any(item in name for item in self.freeze_filter):
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
        if self.pos_embed is not None:
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
            if hasattr(block, 'cross_modulation'):
                nn.init.constant_(block.cross_modulation[-1].weight, 0)
                nn.init.constant_(block.cross_modulation[-1].bias, 0)
            if hasattr(block, 'adaLN_modulation_q'):
                nn.init.constant_(block.adaLN_modulation_q[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation_q[-1].bias, 0)

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
        t_emb = self.t_embedder(t)                         # (N, D)
        if self.dit_attention != 'ca':
            if self.dit_attention != 'bi':
                t_q = self.t_embedder_q(t)
            else:
                t_q = t_emb
                encoder_hidden_states = self.query_proj(encoder_hidden_states)
            for i, block in enumerate(self.blocks):
                if self.gradient_checkpointing:
                    x, encoder_hidden_states = checkpoint(ckpt_wrapper(block), x, t_emb, t_q, encoder_hidden_states, mask, use_reentrant=False)
                else:
                    x, encoder_hidden_states = block(x, t_emb, t_q, encoder_hidden_states, mask)
        else:
            for i, block in enumerate(self.blocks):
                if self.gradient_checkpointing:
                    x = checkpoint(ckpt_wrapper(block), x, t_emb, encoder_hidden_states, mask, use_reentrant=False)
                else:
                    x = block(x, t_emb, encoder_hidden_states, mask)  # (N, T, D)
 
            # x = checkpoint(self.ckpt_wrapper(block), x, c)
        x = self.final_layer(x, t_emb)                     # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                         # (N, out_channels, H, W)
        return x