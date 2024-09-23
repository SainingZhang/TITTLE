import torch
from torch import einsum
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
import torch.nn.functional as F
import torch.distributed as dist
from mimogpt.tokenizer.selftok import VectorQuantize as VectorQuantize_EMA
from mimogpt.tokenizer.selftok.vector_quantize_pytorch import gumbel_sample

def calc_entropy(input_tensor, min_ref=None):
    assert len(input_tensor.shape) == 2
    p = input_tensor.softmax(dim=-1)
    # H(E(p))
    ap = p.mean(dim=0)
    p_log_p = ap * torch.log(ap)
    entropy_to_max = -p_log_p.sum(dim=-1)
    # E(H(p))
    p_log_p = p * torch.log(p)
    entropy_to_min = -p_log_p.sum(dim=-1)
    if min_ref:
        entropy_to_min = torch.maximum(entropy_to_min, torch.ones_like(entropy_to_min) * min_ref)
    entropy_to_min = entropy_to_min.mean()
    return entropy_to_max, entropy_to_min

def ema_inplace(old, new, decay):
    old.mul_(decay).add_(new * (1 - decay))


def construct_quantizer(
        latent_dim, code_dim, output_dim, codebook_size,
        use_ema, diversity_weight, commitment_weight,
        dead_code_threshold=0.0, ema_decay=0.99, smart_re_K=0):
    if use_ema:
        constructor = VectorQuantize_EMA
        args = dict(
            dim=latent_dim,
            output_dim=output_dim,
            codebook_dim=code_dim,
            codebook_size=codebook_size,
            ema_update=True,
            decay=ema_decay,
            kmeans_init=True,
            kmeans_iters=10,
            threshold_ema_dead_code=dead_code_threshold,
            use_cosine_sim=True,
            commitment_weight=commitment_weight,
            diversity_weight=diversity_weight,
            smart_re_K=smart_re_K,
        )
        # constructor = EMAQuantizer
        # args = dict(
        #     latent_dim=latent_dim,
        #     n_e=codebook_size,
        #     e_dim=code_dim,
        #     output_dim=output_dim,
        #     dead_code_threshold=dead_code_threshold,
        #     commitment_weight=commitment_weight,
        #     diversity_weight=diversity_weight,
        #     decay=ema_decay
        # )
    else:
        constructor = VectorQuantizer_L2norm
        args = dict(
            latent_dim=latent_dim,
            n_e=codebook_size,
            e_dim=code_dim,
            output_dim=output_dim,
            dead_code_threshold=dead_code_threshold,
            commitment_weight=commitment_weight,
            diversity_weight=diversity_weight,
        )
    return constructor(**args)


class VectorQuantizer_L2norm(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """

    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(
        self,
        n_e,
        e_dim,
        output_dim,
        commitment_weight=1.0,
        diversity_weight=1.0,
        dead_code_threshold=0.05,
        beta=0.25,
        latent_dim=None,
        remap=None,
        unknown_index="random",
        sane_index_shape=True,
        legacy=True,
        preserve_gradient=True,
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy
        self.w_commit = commitment_weight
        self.w_diversity = diversity_weight
        self.preserve_gradient = preserve_gradient
        self.dead_code_threshold = dead_code_threshold

        self.norm = lambda x: F.normalize(x, dim=-1)
        self.project_in = nn.Linear(latent_dim, e_dim) if latent_dim != e_dim else nn.Identity()
        self.project_out = nn.Linear(e_dim, output_dim) if output_dim != e_dim else nn.Identity()
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        self.register_buffer("cluster_size", torch.ones(self.n_e) * dead_code_threshold * 2.0)      # avg cluster size for each code among n_e codes
        self.reset_cluster_size = dead_code_threshold * 1.2
        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(
                f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z):
        z = self.project_in(z)
        z_flattened_ori = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        z_flattened = self.norm(z_flattened_ori)
        num_reactivate = self.expire_codes(z_flattened_ori, None)
        embedding_norm = self.norm(self.embedding.weight)
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(embedding_norm**2, dim=1)
            - 2 * torch.einsum("bd,dn->bn", z_flattened, rearrange(embedding_norm, "n d -> d n"))
        )
        # d = einsum('n d, c d -> n c', z_flattened, embedding_norm)

        scaled_distances = d * 10.0
        entropy_to_max, entropy_to_min = calc_entropy(
            scaled_distances.flatten(end_dim=-2)
        )
        # diversity_loss = entropy_to_min - entropy_to_max
        diversity_loss = -entropy_to_max
        diversity_entropy = entropy_to_max.detach()
        deterministic_entropy = entropy_to_min.detach()
        min_encoding_indices = torch.argmin(d, dim=1)
        # min_encoding_indices, onehot = gumbel_sample(d, dim=-1, temperature=1.0, training=self.training)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        z_q, z = self.norm(z_q), self.norm(z)

        # compute loss for embedding
        if not self.legacy:
            commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)
        else:
            commit_loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        loss = self.w_commit * commit_loss + self.w_diversity * diversity_loss

        # preserve gradients
        if self.preserve_gradient:
            z_q = z + (z_q - z).detach()
        z_q = self.project_out(z_q)
        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten

        # calc cluster size
        total_codes = len(z_flattened) * float(dist.get_world_size())
        onehot_assignments = F.one_hot(min_encoding_indices.flatten(), num_classes=self.n_e).float().sum(dim=0)
        dist.all_reduce(onehot_assignments)
        # print(f"total assignments: {onehot_assignments.sum().item()}, total codes: {total_codes}.")
        ema_inplace(self.cluster_size.data, onehot_assignments * self.n_e / total_codes, 0.99)
        probs = onehot_assignments / total_codes

        perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10)))
        num_active_codes = (self.cluster_size > 0.2).sum()

        log_dict = {
            'num_active_codes': num_active_codes.item(),
            'num_reactivate': num_reactivate,
            'perplexity': perplexity.item(),
            'commitment_loss': commit_loss.item(),
            'deterministic_entropy': deterministic_entropy.item(),
            'diversity_entropy': diversity_entropy.item(),
        }
        
        # return z_q, loss, (perplexity, self.cluster_size, min_encoding_indices, diversity_entropy, deterministic_entropy, probs, total_codes)
        return z_q, min_encoding_indices, loss, log_dict
    
    def expire_codes(self, batch_samples, matched_indices):
        # first check if already no dead code
        expired_codes = self.cluster_size < self.dead_code_threshold
        if not torch.any(expired_codes):
            return 0
        # then filter off any code that is matched in current batch
        if matched_indices is not None:
            all_indices = [torch.empty_like(matched_indices) for _ in range(dist.get_world_size())]
            dist.all_gather(all_indices, matched_indices)
            all_indices = torch.stack(all_indices)
            expired_codes[all_indices.unique()] = False
            if not torch.any(expired_codes):
                return 0
        # finally replaced the rest dead code
        return self.replace(batch_samples, replace_mask=expired_codes)

    def replace(self, batch_samples, replace_mask):
        all_samples = [torch.empty_like(batch_samples) for _ in range(dist.get_world_size())]
        dist.all_gather(all_samples, batch_samples)
        all_samples = torch.stack(all_samples)
        all_samples = rearrange(all_samples, '... d -> (...) d')
        if replace_mask.sum() > len(all_samples):
            all_samples = [all_samples for _ in range(int(replace_mask.sum() / len(all_samples)) + 1)]
            all_samples = torch.cat(all_samples, dim=0)
        self.embedding.weight.data[replace_mask] = \
            all_samples[:replace_mask.sum()].detach().to(dtype=self.embedding.weight.data.dtype)
        self.cluster_size.data[replace_mask] = self.reset_cluster_size
        # replace_indices = replace_mask.nonzero(as_tuple=True)[0]
        # assigned_mask = replace_indices.tensor_split(dist.get_world_size())[dist.get_rank()]
        # if len(assigned_mask) > len(batch_samples):
        #     batch_samples = [batch_samples for _ in range(int(len(assigned_mask) / len(batch_samples)) + 1)]
        #     batch_samples = torch.cat(batch_samples, dim=0)
        # self.embedding.weight.data[assigned_mask] = \
        #     batch_samples[:len(assigned_mask)].detach().to(dtype=self.embedding.weight.data.dtype)
        # self.cluster_size.data[assigned_mask] = self.reset_cluster_size
        return replace_mask.sum().item()

def l2norm(t):
    return F.normalize(t, p=2, dim=-1)

def laplace_smoothing(x, n_categories, eps=1e-5, dim=-1):
    denom = x.sum(dim=dim, keepdim=True)
    return (x + eps) / (denom + n_categories * eps)

def batched_embedding(indices, embeds):
    batch, dim = indices.shape[0], embeds.shape[-1]
    indices = repeat(indices, "b n -> b n d", d=dim)
    embeds = repeat(embeds, "c d -> b c d", b=batch)
    return embeds.gather(2, indices)


class EMAQuantizer(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    def __init__(
        self,
        n_e,
        e_dim,
        output_dim,
        commitment_weight=1.0,  # in this case, only ||x - quantized.detach|| loss is applied
        diversity_weight=1.0,
        dead_code_threshold=0.05,
        decay=0.99,
        latent_dim=None,
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.decay = decay
        self.w_commit = commitment_weight
        self.w_diversity = diversity_weight
        self.dead_code_threshold = dead_code_threshold
        self.project_in = nn.Linear(latent_dim, e_dim) if latent_dim != e_dim else nn.Identity()
        self.project_out = nn.Linear(e_dim, output_dim) if output_dim != e_dim else nn.Identity()
        embed = torch.zeros(self.n_e, self.e_dim)
        embed.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        embed = l2norm(embed)
        self.register_buffer('embed', embed)
        self.register_buffer('embed_avg', embed.clone())
        self.register_buffer("cluster_size", torch.ones(self.n_e))      # avg cluster size for each code among n_e codes
        self.reset_cluster_size = dead_code_threshold * 1.2
        self.world_size = dist.get_world_size()
        self.total_codes = 0
        self.initted = False
        self.ratio = 0.0

    def init_quantizer(self, x):
        if not self.initted:
            self.total_codes = len(x) * self.world_size
            self.ratio = self.total_codes / self.n_e
            self.cluster_size.data *= self.ratio
            self.initted = True

    def forward(self, x):
        # 1. transform input
        x_proj = self.project_in(x)
        x = x_proj.view(-1, self.e_dim)
        x = l2norm(x)
        self.init_quantizer(x)

        # 2. quantize
        embed = self.embed.detach()
        d = einsum('n d, c d -> n c', x, embed)
        embed_ind = torch.argmin(d, dim=1)
        embed_onehot = F.one_hot(embed_ind, num_classes=self.n_e).float()
        # z_q = embed.index_select(dim=0, index=embed_ind)
        # z_q2 = einsum('n c, c d -> n d', embed_onehot, embed)
        # z_q_cpu = embed.cpu().index_select(dim=0, index=embed_ind.cpu())
        # z_q_cpu2 = einsum('n c, c d -> n d', embed_onehot.cpu(), embed.cpu())
        if self.training:
            z_q = einsum('n c, c d -> n d', embed_onehot, embed)
        else:
            z_q = embed.index_select(dim=0, index=embed_ind)
        # 3. update codebook with ema
        bins = embed_onehot.sum(dim=0)
        dist.all_reduce(bins)
        ema_inplace(self.cluster_size.data, bins, self.decay)
        embed_sum = einsum('n d, n c -> c d', x, embed_onehot).contiguous()
        dist.all_reduce(embed_sum)
        ema_inplace(self.embed_avg.data, embed_sum, self.decay)
        cluster_size = laplace_smoothing(
            self.cluster_size, self.n_e,
        ) * self.cluster_size.sum(dim = -1, keepdim = True)
        embed_normalized = self.embed_avg / rearrange(cluster_size, '... -> ... 1')
        self.embed.data.copy_(embed_normalized)

        # 3. diversity
        scaled_distances = d * 10.0
        entropy_to_max, entropy_to_min = calc_entropy(
            scaled_distances.flatten(end_dim=-2)
        )
        # diversity_loss = entropy_to_min - entropy_to_max
        diversity_loss = -entropy_to_max
        diversity_entropy = entropy_to_max.detach()
        deterministic_entropy = entropy_to_min.detach()

        # 4. commitment
        commit_loss = torch.mean((z_q.detach() - x) ** 2)
        loss = self.w_commit * commit_loss + self.w_diversity * diversity_loss

        # 5. straight-through
        z_q = x + (z_q - x).detach()
        z_q = z_q.view(x_proj.shape)
        z_q = self.project_out(z_q)

        # 6. perplexity & active codes
        num_reactivate = self.expire_codes(x, None)
        probs = bins / self.total_codes
        perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10)))
        num_active_codes = (self.cluster_size / self.ratio > 0.2).sum()

        log_dict = {
            'num_active_codes': num_active_codes.item(),
            'num_reactivate': num_reactivate,
            'perplexity': perplexity.item(),
            'commitment_loss': commit_loss.item(),
            'deterministic_entropy': deterministic_entropy.item(),
            'diversity_entropy': diversity_entropy.item(),
        }
        
        # return z_q, loss, (perplexity, self.cluster_size, min_encoding_indices, diversity_entropy, deterministic_entropy, probs, total_codes)
        return z_q, embed_ind, loss, log_dict
    
    def expire_codes(self, batch_samples, matched_indices):
        # first check if already no dead code
        expired_codes = self.cluster_size < self.dead_code_threshold * self.ratio
        if not torch.any(expired_codes):
            return 0
        # then filter off any code that is matched in current batch
        if matched_indices is not None:
            all_indices = [torch.empty_like(matched_indices) for _ in range(dist.get_world_size())]
            dist.all_gather(all_indices, matched_indices)
            all_indices = torch.stack(all_indices)
            expired_codes[all_indices.unique()] = False
            if not torch.any(expired_codes):
                return 0
        # finally replaced the rest dead code
        return self.replace(batch_samples, replace_mask=expired_codes)

    def replace(self, batch_samples, replace_mask):
        all_samples = [torch.empty_like(batch_samples) for _ in range(dist.get_world_size())]
        dist.all_gather(all_samples, batch_samples)
        all_samples = torch.stack(all_samples)
        all_samples = rearrange(all_samples, '... d -> (...) d')
        if replace_mask.sum() > len(all_samples):
            all_samples = [all_samples for _ in range(int(replace_mask.sum() / len(all_samples)) + 1)]
            all_samples = torch.cat(all_samples, dim=0)
        sampled = all_samples[:replace_mask.sum()].detach().to(dtype=self.embed.data.dtype)
        self.embed.data[replace_mask] = sampled
        self.embed_avg.data[replace_mask] = sampled * self.reset_cluster_size * self.ratio
        self.cluster_size.data[replace_mask] = self.reset_cluster_size * self.ratio
        return replace_mask.sum().item()