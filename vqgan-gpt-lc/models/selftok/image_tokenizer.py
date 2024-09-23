import os
import yaml
from copy import deepcopy
from collections import OrderedDict
import random
from .model_zoo import Enc_models, DiT_models
from .models_ours import Encoder
import torch
from mimogpt.utils import hf_logger
from torch import nn
import numpy as np
from diffusers.models import AutoencoderKL
from mimogpt.models.selftok.diffusion import create_diffusion
from mimogpt.models.selftok.diti_utils import DiTi
from mimogpt.models.selftok.sd3.rectified_flow import RectifiedFlow
import copy
MAX_LATENT_SIZE = 384


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def norm_ip(img, low, high):
    img.clamp_(min=low, max=high)
    img.sub_(low).div_(max(high - low, 1e-5))


def ori_reconstruct(
    num_steps,
    t,
    model,
    noise=None,
    x0=None,
    x0_e=None,
    y=None,
    diti=None,
    encoder=None,
    add_noise_mode=2,
    cond_vary=False,
    ddim=False,
):
    diffusion = create_diffusion(str(num_steps))
    N = x0.shape[0]
    device = x0.device

    with torch.no_grad():
        if diti is None:
            # When using cfg, need to duplicate input. Recon is closer to something in the class.
            # model_kwargs = {'y': y, 'cfg_scale': 4.0}
            # model = model.forward_with_cfg
            model_kwargs = {"y": y}
        else:
            t_mapped = torch.tensor([diffusion.timestep_map[t]] * N, device=device)
            k = diti.t_to_idx.to(device)[t_mapped]
            encoder_hidden_states, ori_hidden_states, mask, _, _ = encoder(x0_e, d=k)

            # get noise
            if add_noise_mode == 2:
                model_kwargs1 = dict(encoder_hidden_states=encoder_hidden_states, mask=mask)
                x_t = diffusion.ddim_reverse_sample_loop(
                    model.forward,
                    x0.shape,
                    x0,
                    clip_denoised=False,
                    model_kwargs=model_kwargs1,
                    progress=True,
                    device=device,
                    start_t=t,
                )
            elif add_noise_mode == 1:
                # gt + noise
                x_t = diffusion.q_sample(x0, torch.tensor([t] * N, device=device), noise)

            else:
                # 随机噪音
                # x_t = torch.randn(x0.shape, device=device)
                x_t = noise

            model_kwargs = dict(encoder_hidden_states=encoder_hidden_states, mask=mask)
        pred_x_0 = diffusion.p_sample_loop(
            model.forward,
            x_t.shape,
            x_t,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
            device=device,
            start_t=t,
            ddim=ddim,
            cond_vary=cond_vary,
            diti=diti,
            encoder=encoder,
            # x_0=x0_e,
            ori_hidden_states=ori_hidden_states,
        )
    return {"x_t": x_t, "pred_x_0": pred_x_0}


def reconstruct(
    num_steps,
    t,
    model,
    noise=None,
    indices=None,
    y=None,
    diti=None,
    encoder: Encoder = None,
    cond_vary=False,
    hidden=None,
):
    diffusion = create_diffusion(str(num_steps))
    N = indices.shape[0]
    device = indices.device
    print(cond_vary)
    with torch.no_grad():
        if diti is None:
            model_kwargs = {"y": y}
        else:
            t_mapped = torch.tensor([diffusion.timestep_map[t]] * N, device=device)
            k = diti.t_to_idx.to(device)[t_mapped]

            if hidden is None:
                encoder_hidden_states, ori_hidden_states, mask = encoder.encode_indices(indices, d=k)
            else:
                encoder_hidden_states, ori_hidden_states, mask = encoder.encode_indices(
                    indices, hidden_states=hidden, d=k
                )

            x_t = noise

            model_kwargs = dict(encoder_hidden_states=encoder_hidden_states, mask=mask)
        pred_x_0 = diffusion.p_sample_loop(
            model.forward,
            x_t.shape,
            x_t,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
            device=device,
            start_t=t,
            ddim=True,
            cond_vary=cond_vary,
            diti=diti,
            encoder=encoder,
            x_indices=indices,
            ori_hidden_states=ori_hidden_states,
        )
    return {"x_t": x_t, "pred_x_0": pred_x_0}


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def reconstruct(num_steps, t, model, noise=None, x0=None, x0_e=None, y=None, diti=None, encoder=None, add_noise_mode=2, cond_vary=False, ddim=False, dit=None, remove_range=None):
    diffusion = create_diffusion(str(num_steps))
    N = x0.shape[0]
    device = x0.device
    with torch.no_grad():
        if diti is None:
            # When using cfg, need to duplicate input. Recon is closer to something in the class.
            # model_kwargs = {'y': y, 'cfg_scale': 4.0}
            # model = model.forward_with_cfg
            model_kwargs = {'y': y}
        else:
            t_mapped = torch.tensor([diffusion.timestep_map[t]]*N, device=device)
            k = diti.t_to_idx.to(device)[t_mapped]
            encoder_hidden_states, ori_hidden_states, mask, _, _ = encoder(x0_e, d=k)
            # get noise
            if add_noise_mode==2:
                model_kwargs1 = dict(
                    encoder_hidden_states=encoder_hidden_states,
                    mask=mask
                )
                x_t = diffusion.ddim_reverse_sample_loop(model.forward, x0.shape, x0, clip_denoised=False, 
                                                         model_kwargs=model_kwargs1, progress=True, device=device, 
                                                         start_t=t)
            elif add_noise_mode==1:
                # gt + noise
                x_t = diffusion.q_sample(x0, torch.tensor([t]*N, device=device), noise)

            else:
                x_t = noise
            
            model_kwargs = dict(
                encoder_hidden_states=encoder_hidden_states,
                mask=mask
            )
        pred_x_0 = diffusion.p_sample_loop(
            model.forward, x_t.shape, x_t, clip_denoised=False, 
            model_kwargs=model_kwargs, progress=True, device=device,
            start_t=t, ddim=ddim, remove_range=remove_range,
            cond_vary=cond_vary, diti=diti, encoder=encoder, x_0 = x0_e, ori_hidden_states=ori_hidden_states, dit=dit
        )
    return {"x_t": x_t, "pred_x_0": pred_x_0}

def load_state(model, prefix, state_dict, excludes=[]):
    model_dict = model.state_dict()  # 当前网络结构
    pretrained_dict = {k.replace(prefix,''): v for k, v in state_dict.items() if k.replace(prefix,'') in model_dict}  # 预训练模型中可用的weight
    dict_t = copy.deepcopy(pretrained_dict)
    for key, weight in dict_t.items():
        if key in model_dict and model_dict[key].shape != dict_t[key].shape:
            if 'final_layer' in key:
                pretrained_dict[key] = torch.cat(
                                        (
                                            pretrained_dict[key],
                                            model_dict[key][64:].to(pretrained_dict[key].device),
                                        ),
                                        dim=0,
                                    )
            else:
                pretrained_dict.pop(key)
        elif any(item in key for item in excludes):
            pretrained_dict.pop(key)
   
    m, u = model.load_state_dict(pretrained_dict, strict=False)
    if len(m) > 0:
        print("model_ad missing keys:")
        print(m)
    if len(u) > 0:
        print("model_ad unexpected keys:")
        print(u)


class ImageTokenizer(nn.Module):
    def __init__(
        self,
        init_with_pretrained,
        train_filter,
        image_size,
        k,
        stages,
        k_per_stage,
        encoder_yaml,
        enc,
        encoder_hidden_size,
        n_e,
        share_codebook,
        reverse_code,
        code_dim,
        w_diversity,
        w_quan,
        quan_temp,
        quantize_kmeans_init,
        ema_decay,
        dead_code_threshold,
        use_cosine_sim,
        pre_norm,
        no_post_norm,
        lfq,
        k_embed,
        model,
        cross_modulate,
        pretrained_dit_path,
        random_t_weight,
        pred_eps,
        train_encoder_only,
        pdae,
        w_cm,
        noise_schedule_config=None,
        ori_resume_from_ckpt=None,
        ema_update = True,
        freeze_filter="",
        gradient_checkpointing=False,
        learn_sigma=True,
        in_channels=4,
        sd3_cond_pooling=None,
        class_dropout_prob=0.0,
        enable_enc_variable_size=False,    # to enable variable image size for encoder; max size=MAX_LATENT_SIZE after downsampling by vae
        xavier_init=False,
        smart_react=False,
        **kwargs,
    ):
        super().__init__()
        self.random_t_weight = random_t_weight
        self.pred_eps = pred_eps
        self.train_encoder_only = train_encoder_only
        self.pdae = pdae
        self.w_cm = w_cm
        # 253-272
        # Create model:
        predict_xstart = False if init_with_pretrained else True
        train_filter = train_filter.split('+') if train_filter != 'all' else None
        if train_encoder_only:
            train_filter = []       # do not train anything
        freeze_filter = freeze_filter.split('+') if freeze_filter != '' else []
        self.model_name = model
        if self.model_name == 'MMDiT_XL':
            self.diffusion = RectifiedFlow(**noise_schedule_config)
            self.recon_ratio = 1.0
        else:
            self.diffusion = create_diffusion(
                timestep_respacing="", predict_xstart=predict_xstart, learn_sigma=True, use_kl=False
            )  # default: 1000 steps, linear noise schedule, pred xstart!

        assert image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
        
        self.diti = DiTi(1000, k, stages, k_per_stage)
        
        # Load additional encoder configs from yaml
        if encoder_yaml:
            stream = open(f"configs/{encoder_yaml}", 'r')
            encoder_cfg = yaml.safe_load(stream)
            n_e_cfg = encoder_cfg['n-e']
            hf_logger.info(f"loaded n-e config: {','.join([str(n) for n in n_e_cfg])}")
        else:
            n_e_cfg = None
        
        latent_size = image_size // 8
        
        if 'Qformer' in enc:
            if enable_enc_variable_size:
                kwargs['pos_embed_max_size'] = MAX_LATENT_SIZE // int(enc[-1])
            kwargs['xavier_init'] = xavier_init
        else:
            if enable_enc_variable_size:
                assert False, "Other encoder does not support variable input size."

        self.encoder = Enc_models[enc](
            K=k,
            input_size=latent_size,
            encoder_hidden_size=encoder_hidden_size,
            n_e=n_e,
            n_e_cfg=n_e_cfg, 
            share_codebook=share_codebook,
            reverse_code=reverse_code,
            code_dim=code_dim,
            w_diversity=w_diversity,
            w_commit=w_quan,
            quantizer_temp=quan_temp,
            quantize_kmeans_init = quantize_kmeans_init, 
            decay = ema_decay,
            dead_code_threshold=dead_code_threshold,
            use_cosine_sim=use_cosine_sim,
            pre_norm=pre_norm,
            post_norm=not no_post_norm,
            lfq=lfq,
            k_embed=k_embed,
            ema_update=ema_update,
            in_channels=in_channels,
            gradient_checkpointing=gradient_checkpointing,
            smart_react=smart_react,
            **kwargs
        )
        params = dict(
            input_size=latent_size, learn_sigma=learn_sigma,
            cross_modulate=cross_modulate,
            encoder_hidden_size=encoder_hidden_size,
            train_filter=train_filter,
            freeze_filter=freeze_filter,
            in_channels=in_channels,
            gradient_checkpointing=gradient_checkpointing,
            K=k,
        )
        if self.model_name == 'MMDiT_XL':
            params["sd3_cond_pooling"] = sd3_cond_pooling
            params["class_dropout_prob"] = class_dropout_prob
        self.model = DiT_models[model](**params)
        # self.sd3_cond_pooling = sd3_cond_pooling
        
        # 等模型传到s3再解除注释
        if init_with_pretrained:
            self.pretrained_dit_path = pretrained_dit_path
            self.load_pretrain_teacher(pretrained_dit_path)
        else:
            assert train_filter is None

        self.model.freeze()  # keep only params matching train_filter
        
        if ori_resume_from_ckpt:
            state_dict = torch.load(
                ori_resume_from_ckpt,
                map_location='cpu'
            )
            hf_logger.info(f"model loading status: {self.model.load_state_dict(state_dict['model'])}")
            hf_logger.info(f"encoder loading status: {self.encoder.load_state_dict(state_dict['encoder'])}")

        hf_logger.info(f"ScDiT Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        hf_logger.info(f"Encoder Parameters: {sum(p.numel() for p in self.encoder.parameters()):,}")

        # recon prev timestep weight
        self.T = self.diffusion.num_timesteps
        w_prev = np.array([float(t+1.)/self.T for t in range(self.T)])
        self.w_prev = w_prev / w_prev.sum()

        # objective weight
    
    def load_pretrain_teacher(self, teacher_path):
        hf_logger.info(f'init_from {teacher_path}')
        state_dict = torch.load(
            teacher_path,
            map_location='cpu'
        )
        if self.model_name == 'MMDiT_XL':
            # load_state(self.model, 'model.diffusion_model.', state_dict, excludes=['y_embedder', 'context_embedder'])
            load_state(self.model, 'model.diffusion_model.', state_dict)
        else:
            self.model.load_state_dict(state_dict, strict=False)

    def set_train(self):
        self.model.train()
        self.encoder.train()
        # self.ema.eval()

    def set_eval(self):
        self.model.eval()
        self.encoder.eval()
        # self.ema.eval()

    def get_log(self, log_dict):
        perplexity = log_dict["perplexity"]
        n_active = log_dict["num_active_codes"]
        n_reactive = log_dict["num_reactivate"]
        commitment_loss = log_dict["commitment_loss"]
        diversity_entropy = log_dict["diversity_entropy"]
        deterministic_entropy = log_dict["deterministic_entropy"]
        perplexity_list = log_dict["perplexity_list"]
        deter_list = log_dict["deterministic_list"]
        return perplexity, n_active, n_reactive, commitment_loss, diversity_entropy, deterministic_entropy, perplexity_list, deter_list

    def forward(self, x, ema=None, full_tokens=False):
        device = x.device
        if self.model_name == 'MMDiT_XL':
            t = self.diffusion.sample_t(x.shape[0]).cuda()
            T = torch.floor(t * 1000).int().clamp(0, 999)
            k_batch = self.diti.t_to_idx.to(device)[T]
        else:
            t = torch.randint(0, self.diffusion.num_timesteps, (x.shape[0],), device=device)
            k_batch = self.diti.t_to_idx.to(device)[t]
        k_batch = self.diti.t_to_idx.to(device)[torch.zeros_like(t).long()+999] if full_tokens else k_batch
        if not self.encoder.training:
            with torch.no_grad():
                encoder_hidden_states, ori_hidden_states, attn_mask, quan_loss, log_dict = self.encoder(x=x, d=k_batch)
        else:
            encoder_hidden_states, ori_hidden_states, attn_mask, quan_loss, log_dict = self.encoder(x=x, d=k_batch)

        perplexity, n_active, n_reactive, commitment_loss, diversity_entropy, deterministic_entropy, perplexity_list, deter_list = self.get_log(log_dict)
        # store perplexity list
        # if len(perplexity_list) > 1:
        #     running_perplexity_list = list(map(add, running_perplexity_list, perplexity_list))
        #     running_deter_list = list(map(add, running_deter_list, deter_list))
        noise = torch.randn_like(x)
        model_kwargs = dict(
            encoder_hidden_states=encoder_hidden_states,
            mask=attn_mask,
        )
        force_recon_loss = True if not self.random_t_weight else random.uniform(0, 1) > 0.5
        if self.pred_eps:
            force_recon_loss = False

        # dm_model = ema if self.train_encoder_only else self.model
        dm_model = self.model
        if self.model_name == 'MMDiT_XL':
            # if self.sd3_cond_pooling == 'last':
            #     model_kwargs["y"] = encoder_hidden_states[torch.arange(x.shape[0]), k_batch, :]
            # elif self.sd3_cond_pooling == 'mean':
            #     model_kwargs["y"] = encoder_hidden_states.sum(dim=1) / attn_mask.sum(dim=-1).unsqueeze(1)
            loss_dict = self.diffusion.training_losses(
                dm_model, x, t, model_kwargs, noise=noise, recon_ratio=self.recon_ratio
            )
            batch_mse = loss_dict["loss"].mean()
        else:
            loss_dict = self.diffusion.training_losses(
                dm_model, x, t, model_kwargs, noise=noise,
                force_recon_loss=force_recon_loss, weighting=self.pdae
            )
            per_sample_mse_loss = mean_flat((x - loss_dict['pred_xstart']) ** 2)
            batch_mse = per_sample_mse_loss.mean()

        dm_loss = loss_dict["loss"].mean()

        if self.w_cm != 0.0:
            raise NotImplementedError()
        else:
            cm_loss = torch.tensor(0.0).to(device)
            x0_p_dit = None

        loss = dm_loss + self.w_cm * cm_loss + quan_loss
        log_dict = {
            "cm": cm_loss.item(),
            "loss": loss.item(),
            "mse": batch_mse.item(),
            "avg_quan": commitment_loss,
            "diversity":diversity_entropy,
            "deterministic": deterministic_entropy,
            "perplexity": perplexity,
            "n_active":n_active,
            "n_reactive":n_reactive,
            "perplexity_list": perplexity_list,
            "deter_list": deter_list,
        }
        return loss, log_dict
        
    def rec(self, x_0):
        noise = torch.randn_like(x_0, device=x_0.device)
        with torch.no_grad():
            recon = ori_reconstruct(
                100,
                99,
                self.model,
                noise,
                x_0,
                x_0,
                diti=self.diti,
                encoder=self.encoder,
                add_noise_mode=0,
                cond_vary=False,
            )["pred_x_0"]
            img_recon = self.vae.decode(recon / 0.18215).sample
            norm_ip(img_recon, -1, 1)

        return img_recon

    def get_vae_latent(self, x):
        return self.vae.encode(x).latent_dist.sample().mul_(0.18215)

    def encode(self, x, pre_encode=True):
        # token_num_list = get_token_num(self.k, self.n_e_list, class_num=1000, share_embed=True)
        with torch.no_grad():
            if not pre_encode:
                x0 = self.vae.encode(x).latent_dist.sample().mul_(0.18215)
            else:
                # x0 = x.squeeze(dim=1)
                x0 = x
            _, indices, encoder_hidden_states = self.encoder(x0, d=None)

        return encoder_hidden_states, indices

    def decode(self, indices, shape, hidden=None):
        noise = torch.randn(shape, device=indices.device)
        with torch.no_grad():
            recon = reconstruct(
                100,
                99,
                self.model,
                noise,
                indices,
                diti=self.diti,
                encoder=self.encoder,
                cond_vary=self.cond_vary,
                hidden=hidden,
            )["pred_x_0"]
            img_recon = self.vae.decode(recon / 0.18215).sample
            norm_ip(img_recon, -1, 1)

        return img_recon
