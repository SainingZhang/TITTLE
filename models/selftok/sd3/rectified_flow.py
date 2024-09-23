import torch
import math
from einops import rearrange

TRADITION = 1000

def append_to_shape(t, x_shape):
    return t.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def sum_flat(tensor):
    """
    Take the sum over all non-batch dimensions.
    """
    return tensor.sum(dim=list(range(1, len(tensor.shape))))

class RectifiedFlow(torch.nn.Module):
    def __init__(self, num_timesteps=100, schedule="log_norm", parameterization='x0', shift=1.0, m=0, s=1, force_recon=False, device='cuda'):
        super().__init__()
        self.schedule = schedule
        self.parameterization = parameterization
        self.m = m
        self.s = s
        self.shift = shift
        self.num_timesteps = num_timesteps
        self.make_schedule()
        self.device = device
        self.force_recon = force_recon
        if self.schedule == "cosmap":
            self.t_trajectory = self.schedule_by_cosmap
        elif self.schedule == "log_norm":
            self.t_trajectory = self.schedule_by_logNorm
        elif self.schedule == "heavy_tails":
            self.t_trajectory = self.schedule_by_heavyTails
        elif self.schedule == "uniform":
            self.t_trajectory = self.schedule_by_uniform
        else:
            raise NotImplementedError
            
    def make_schedule(self, schedule="uniform", args=None):
        base_t = torch.linspace(1, 0, self.num_timesteps+1).cuda()
        if schedule == "uniform":
            scheduled_t = base_t
        elif schedule == "shift":
            scheduled_t =self.shift * base_t / (1 + (self.shift - 1) * base_t)
        elif schedule == "align_resolution":
            e = torch.e
            res1, s1, res2, s2, target_res, c = args
            m = (s1 -s2) / (res1 - res2) * (target_res - res1) + s1
            scheduled_t = e ** m / (e ** m + (1/base_t - 1) ** c)
        self.register_buffer("timestep_map", scheduled_t[:-1] * TRADITION)
        self.register_buffer("scheduled_t", scheduled_t[:-1])
        self.register_buffer("scheduled_t_prev", scheduled_t[1:])
        self.register_buffer("one_minus_scheduled_t", 1-scheduled_t[:-1])

    def sample_t(self, bs):
        t = self.t_trajectory(torch.rand(bs))
        return t

    def q_sample(self, x, t, noise=None):
        t = append_to_shape(t,x.shape)
        if noise is None:
            noise = torch.randn_like(x)
        return t * noise + (1 - t) * x

    def get_target(self, x, noise):
        target = noise - x
        return target
    
    def sigma(self, timestep: torch.Tensor):
        timestep = timestep / TRADITION
        if self.shift == 1.0:
            return timestep
        return self.shift * timestep / (1 + (self.shift - 1) * timestep)

    def schedule_by_logNorm(self, t):
        proj = torch.distributions.normal.Normal(self.m, self.s)
        t = torch.log(t / (1 - t))
        t = proj.cdf(t)
        return t

    def schedule_by_cosmap(self, t):
        t = 1. - 1 / (torch.tan(torch.pi/2 * t) + 1)
        return t

    def schedule_by_heavyTails(self, t):
        t = 1 - t - self.s * (torch.cos(torch.pi/2 * t) - 1 + t)
        return t
    
    def schedule_by_uniform(self, t):
        return t
    
    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None, recon_ratio=None):
        """
        Compute training losses for a single timestep.
        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        #sig = self.sigma(t)
        x_t = self.q_sample(x_start, t, noise=noise)
        
        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "velocity":
            target = noise - x_start
        else:
            raise NotImplementedError()
            
        terms = {}

       
        v = model(x_t, t, **model_kwargs)
        v_gt = noise - x_start

        if self.force_recon:
            assert self.parameterization == 'velocity'
            # model_output = noise - model_output
            model_output = x_t - rearrange(t, 'b -> b 1 1 1') * v
            target = x_start
        else:
            model_output = v

        if "loss_mask" in model_kwargs:
            loss_mask = model_kwargs["loss_mask"].unsqueeze(1).repeat(1, target.shape[1], 1, 1)
            mse_loss = (target - model_output) ** 2
            terms["loss"] = sum_flat(mse_loss * loss_mask.float()) / sum_flat(loss_mask)
        else:
            terms["loss"] = mean_flat((target - model_output) ** 2)
            
        terms["mse"] = mean_flat((target - model_output) ** 2)
        if recon_ratio != 1.0 and self.force_recon:
            terms["loss"] = recon_ratio*terms["loss"] + (1-recon_ratio)*mean_flat((v_gt - v) ** 2)
        return terms

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        start_t=None,
        model_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        x_0=None,
        encoder=None,
        diti=None,
        dit=None,
        ori_hidden_states=None,
        cond_vary=False,
        device=None,
        **kwargs,
    ):
        batch_size = shape[0]
        if device is None:
            device = next(model.parameters()).device
 
        if noise is None:
            img = torch.randn(*shape, device=device)
        else:
            img = noise
 
        #for i in indices:
        for i, step in enumerate(self.scheduled_t):
            t = torch.tensor([step] * batch_size, device=device)  # step：1~0
            with torch.no_grad():
                if cond_vary:
                    t_mapped = torch.tensor([self.timestep_map[i]]*batch_size, device=device).long()
                    #print('IN t_mapped', t_mapped)
                    k = diti.t_to_idx.to(device)[t_mapped-1]
                    
                    encoder_hidden_states, _, mask, _, _ = encoder(x=x_0, hidden_states=ori_hidden_states, d=k)
                    model_kwargs = dict(
                        encoder_hidden_states=encoder_hidden_states,
                        mask=mask
                    )
                    if encoder_hidden_states.sum() == 0 and dit is not None:
                        print("No condition is given...")
                        model_kwargs = {
                            'y': torch.tensor([1000] * len(x_0)).to(x_0.device)
                        }
                        model_to_use = dit
                    else:
                        model_to_use = model
                else:
                    model_to_use = model
 
                img, pred_x0 = self.sample_one_step(
                    model_to_use,
                    img,
                    t,
                    index=i,
                    model_kwargs=model_kwargs,
                    cfg_scale=unconditional_guidance_scale,
                    uc=unconditional_conditioning,
                    **kwargs,
                )

        return img
 
    def sample_one_step(
        self,
        model,
        x,
        t,
        index,
        model_kwargs=None,
        cfg_scale=1.0,
        uc=None,
        **kwargs,
    ):
        if model_kwargs is None:
            model_kwargs = {}
        b, *_, device = *x.shape, x.device
        a_t = torch.full((b, 1, 1, 1), self.scheduled_t[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), self.scheduled_t_prev[index], device=device)
        if uc is None or cfg_scale == 1.0:
            out = model(x, t, **model_kwargs)
        else:
            out_uncond, out = model(torch.cat([x] * 2), torch.cat([t] * 2), torch.cat([uc, model_kwargs['encoder_hidden_states']]), mask = model_kwargs['mask']).chunk(2)
            out = out_uncond + cfg_scale * (out - out_uncond)
            
        img, pred_x0 = self.base_step(
            x, out, a_t=a_t, a_prev=a_prev, **kwargs
        )
        return img, pred_x0
    
    def base_step(self, x, v, a_t, a_prev,**kwargs):
        # Base sampler uses Euler numerical integrator.
        x_prev, pred_x0 = self.euler_step(x, v, a_t, a_prev)
        return x_prev, pred_x0
 
    def euler_step(self, x, v, a_t, a_prev, **kwargs):
        if self.parameterization == "velocity":
            x_prev = x - (a_t - a_prev) * v
            pred_x0 = x - a_t * v
        elif self.parameterization == "x0":
            #x_prev = x - (a_t - a_prev) * (noise-v)
            x_prev = v + a_prev * (x - v) / a_t
            pred_x0 = v
            
        return x_prev, pred_x0
        
    

if __name__ == '__main__':
    import ipdb
    import matplotlib.pyplot as plt
    sampler1 = RectifiedFlow()
    print(f"cosmap: {sampler1.sample_t(100000).max()}, {sampler1.sample_t(100000).min()}")
    sampler2 = RectifiedFlow("log_norm")
    print(f"log_norm: {sampler1.sample_t(100000).max()}, {sampler1.sample_t(100000).min()}")
    sampler3 = RectifiedFlow("heavy_tails")
    print(f"heavy_tails: {sampler1.sample_t(100000).max()}, {sampler1.sample_t(100000).min()}")

    t = torch.arange(10000)/10000
    y1 = sampler1.schedule_by_cosmap(t)
    y2 = sampler1.schedule_by_logNorm(t)
    y3 = sampler1.schedule_by_heavyTails(t)

    plt.figure()
    fig, axs = plt.subplots(3)
    axs[0].plot(t, y1)
    axs[1].plot(t, y2)
    axs[2].plot(t, y3)
    fig.savefig("/cache/schedules.png")
    ipdb.set_trace()