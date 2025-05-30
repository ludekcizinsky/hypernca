import random
from ast import mod
import numpy as np
import torch
import scipy.stats
from diffusers.utils.torch_utils import randn_tensor
import torch.nn as nn
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
import math
from typing import Dict, List, Optional, Union
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import (
    KarrasDiffusionSchedulers,
    SchedulerMixin,
)
import torch.nn.functional as F
from tqdm import tqdm

def get_diffusion(cfg):
    
    if cfg.diffusion.type == 'EDM':
        scheduler = EDMScheduler(
            P_mean=cfg.diffusion.p_mean,
            P_std=cfg.diffusion.p_std,
            sigma_data=cfg.diffusion.sigma_data,
        )

        num_weight_tokens = cfg.model.num_weight_tokens
        weight_dim = cfg.model.weight_dim
        diffusion = EDMPipeline(
            scheduler=scheduler,
            guidance_scale=cfg.diffusion.guidance_scale,
            conditional=cfg.model.conditioning,
            use_cfg=cfg.model.use_cfg,
            parameter_size=(num_weight_tokens, weight_dim),
        )

    elif cfg.diffusion.type == 'DDIM':
        scheduler = DDIMScheduler(
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule='squaredcos_cap_v2',
            num_train_timesteps=1000,
            prediction_type="sample",
            trained_betas=None,
            clip_sample=True,
            clip_sample_range=15.0,
            thresholding=False,
        )

        diffusion = DDIMPipeline(
            scheduler=scheduler,
            guidance_scale=cfg.diffusion.guidance_scale,
            conditional=cfg.model.conditioning,
            use_cfg=cfg.model.use_cfg,
            parameter_size=(cfg.model.num_weight_tokens, ),
        )

    else:
        raise ValueError(f"Unknown diffusion type: {cfg.diffusion.type}")

    return diffusion

def randn_like(x, generator):
    return randn_tensor(x.shape, generator=generator, device=x.device, dtype=x.dtype)

class EDMScheduler:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=1.0, bins=25, **kwargs):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

        self.dist = scipy.stats.norm(loc=P_mean, scale=P_std)
        self.bins = bins
        self.sigma_bins = np.exp(self.dist.ppf(np.linspace(0.005, 0.995, self.bins)))

    def sample_noise(self, x):
        rnd_normal = torch.randn([x.shape[0], 1, 1], device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        loss_weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        noise = torch.randn_like(x) * sigma

        log_sigma = torch.log(sigma).squeeze().detach().cpu().numpy()
        bins = (self.dist.cdf(log_sigma) * self.bins).astype(int)

        return noise, sigma, loss_weight, bins

    def get_c(self, sigma):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        return c_skip, c_out, c_in, c_noise

    def sample_discrete_noise_levels(self, x, noise_level, max_noise_levels):
        log_sigma_a = self.dist.ppf(self.dist.cdf(np.log(self.sigma_data / 50.0)))
        log_sigma_b = self.dist.ppf(self.dist.cdf(np.log(self.sigma_data * 50.0)))
        sigma = np.exp(log_sigma_a + (log_sigma_b - log_sigma_a) * noise_level / max_noise_levels)
        noise = torch.randn_like(x) * sigma
        return noise, sigma


class EDMPipeline:
    def __init__(self, scheduler, guidance_scale,conditional,use_cfg, parameter_size):
        super().__init__()
        assert guidance_scale >= 1.0, "Guidance scale should be greather than or equal to 1.0"
        self.scheduler = scheduler
        self.parameter_size = parameter_size
        self.guidance_scale = guidance_scale
        self.use_cfg = use_cfg

        self.cond_generation = conditional


    def forward_precond(self, net, x, sigma, cond):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32)
        c_skip, c_out, c_in, c_noise = self.scheduler.get_c(sigma)

        F_x = net(c_in * x, c_noise.flatten(), cond)
        D_x = c_skip * x + c_out * F_x

        return D_x
    
    def forward_precond_guided(self, net, x, sigma, cond):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32)
        c_skip, c_out, c_in, c_noise = self.scheduler.get_c(sigma)

        F_x = net(c_in * x, c_noise.flatten(), cond)
        D_x = c_skip * x + c_out * F_x

        if self.cond_generation and self.guidance_scale!= 1.0 and cond is not None and self.use_cfg:
            F_cond = net(c_in * x, c_noise.flatten(), None)
            D_uncond = c_skip * x + c_out * F_cond
            D_x = D_uncond + self.guidance_scale * (D_x - D_uncond)
        return D_x
    
    def get_loss(self, model, x, cond, return_aux=False):
        noise, sigma, loss_weight, bins = self.scheduler.sample_noise(x)
        D_x = self.forward_precond(model, x + noise, sigma, cond)

        if return_aux:
            return (loss_weight * ((D_x - x) ** 2)), loss_weight, D_x, sigma, bins
        else:
            return (loss_weight * ((D_x - x) ** 2))

    @torch.no_grad()
    def sample(self,
            denoiser_model:nn.Module,
            seed:int=None,
            device:torch.device=None,
            num_samples:int=1,
            x:torch.Tensor = None,
            cond:torch.Tensor = None,
            generator:torch.Generator = None,
            num_steps:int=18,
            sigma_min:float=0.002, 
            sigma_max:int=80,
            rho:int=7,
            S_churn:int=0,
            S_min:int=0,
            S_max=float('inf'),
            S_noise:int=1):
        """
        Sample from the diffusion model.

        Args:
            denoiser_model (nn.Module) :  The denoiser model to use for sampling.
            x (torch.Tensor, optional) : The initial input tensor. If None, a random tensor will be generated.
            cond (torch.Tensor): The conditioning tensor.
            generator (torch.Generator, optional): The random number generator to use.
            num_steps (int): The number of steps to take in the diffusion process.
            sigma_min (float): The minimum noise level.
            sigma_max (float): The maximum noise level.
            rho (int): The exponent for the noise schedule.
            S_churn (int): The amount of noise to add during the sampling process.
            S_min (int): The minimum noise level for the sampling process.
            S_max (float): The maximum noise level for the sampling process.
            S_noise (int): The amount of noise to add during the sampling process.

        
        """
        if x is not None:
            if x.shape[0] != num_samples:
                raise ValueError(
                    f"You have passed a tensor of shape {x.shape}, but requested an effective batch size of {num_samples}."
                    f" Make sure the batch size matches the shape of the tensor."
                )
        if cond is not None:
            if cond.shape[0] != num_samples:
                raise ValueError(
                    f"You have passed a tensor of shape {cond.shape}, but requested an effective batch size of {num_samples}."
                    f" Make sure the batch size matches the shape of the tensor."
                )
        if cond is not None and x is not None:
            if cond.shape[0] != x.shape[0]:
                raise ValueError(
                    f"You have passed a tensor of shape {cond.shape} and a tensor of shape {x.shape}, but requested an"
                    f" effective batch size of {num_samples}. Make sure the batch size matches the shape of the tensors."
                )
        if device is None:
            device = x.device if x is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if generator is None:
            generator = torch.Generator(device=device)
            if seed is not None:
                generator.manual_seed(seed)

        batch_size = num_samples
        input_shape = (batch_size, *self.parameter_size)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if x is None:
            x = randn_tensor(input_shape, generator=generator, device=generator.device)

        step_indices = torch.arange(num_steps, dtype=torch.float64, device=x.device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0

        x_next = x.to(torch.float64) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur, generator)
            t_hat = torch.ones((num_samples,1,1),device=x_next.device) * t_hat

            # Euler step.
            denoised = self.forward_precond_guided(denoiser_model, x_hat, t_hat, cond)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                t_next = torch.ones((num_samples,1,1),device=x_next.device) * t_next
                denoised = self.forward_precond_guided(denoiser_model, x_next, t_next, cond)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next.to(torch.float32)


class DDIMPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        model : Model architecture to denoise the encoded input.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `model` to denoise the encoded input. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    def __init__(self, scheduler, guidance_scale,conditional,use_cfg, parameter_size) -> None:
        super().__init__()

        # make sure scheduler can always be converted to DDIM
        self.scheduler = DDIMScheduler.from_config(scheduler.config)

        self.parameter_size = parameter_size
        self.guidance_scale = guidance_scale
        self.cond_generation = conditional
        self.use_cfg = use_cfg



    def get_loss(self, model, x, cond, return_aux=False):
        noise = torch.randn_like(x)
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (x.shape[0],),
            device=x.device,
        ).long()
        noisy_params = self.scheduler.add_noise(x, noise, timesteps)

        denoiser_output = model(noisy_params, timesteps, cond)


        # Compute loss based on the prediction type
        if self.scheduler.config.prediction_type == "sample":
            loss = F.mse_loss(denoiser_output, x, reduction="none")
        elif self.scheduler.config.prediction_type == "v_prediction":
            loss = F.mse_loss(
                denoiser_output,
                self.scheduler.get_velocity(x, noise, timesteps),
                reduction="none",
            )
        elif self.scheduler.config.prediction_type == "epsilon":
            loss = F.mse_loss(denoiser_output, noise, reduction="none")

        loss = (loss).mean()

        if return_aux:
            return loss, None,denoiser_output,None,None
        return loss


    @torch.no_grad()
    def sample(
            self,
            model:nn.Module,
            cond=None,
            num_samples=None,
            generator = None,
            eta: float = 0.0,
            seed:int=None,
            guidance_scale=1.0,
            device:torch.device=None,
            phi=0.7,
            num_steps: int = 50,  # Default was 50
            use_clipped_model_output: Optional[bool] = None,
    ) -> Dict:

        if generator is None:
            generator = torch.Generator(device=device if device else 'cuda')
            if seed is not None:
                generator.manual_seed(seed)
            else:
                seed = random.randint(0, 2**32 - 1)
                generator.manual_seed(seed)


        # Sample gaussian noise to begin loop
        input_shape = (num_samples, *self.parameter_size)

        if isinstance(generator, list) and len(generator) != num_samples:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {num_samples}. Make sure the batch size matches the length of the generators."
            )

        x = randn_tensor(input_shape, generator=generator, device=generator.device)

        # set step values
        self.scheduler.set_timesteps(num_steps, device=generator.device)

        for t in tqdm(self.scheduler.timesteps, desc="Denoising with DDIM"):
            # 1. predict noise model_output
            t =  torch.ones((num_samples,1),device=x.device) * t.to(x.device)

            if self.cond_generation and guidance_scale > 0.0 and cond is not None and self.use_cfg:
                model_output_cond = model(x, t, cond=cond.to(x.device))
                model_output_uncond = model(x, t, cond=None)

                model_output_cfg = model_output_uncond + guidance_scale * (
                        model_output_cond - model_output_uncond
                )
                model_output_rescaled = (
                        model_output_cfg
                        * torch.std(model_output_cond)
                        / torch.std(model_output_cfg)
                )
                model_output = (
                        phi * model_output_rescaled + (1.0 - phi) * model_output_cfg
                )
            elif cond is not None:
                model_output = model(x, t, cond=cond.to(x.device))
            else:
                model_output = model(x, t, cond=None)


            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            x = self.scheduler.step(
                model_output.unsqueeze(1),
                t,
                x.unsqueeze(1),
                eta=eta,
                use_clipped_model_output=use_clipped_model_output,
                generator=generator,
            )["prev_sample"].squeeze(1)

        return x

    def get_mixing_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        #         self.scheduler.set_timesteps(num_inference_steps, device=device)
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order:]

        return timesteps, num_inference_steps - t_start


# Copied from diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar
def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999) -> torch.Tensor:
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


def rescale_zero_terminal_snr(betas):
    """
    Rescales betas to have zero terminal SNR Based on https://arxiv.org/pdf/2305.08891.pdf (Algorithm 1)


    Args:
        betas (`torch.FloatTensor`):
            the betas that the scheduler is being initialized with.

    Returns:
        `torch.FloatTensor`: rescaled betas with zero terminal SNR
    """
    # Convert betas to alphas_bar_sqrt
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2  # Revert sqrt
    alphas = alphas_bar[1:] / alphas_bar[:-1]  # Revert cumprod
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas


class DDIMScheduler(SchedulerMixin, ConfigMixin):
    """
    Denoising diffusion implicit models is a scheduler that extends the denoising procedure introduced in denoising
    diffusion probabilistic models (DDPMs) with non-Markovian guidance.

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`SchedulerMixin`] provides general loading and saving functionality via the [`SchedulerMixin.save_pretrained`] and
    [`~SchedulerMixin.from_pretrained`] functions.

    For more details, see the original paper: https://arxiv.org/abs/2010.02502

    Args:
        num_train_timesteps (`int`): number of diffusion steps used to train the model.
        beta_start (`float`): the starting `beta` value of inference.
        beta_end (`float`): the final `beta` value.
        beta_schedule (`str`):
            the beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear`, `scaled_linear`, or `squaredcos_cap_v2`.
        trained_betas (`np.ndarray`, optional):
            option to pass an array of betas directly to the constructor to bypass `beta_start`, `beta_end` etc.
        clip_sample (`bool`, default `True`):
            option to clip predicted sample for numerical stability.
        clip_sample_range (`float`, default `1.0`):
            the maximum magnitude for sample clipping. Valid only when `clip_sample=True`.
        set_alpha_to_one (`bool`, default `True`):
            each diffusion step uses the value of alphas product at that step and at the previous one. For the final
            step there is no previous alpha. When this option is `True` the previous alpha product is fixed to `1`,
            otherwise it uses the value of alpha at step 0.
        steps_offset (`int`, default `0`):
            an offset added to the inference steps. You can use a combination of `offset=1` and
            `set_alpha_to_one=False`, to make the last step use step 0 for the previous alpha product, as done in
            stable diffusion.
        prediction_type (`str`, default `epsilon`, optional):
            prediction type of the scheduler function, one of `epsilon` (predicting the noise of the diffusion
            process), `sample` (directly predicting the noisy sample`) or `v_prediction` (see section 2.4
            https://imagen.research.google/video/paper.pdf)
        thresholding (`bool`, default `False`):
            whether to use the "dynamic thresholding" method (introduced by Imagen, https://arxiv.org/abs/2205.11487).
            Note that the thresholding method is unsuitable for latent-space diffusion models (such as
            stable-diffusion).
        dynamic_thresholding_ratio (`float`, default `0.995`):
            the ratio for the dynamic thresholding method. Default is `0.995`, the same as Imagen
            (https://arxiv.org/abs/2205.11487). Valid only when `thresholding=True`.
        sample_max_value (`float`, default `1.0`):
            the threshold value for dynamic thresholding. Valid only when `thresholding=True`.
        timestep_spacing (`str`, default `"leading"`):
            The way the timesteps should be scaled. Refer to Table 2. of [Common Diffusion Noise Schedules and Sample
            Steps are Flawed](https://arxiv.org/abs/2305.08891) for more information.
        rescale_betas_zero_snr (`bool`, default `False`):
            whether to rescale the betas to have zero terminal SNR (proposed by https://arxiv.org/pdf/2305.08891.pdf).
            This can enable the model to generate very bright and dark samples instead of limiting it to samples with
            medium brightness. Loosely related to
            [`--offset_noise`](https://github.com/huggingface/diffusers/blob/74fd735eb073eb1d774b1ab4154a0876eb82f055/examples/dreambooth/train_dreambooth.py#L506).
    """

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        #         clip_sample: bool = True,
        clip_sample: bool = False,
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        clip_sample_range: float = 1.0,
        sample_max_value: float = 1.0,
        #         timestep_spacing: str = "leading",
        timestep_spacing: str = "trailing",
        #         rescale_betas_zero_snr: bool = False,
        rescale_betas_zero_snr: bool = True,
    ):
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_train_timesteps, dtype=torch.float32
            )
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = (
                torch.linspace(
                    beta_start**0.5,
                    beta_end**0.5,
                    num_train_timesteps,
                    dtype=torch.float32,
                )
                ** 2
            )
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(
                f"{beta_schedule} does is not implemented for {self.__class__}"
            )

        # Rescale for zero SNR
        if rescale_betas_zero_snr:
            self.betas = rescale_zero_terminal_snr(self.betas)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # At every step in ddim, we are looking into the previous alphas_cumprod
        # For the final step, there is no previous alphas_cumprod because we are already at 0
        # `set_alpha_to_one` decides whether we set this parameter simply to one or
        # whether we use the final alpha of the "non-previous" one.
        self.final_alpha_cumprod = (
            torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
        )

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # setable values
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(
            np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64)
        )

    def scale_model_input(
        self, sample: torch.FloatTensor, timestep: int = None
    ) -> torch.FloatTensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.FloatTensor`): input sample
            timestep (`int`, optional): current timestep

        Returns:
            `torch.FloatTensor`: scaled input sample
        """
        return sample

    def _get_variance(self, timestep, prev_timestep):
        alpha_prod_t = self.alphas_cumprod[timestep.long()]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep.long()]
            if prev_timestep[0][0] >= 0
            else self.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # if len(alpha_prod_t.shape) == 2:
        #     alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        #     alpha_prod_t_prev = alpha_prod_t_prev.unsqueeze(-1)
        #     beta_prod_t = beta_prod_t.unsqueeze(-1)
        #     beta_prod_t_prev = torch.ones_like(alpha_prod_t,device=alpha_prod_t.device)* beta_prod_t_prev


        variance = (beta_prod_t_prev / beta_prod_t) * (
            1 - alpha_prod_t / alpha_prod_t_prev
        )

        return variance

    # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler._threshold_sample
    def _threshold_sample(self, sample: torch.FloatTensor) -> torch.FloatTensor:
        """
        "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the
        prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by
        s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing
        pixels from saturation at each step. We find that dynamic thresholding results in significantly better
        photorealism as well as better image-text alignment, especially when using very large guidance weights."

        https://arxiv.org/abs/2205.11487
        """
        dtype = sample.dtype
        batch_size = sample.shape[0]

        if dtype not in (torch.float32, torch.float64):
            sample = (
                sample.float()
            )  # upcast for quantile calculation, and clamp not implemented for cpu half

        # Flatten sample for doing quantile calculation along each image
        sample_flat = sample.reshape(batch_size, -1)

        abs_sample = sample_flat.abs()  # "a certain percentile absolute pixel value"

        s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
        s = torch.clamp(
            s, min=1, max=self.config.sample_max_value
        )  # When clamped to min=1, equivalent to standard clipping to [-1, 1]

        s = s.unsqueeze(1)  # (batch_size, 1) because clamp will broadcast along dim=0
        sample_flat = (
            torch.clamp(sample_flat, -s, s) / s
        )  # "we threshold xt0 to the range [-s, s] and then divide by s"

        sample_flat = sample_flat.to(dtype)
        sample = sample_flat.reshape(sample.shape)

        return sample

    def set_timesteps(
        self, num_inference_steps: int, device: Union[str, torch.device] = None
    ):
        """
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
        """

        if num_inference_steps > self.config.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.config.num_train_timesteps} timesteps."
            )

        self.num_inference_steps = num_inference_steps

        # "leading" and "trailing" corresponds to annotation of Table 1. of https://arxiv.org/abs/2305.08891
        if self.config.timestep_spacing == "leading":
            step_ratio = self.config.num_train_timesteps // self.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = (
                (np.arange(0, num_inference_steps) * step_ratio)
                .round()[::-1]
                .copy()
                .astype(np.int64)
            )
            timesteps += self.config.steps_offset
        elif self.config.timestep_spacing == "trailing":
            step_ratio = self.config.num_train_timesteps / self.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = np.round(
                np.arange(self.config.num_train_timesteps, 0, -step_ratio)
            ).astype(np.int64)
            timesteps -= 1
        else:
            raise ValueError(
                f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'leading' or 'trailing'."
            )

        self.timesteps = torch.from_numpy(timesteps).to(device)

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.FloatTensor] = None,
    ) -> Dict:

        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        # 1. get previous step value (=t-1)
        prev_timestep = (
            timestep - self.config.num_train_timesteps // self.num_inference_steps
        )

        self.alphas_cumprod = self.alphas_cumprod.to(model_output.device)

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep.long()]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep.long()]
            if prev_timestep[0][0] >= 0
            else self.final_alpha_cumprod
        )

        if len(alpha_prod_t.shape) ==2:
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        if len(alpha_prod_t_prev.shape) == 2:
            alpha_prod_t_prev = alpha_prod_t_prev.unsqueeze(-1)
        elif len(alpha_prod_t.shape) == 1:
            alpha_prod_t = alpha_prod_t.unsqueeze(-1).unsqueeze(-1)

        beta_prod_t = 1 - alpha_prod_t


        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (
                sample - beta_prod_t ** (0.5) * model_output
            ) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (
                sample - alpha_prod_t ** (0.5) * pred_original_sample
            ) / beta_prod_t ** (0.5)
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (
                beta_prod_t**0.5
            ) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (
                beta_prod_t**0.5
            ) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )

        # 4. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance ** (0.5)

        if variance.shape != alpha_prod_t_prev.shape:
            variance = variance.unsqueeze(-1)
            std_dev_t = std_dev_t.unsqueeze(-1)

        if use_clipped_model_output:
            # the pred_epsilon is always re-derived from the clipped x_0 in Glide
            pred_epsilon = (
                sample - alpha_prod_t ** (0.5) * pred_original_sample
            ) / beta_prod_t ** (0.5)

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (
            0.5
        ) * pred_epsilon

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = (
            alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        )

        if eta > 0:
            if variance_noise is not None and generator is not None:
                raise ValueError(
                    "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                    " `variance_noise` stays `None`."
                )

            if variance_noise is None:
                variance_noise = randn_tensor(
                    model_output.shape,
                    generator=generator,
                    device=model_output.device,
                    dtype=model_output.dtype,
                )
            variance = std_dev_t * variance_noise

            prev_sample = prev_sample + variance

        return dict(prev_sample=prev_sample, pred_original_sample=pred_original_sample)

    # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler.add_noise
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        alphas_cumprod = self.alphas_cumprod.to(
            device=original_samples.device, dtype=original_samples.dtype
        )
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = (
            sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        )
        return noisy_samples

    # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler.get_velocity
    def get_velocity(
        self,
        sample: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as sample
        alphas_cumprod = self.alphas_cumprod.to(
            device=sample.device, dtype=sample.dtype
        )
        timesteps = timesteps.to(sample.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity

    def __len__(self):
        return self.config.num_train_timesteps

    def compute_snr(self, timesteps):
        """
        Computes SNR as per
        https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = self.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
            timesteps
        ].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
            device=timesteps.device
        )[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

