
from typing import Dict, List
import torch
import random

from diffusers.utils.torch_utils import randn_tensor
from tqdm import trange

import comfy.samplers

from ..configs.renoise_config import ReNoiseConfig
from ..schedulers.renoise_euler_scheduler import ReNoiseEulerAncestralDiscreteScheduler
from ..utils.renoise import noise_regularization


def create_noise_list(latent, length, generator=None):
    latents_size = latent.shape
    return [randn_tensor(latents_size, dtype=torch.float16, device=torch.device("cuda:0"), generator=generator) for i in range(length)]


def get_scheduler(config: ReNoiseConfig):
    scheduler = ReNoiseEulerAncestralDiscreteScheduler() # TODO from model
    return scheduler


def get_timesteps(scheduler, num_inference_steps, strength):
        denoising_start = 1.0 - strength
        # get the original timestep using init_timestep
        if denoising_start is None:
            init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
            t_start = max(num_inference_steps - init_timestep, 0)
        else:
            t_start = 0

        timesteps = scheduler.timesteps[t_start * scheduler.order :]

        # Strength is irrelevant if we directly request a timestep to start at;
        # that is, strength is determined by the denoising_start instead.
        if denoising_start is not None:
            discrete_timestep_cutoff = int(
                round(
                    scheduler.config.num_train_timesteps
                    - (denoising_start * scheduler.config.num_train_timesteps)
                )
            )

            num_inference_steps = (timesteps < discrete_timestep_cutoff).sum().item()
            if scheduler.order == 2 and num_inference_steps % 2 == 0:
                # if the scheduler is a 2nd order scheduler we might have to do +1
                # because `num_inference_steps` might be even given that every timestep
                # (except the highest one) is duplicated. If `num_inference_steps` is even it would
                # mean that we cut the timesteps in the middle of the denoising step
                # (between 1st and 2nd derivative) which leads to incorrect results. By adding 1
                # we ensure that the denoising process always ends after the 2nd derivate step of the scheduler
                num_inference_steps = num_inference_steps + 1

            # because t_n+1 >= t_n, we slice the timesteps starting from the end
            timesteps = timesteps[-num_inference_steps:]
            return timesteps

        return timesteps


def build_sampler_fn(config: ReNoiseConfig, latent: torch.Tensor, model):
    generator = torch.Generator().manual_seed(config.noise_seed)
    scheduler = get_scheduler(config)
    noise_list = create_noise_list(latent, config.inversion_steps, generator=generator)
    scheduler.set_noise_list(noise_list)
    first_step_max_timestep = 250 # TODO
    model_sampling = model.get_model_object("model_sampling")
    sigmas = comfy.samplers.calculate_sigmas(model_sampling, config.scheduler, config.inversion_steps).cpu()

    average_step_range = (config.avg_step_start, config.avg_step_end)
    average_first_step_range = (config.avg_first_step_start, config.avg_first_step_end)

    z_0 = latent.clone()
    base_noise = randn_tensor(latent.shape, dtype=torch.float16, device=torch.device("cuda:0"), generator=generator)


    @torch.no_grad()
    def inversion_step(model, sigmas, step_index, z_t, z_0, z_noise, extra_args):
        nonlocal model_sampling
        sigma = sigmas[step_index]
        t = model_sampling.timestep(sigma)
        s_in = z_t.new_ones([z_t.shape[0]])
        extra_step_kwargs = {}
        avg_range = average_first_step_range if t.item() < first_step_max_timestep else average_step_range
        num_renoise_steps = min(config.avg_first_step_end+1, config.renoise_steps) if t.item() < first_step_max_timestep else config.renoise_steps

        nosie_pred_avg = None
        noise_pred_optimal = None
        # z_tp1_forward = scheduler.add_noise(z_0, z_noise, t.view((1))).detach().to(z_t.device)
        z_tp1_forward = z_0 + z_noise * sigmas[step_index]
        approximated_z_tp1 = z_t.clone()
        for i in range(num_renoise_steps + 1):
            if config.noise_reg_steps > 0 and i == 0:
                extra_args['model_options']['INJECTION_OFF'] = False
                noise_pred = model(approximated_z_tp1, sigma * s_in, **extra_args)
                extra_args['model_options']['INJECTION_OFF'] = True
                noise_pred_optimal = model(z_tp1_forward, sigma * s_in, **extra_args)
            else:
                if i == 0:
                    extra_args['model_options']['INJECTION_OFF'] = False
                noise_pred = model(approximated_z_tp1, sigma * s_in, **extra_args)
                extra_args['model_options']['INJECTION_OFF'] = True

            # Calculate average noise
            if  i >= avg_range[0] and i < avg_range[1]:
                j = i - avg_range[0]
                if nosie_pred_avg is None:
                    nosie_pred_avg = noise_pred.clone()
                else:
                    nosie_pred_avg = j * nosie_pred_avg / (j + 1) + noise_pred / (j + 1)

            if i >= avg_range[0] or (not config.avg_latent_estimations and i > 0):
                noise_pred = noise_regularization(noise_pred, noise_pred_optimal, lambda_kl=config.noise_reg_lambda_kl, lambda_ac=config.noise_reg_lambda_ac, num_reg_steps=config.noise_reg_steps, num_ac_rolls=config.noise_reg_ac_rolls, generator=generator)
            
            approximated_z_tp1 = scheduler.inv_step(noise_pred, sigmas, step_index, z_t, **extra_step_kwargs, return_dict=False)[0].detach()

        # if average latents is enabled, we need to perform an additional step with the average noise
        if config.avg_latent_estimations and nosie_pred_avg is not None:
            nosie_pred_avg = noise_regularization(nosie_pred_avg, noise_pred_optimal, lambda_kl=config.noise_reg_lambda_kl, lambda_ac=config.noise_reg_lambda_ac, num_reg_steps=config.noise_reg_steps, num_ac_rolls=config.noise_reg_ac_rolls, generator=generator)
            approximated_z_tp1 = scheduler.inv_step(nosie_pred_avg, sigmas, step_index, z_t, **extra_step_kwargs, return_dict=False)[0].detach()

        # perform noise correction
        if config.perform_noise_correction:
            # noise_pred = unet_pass(pipe, approximated_z_tp1, t, prompt_embeds, added_cond_kwargs)
            noise_pred = model(approximated_z_tp1, sigma* s_in, **extra_args)
            scheduler.step_and_update_noise(noise_pred, sigmas, step_index, approximated_z_tp1, z_t, return_dict=False, optimize_epsilon_type=config.perform_noise_correction)

        return approximated_z_tp1
    
    def sample_renoise_inversion(model, x, sigmas, extra_args=None, callback=None, disable=None, **kwargs):
        extra_args = extra_args if extra_args is not None else {}
        extra_args = extra_args.copy()
        model_options = extra_args.get('model_options', {})
        model_options['INJECTION_OFF'] = True
        z_base_noise = base_noise.to(x.device)
        z_base = z_0.to(x.device)
        for i in trange(len(sigmas) - 2, -1, -1, disable=disable):
            x = inversion_step(model, sigmas, i, x, z_base, z_base_noise, extra_args)
            if callback is not None:
                callback({'x': x, 'i': len(sigmas) - i -1, 'sigma': sigmas[i], 'denoised': x})
        del model_options['INJECTION_OFF']
        return x
    
    def sample_renoise(model, x, sigmas, extra_args=None, callback=None, disable=None, **kwargs):
        extra_args = extra_args if extra_args is not None else {}
        s_in = x.new_ones([x.shape[0]])
        for i in trange(len(sigmas) - 1, disable=disable):
            noise_pred = model(x, sigmas[i] * s_in, **extra_args)
            x = scheduler.step(noise_pred, sigmas, i, x, generator=generator)[0]
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'denoised': x})
        return x
    
    return sample_renoise_inversion, sample_renoise, sigmas
