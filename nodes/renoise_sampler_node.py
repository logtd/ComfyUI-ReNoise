from dataclasses import dataclass
import comfy.sd
import comfy.model_base
import comfy.samplers
import comfy.sample
import comfy.k_diffusion.sampling

from ..configs.renoise_config import ReNoiseConfig
from ..samplers.renoise_sampler import build_sampler_fn


class ReNoiseSamplerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                 "sampler": (["euler_ancestral"],),
                 "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                 "inersion_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                 "inversion_steps": ("INT", {"default": 4, "min": 1, "max": 10000}),
                 "renoise_steps": ("INT", {"default": 9, "min": 0, "max": 10000}),
                 "avg_latent_estimations": ("BOOLEAN", { "default": True }),
                 "avg_first_step_start": ("INT", {"default": 0, "min": 0, "max": 10000}),
                 "avg_first_step_end": ("INT", {"default": 5, "min": 0, "max": 10000}),
                 "avg_step_start": ("INT", {"default": 8, "min": 0, "max": 10000}),
                 "avg_step_end": ("INT", {"default": 10, "min": 0, "max": 10000}),
                 "noise_reg_lambda_ac": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 1000.0, "step": 0.1, "round": 0.01}),
                 "noise_reg_lambda_kl": ("FLOAT", {"default": 0.065, "min": 0.0, "max": 1000.0, "step": 0.01, "round": 0.001}),
                 "noise_reg_steps": ("INT", {"default": 4, "min": 0, "max": 10000}),
                 "noise_reg_ac_rolls": ("INT", {"default": 5, "min": 0, "max": 10000}),
                 "perform_noise_correction": ("BOOLEAN", { "default": True }),
                 "latent": ("LATENT",),
                 "model": ("MODEL",)
                 }
                }

    RETURN_TYPES = ("SAMPLER", "SAMPLER","SIGMAS")
    RETURN_NAMES = ("inv_sampler", "sampler", "sigmas")
    FUNCTION = "build"

    CATEGORY = "sampling"

    def build(self, 
              scheduler, 
              sampler,
              noise_seed,
              inersion_strength,
              inversion_steps,
              renoise_steps,
              avg_latent_estimations,
              avg_first_step_start,
              avg_first_step_end,
              avg_step_start,
              avg_step_end,
              noise_reg_lambda_ac,
              noise_reg_lambda_kl,
              noise_reg_steps,
              noise_reg_ac_rolls,
              perform_noise_correction,
              latent,
              model,
              ):
        renoise_config = ReNoiseConfig(
            scheduler, 
            sampler,
            noise_seed,
            inversion_steps,
            renoise_steps,
            avg_latent_estimations,
            avg_first_step_start,
            avg_first_step_end,
            avg_step_start,
            avg_step_end,
            noise_reg_lambda_ac,
            noise_reg_lambda_kl,
            noise_reg_steps,
            noise_reg_ac_rolls,
            perform_noise_correction,
            inersion_strength
        )

        inverse_sampler_fn, sampler_fn, sigmas = build_sampler_fn(renoise_config, latent['samples'], model)

        inverse_ksampler = comfy.samplers.KSAMPLER('euler_ancestral')
        inverse_ksampler.sampler_function = inverse_sampler_fn

        ksampler = comfy.samplers.KSAMPLER('euler_ancestral')
        ksampler.sampler_function = sampler_fn

        return (inverse_ksampler, ksampler, sigmas)
