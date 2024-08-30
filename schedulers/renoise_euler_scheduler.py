from diffusers import EulerAncestralDiscreteScheduler
from diffusers.utils import BaseOutput
import torch
from typing import List, Optional, Tuple, Union
import numpy as np

class EulerAncestralDiscreteSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.FloatTensor
    pred_original_sample: Optional[torch.FloatTensor] = None

class ReNoiseEulerAncestralDiscreteScheduler(EulerAncestralDiscreteScheduler):
    def set_noise_list(self, noise_list):
        self.noise_list = noise_list

    def get_noise_to_remove(self, sigmas, step_index):
        sigma_from = sigmas[step_index]
        sigma_to = sigmas[step_index + 1]
        sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5

        return self.noise_list[step_index] * sigma_up
        
    def scale_model_input(
        self, sample: torch.FloatTensor, timestep: Union[float, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep. Scales the denoising model input by `(sigma**2 + 1) ** 0.5` to match the Euler algorithm.

        Args:
            sample (`torch.FloatTensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.FloatTensor`:
                A scaled input sample.
        """

        self._init_step_index(timestep.view((1)))
        return EulerAncestralDiscreteScheduler.scale_model_input(self, sample, timestep)

    
    def step(
        self,
        model_output: torch.FloatTensor,
        sigmas,
        step_index,
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[EulerAncestralDiscreteSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a
                [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] or tuple.

        Returns:
            [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`,
                [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] is returned,
                otherwise a tuple is returned where the first element is the sample tensor.

        """

        sigma = sigmas[step_index]

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        if self.config.prediction_type == "epsilon":
            pred_original_sample = sample - sigma * model_output
        elif self.config.prediction_type == "v_prediction":
            # * c_out + input * c_skip
            pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
        elif self.config.prediction_type == "sample":
            raise NotImplementedError("prediction_type not implemented yet: sample")
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )

        sigma_from = sigmas[step_index]
        sigma_to = sigmas[step_index + 1]
        sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
        sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5

        # 2. Convert to an ODE derivative
        # derivative = (sample - pred_original_sample) / sigma
        derivative = model_output

        dt = sigma_down - sigma

        prev_sample = sample + derivative * dt

        device = model_output.device
        # noise = randn_tensor(model_output.shape, dtype=model_output.dtype, device=device, generator=generator)
        # prev_sample = prev_sample + noise * sigma_up

        prev_sample = prev_sample + self.noise_list[step_index] * sigma_up

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one

        if not return_dict:
            return (prev_sample,)

        return EulerAncestralDiscreteSchedulerOutput(
            prev_sample=prev_sample, pred_original_sample=pred_original_sample
        )
    
    def step_and_update_noise(
        self,
        model_output: torch.FloatTensor,
        sigmas,
        step_index,
        sample: torch.FloatTensor,
        expected_prev_sample: torch.FloatTensor,
        optimize_epsilon_type: bool = False,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[EulerAncestralDiscreteSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a
                [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] or tuple.

        Returns:
            [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`,
                [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] is returned,
                otherwise a tuple is returned where the first element is the sample tensor.

        """

        sigma = sigmas[step_index]

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        if self.config.prediction_type == "epsilon":
            pred_original_sample = sample - sigma * model_output
        elif self.config.prediction_type == "v_prediction":
            # * c_out + input * c_skip
            pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
        elif self.config.prediction_type == "sample":
            raise NotImplementedError("prediction_type not implemented yet: sample")
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )

        sigma_from = sigmas[step_index]
        sigma_to = sigmas[step_index + 1]
        sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
        sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5

        # 2. Convert to an ODE derivative
        # derivative = (sample - pred_original_sample) / sigma
        derivative = model_output

        dt = sigma_down - sigma

        prev_sample = sample + derivative * dt

        device = model_output.device
        # noise = randn_tensor(model_output.shape, dtype=model_output.dtype, device=device, generator=generator)
        # prev_sample = prev_sample + noise * sigma_up

        if sigma_up > 0:
            req_noise = (expected_prev_sample - prev_sample) / sigma_up
            if not optimize_epsilon_type:
                self.noise_list[step_index] = req_noise
            else:
                with torch.inference_mode(False):
                    with torch.enable_grad():
                        noise_list = [noise_.clone() for noise_ in self.noise_list]
                        for i in range(10):
                            n = torch.autograd.Variable(noise_list[step_index].detach().clone(), requires_grad=True)
                            loss = torch.norm(n - req_noise.detach())
                            loss.backward()
                            noise_list[step_index] -= n.grad.detach() * 1.8
                self.noise_list = noise_list
                


        prev_sample = prev_sample + self.noise_list[step_index] * sigma_up

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one

        if not return_dict:
            return (prev_sample,)

        return EulerAncestralDiscreteSchedulerOutput(
            prev_sample=prev_sample, pred_original_sample=pred_original_sample
        )
    
    def inv_step(
        self,
        model_output: torch.FloatTensor,
        sigmas,
        step_index,
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[EulerAncestralDiscreteSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a
                [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] or tuple.

        Returns:
            [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`,
                [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] is returned,
                otherwise a tuple is returned where the first element is the sample tensor.

        """
        sigma = sigmas[step_index]

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        if self.config.prediction_type == "epsilon":
            pred_original_sample = sample - sigma * model_output
        elif self.config.prediction_type == "v_prediction":
            # * c_out + input * c_skip
            pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
        elif self.config.prediction_type == "sample":
            raise NotImplementedError("prediction_type not implemented yet: sample")
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )

        sigma_from = sigmas[step_index]
        sigma_to = sigmas[step_index+1]
        # sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
        sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2).abs() / sigma_from**2) ** 0.5
        # sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
        sigma_down = sigma_to**2 / sigma_from

        # 2. Convert to an ODE derivative
        # derivative = (sample - pred_original_sample) / sigma
        derivative = model_output

        dt = sigma_down - sigma
        # dt = sigma_down - sigma_from

        prev_sample = sample - derivative * dt

        device = model_output.device
        # noise = randn_tensor(model_output.shape, dtype=model_output.dtype, device=device, generator=generator)
        # prev_sample = prev_sample + noise * sigma_up
        # print(step_index)
        prev_sample = prev_sample - self.noise_list[step_index] * sigma_up

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)

        if not return_dict:
            return (prev_sample,)

        return EulerAncestralDiscreteSchedulerOutput(
            prev_sample=prev_sample, pred_original_sample=pred_original_sample
        )
    