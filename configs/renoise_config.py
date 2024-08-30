from dataclasses import dataclass


@dataclass
class ReNoiseConfig:
    scheduler: str
    sampler: str
    noise_seed: int
    inversion_steps: int
    renoise_steps: int
    avg_latent_estimations: bool
    avg_first_step_start: int
    avg_first_step_end: int
    avg_step_start: int
    avg_step_end: int
    noise_reg_lambda_ac: float
    noise_reg_lambda_kl: float
    noise_reg_steps: int
    noise_reg_ac_rolls: int
    perform_noise_correction: bool
    inversion_strength: float
