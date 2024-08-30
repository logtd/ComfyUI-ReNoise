import torch
import torch.nn.functional as F

# Based on code from https://github.com/pix2pixzero/pix2pix-zero
@torch.inference_mode(False)
@torch.enable_grad()
def noise_regularization(
    e_t, noise_pred_optimal, lambda_kl, lambda_ac, num_reg_steps, num_ac_rolls, generator=None
):
    for _outer in range(num_reg_steps):
        if lambda_kl > 0:
            _var = torch.autograd.Variable(e_t.detach().clone(), requires_grad=True)
            l_kld = patchify_latents_kl_divergence(_var, noise_pred_optimal)
            l_kld.backward()
            _grad = _var.grad.detach()
            _grad = torch.clip(_grad, -100, 100)
            e_t = e_t - lambda_kl * _grad
        if lambda_ac > 0:
            for _inner in range(num_ac_rolls):
                _var = torch.autograd.Variable(e_t.detach().clone(), requires_grad=True)
                l_ac = auto_corr_loss(_var, generator=generator)
                l_ac.backward()
                _grad = _var.grad.detach() / num_ac_rolls
                e_t = e_t - lambda_ac * _grad
        e_t = e_t.detach()

    return e_t

# Based on code from https://github.com/pix2pixzero/pix2pix-zero
def auto_corr_loss(
        x, random_shift=True, generator=None
):
    B, C, H, W = x.shape
    # assert B == 1
    x = x.squeeze(0)
    # x must be shape [C,H,W] now
    reg_loss = 0.0
    for ch_idx in range(x.shape[0]):
        noise = x[ch_idx][None, None, :, :]
        while True:
            if random_shift:
                roll_amount = torch.randint(0, noise.shape[2] // 2, (1,), generator=generator).item()
            else:
                roll_amount = 1
            reg_loss += (
                noise * torch.roll(noise, shifts=roll_amount, dims=2)
            ).mean() ** 2
            reg_loss += (
                noise * torch.roll(noise, shifts=roll_amount, dims=3)
            ).mean() ** 2
            if noise.shape[2] <= 8:
                break
            noise = F.avg_pool2d(noise, kernel_size=2)
    return reg_loss


def patchify_latents_kl_divergence(x0, x1, patch_size=4, num_channels=4):

    def patchify_tensor(input_tensor):
        patches = (
            input_tensor.unfold(1, patch_size, patch_size)
            .unfold(2, patch_size, patch_size)
            .unfold(3, patch_size, patch_size)
        )
        patches = patches.contiguous().view(-1, num_channels, patch_size, patch_size)
        return patches

    x0 = patchify_tensor(x0)
    x1 = patchify_tensor(x1)

    kl = latents_kl_divergence(x0, x1).sum()
    return kl


def latents_kl_divergence(x0, x1):
    EPSILON = 1e-6
    x0 = x0.view(x0.shape[0], x0.shape[1], -1)
    x1 = x1.view(x1.shape[0], x1.shape[1], -1)
    mu0 = x0.mean(dim=-1)
    mu1 = x1.mean(dim=-1)
    var0 = x0.var(dim=-1)
    var1 = x1.var(dim=-1)
    kl = (
        torch.log((var1 + EPSILON) / (var0 + EPSILON))
        + (var0 + (mu0 - mu1) ** 2) / (var1 + EPSILON)
        - 1
    )
    kl = torch.abs(kl).sum(dim=-1)
    return kl


# TODO notice it does 2x here
@torch.no_grad()
def unet_pass(pipe, z_t, t, prompt_embeds, added_cond_kwargs):
    latent_model_input = torch.cat([z_t] * 2) if pipe.do_classifier_free_guidance else z_t
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
    return pipe.unet(
        latent_model_input,
        t,
        encoder_hidden_states=prompt_embeds,
        timestep_cond=None,
        cross_attention_kwargs=pipe.cross_attention_kwargs,
        added_cond_kwargs=added_cond_kwargs,
        return_dict=False,
    )[0]
