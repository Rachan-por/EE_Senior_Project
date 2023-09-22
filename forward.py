import torch.nn.functional as F
import torch


class Diffusion():
    def __init__(self, timesteps: int, device: str):
        self.device = device
        self.T = timesteps
        self.betas = torch.linspace(0.0001, 0.02, self.T).to(self.device)
        self.alphas = 1. - self.betas.to(self.device)
        self.initialize()

    def initialize(self):
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * \
            (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def forward_diffusion_sample(self, x_0, t, device):
        """ 
        Takes an image and a timestep as input and 
        returns the noisy version of it
        """
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_alphas_cumprod, t, x_0.shape, device)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape, device
        )
        # mean + variance
        return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
            + sqrt_one_minus_alphas_cumprod_t.to(device) * \
            noise.to(device), noise.to(device)

    def get_index_from_list(self, vals, t, x_shape, device):
        """ 
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t.to(device))
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    @torch.no_grad()
    def sample_timestep(self, model, x, t):
        """
        Calls the model to predict the noise in the image and returns 
        the denoised image. 
        Applies noise to this image, if we are not in the last step yet.
        """
        device = next(model.parameters()).device
        betas_t = self.get_index_from_list(self.betas, t, x.shape, device)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape, device
        )
        sqrt_recip_alphas_t = self.get_index_from_list(
            self.sqrt_recip_alphas, t, x.shape, device)

        # Call model (current image - noise prediction)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = self.get_index_from_list(
            self.posterior_variance, t, x.shape, device)

        if t == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample_image(self, model, T):

        # device
        device = next(model.parameters()).device

        # Sample noise
        img_size = 128
        img = torch.randn((1, 3, img_size, img_size), device=device)
        imgs = []

        # for loop sampling
        for i in reversed(range(0, T)):
            t = torch.full((1,), i, device=device, dtype=torch.long)
            img = self.sample_timestep(model, img, t)
            imgs.append(img)
            if i == 0:
                print(f"Sampling is successful.")
        return imgs
