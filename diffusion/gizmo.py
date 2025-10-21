import torch
import torch.nn as nn

class LinearNoiseScheduler:
    def __init__(self, num_timesteps, beta_start, beta_end):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # `torch.linspace()` does the following:
        # a = torch.linspace(0, 10, steps=5)
        #   = tensor([ 0.0000,  2.5000,  5.0000,  7.5000, 10.0000])
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        
        # $\alpha_t = 1 - \beta_t$
        self.alphas = 1. - self.betas
        

        # `torch.cumprod()` calculates the cumulative product along a dim
        # x = torch.tensor([[1, 2, 3],
        #                   [4, 5, 6]])
        # torch.cumprod(x, dim=0)
        #   =       tensor([[ 1,  2,  3],
        #             -->   [ 4, 10, 18]])
        
        # $\bar{\alpha}_t = \prod_{i=1}^{t} \alpha_i$

        self.alpha_cum_prod                 = torch.cumprod(self.alphas, dim = 0)

        self.sqrt_alpha_cum_prod            = torch.sqrt(self.alpha_cum_prod) # $\sqrt{\bar{\alpha}_t}$        

        self.one_minus_sqrt_alpha_cum_prod  = torch.sqrt(1. - self.alpha_cum_prod) # $\sqrt{1- \bar{\alpha}_t}$

    # This will be our fwd process of adding noise
    def add_noise(self, original, noise, t):
        # The images and noise will be of $(B \times C \times H \times W)$
        # Timestep will be a 1D tensor of side $B$
        
        batch_size = original.shape[0]
        
        sqrt_alpha_cum_prod             = self.sqrt_alpha_cum_prod[t].reshape(batch_size)
        sqrt_one_minus_alpha_cum_prod   = self.sqrt_one_minus_alpha_cum_prod[t].reshape(batch_size)

        # (batch_size,) to (batch_size, 1, 1, 1) via 4 unsqueezes
        for _ in range(len(original.shape)-1):
            sqrt_alpha_cum_prod             = sqrt_alpha_cum_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_cum_prod   = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)

        # $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1- \bar{\alpha}_t}\epsilon$
        return sqrt_alpha_cum_prod * original + sqrt_one_minus_alpha_cum_prod * noise
        

    # Takes image $x_t$ and gives us sample from learned reverse distrubtion
    def sample_prev_timestep(self, xt, noise_pred, t):
        # Predict $x_0$ from $x_t$ and noise prediction
        
        # $x_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$
        
        x0 = (xt - (self.sqrt_one_minus_alpha_cum_prod[t] * noise_pred)) / self.sqrt_alpha_cum_prod[t]
        x0 = torch.clamp(x0, -1., max=1.)
        
        # Compute posterior mean
        # That's the expected value of the previous timestep's image
        # given the current noisy image and the model's noise prediction.
        
        # $\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t)\right)$
        
        mean = xt - ((self.betas[t] * noise_pred) / (self.sqrt_one_minus_alpha_cum_prod[t]))
        mean = mean / torch.sqrt(self.alphas[t])
        
        if t == 0:
            return mean, x0
        else:
            # Compute posterior variance
            
            # $\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$
            
            variance = (1 - self.alpha_cum_prod[t-1]) / (1. - self.alpha_cum_prod[t])
            variance = variance * self.betas[t]
            
            # $\sigma_t = \sqrt{\tilde{\beta}_t}$
            
            sigma = variance ** 0.5
            
            # $z \sim \mathcal{N}(0, I)$
            z = torch.randn(xt.shape).to(xt.device)
            
            # $x_{t-1} = \mu_\theta(x_t, t) + \sigma_t z$
            
            # Also return $x_0$ for funsies

            return mean + sigma*z, x0
        

# `TimeEmbeddingBlock`
# Takes a 1D tensor of timesteps of size batch_size (B,)
# and gives us a time embedding of (B * t_emb_dim)



def get_time_embedding(time_steps, t_emb_dim):

    # so basically, $\sin(pos / 10000^{2i/d_{\text{model}}})$ AND $\cos(pos / 10000^{2i/d_{\text{model}}})$
    
    factor = 10000 ** ((torch.arange(
        start=0, end=t_emb_dim//2, device=time_steps.device) / (t_emb_dim // 2)
    ))
    t_emb = time_steps[:, None].repeat(1, t_emb_dim // 2) / factor
    t_emb = torch.cat(tensors=[torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, down_sample, num_heads):
        super().__init__()
        self.down_sample = down_sample
        
        # `nn.Sequential(a, b c)` is in place of `x = c(b(a(x)))`
        self.resnet_conv_first = nn.Sequential(
            # Normalization that splits channels into groups and normalizes within each group.
            nn.GroupNorm(num_groups=8, in_channels=in_channels),

            # [how i feel not having to implement that swiglu bullshit](https://tenor.com/view/ishowspeed-try-not-to-laugh-gif-7682731162751353849)
            nn.SiLU(),
            
            # applies a learnable filter that slides across an image to extract features
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

        self.t_emb_layers = nn.Sequential(
            nn.SilU(),
            nn.Linear(t_emb_dim, out_channels)
        )

        self.resnet_conv_second = nn.Sequential(
            nn.GroupNorm(num_groups=8, in_channels=out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

        self.attention_norm = nn.GroupNorm(8, out_channels)
        self.attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
        self.residual_input_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # residual ensures input of entire resnet block can be added to output of last conv layer
        self.down_sample_conv = nn.Conv2d(out_channels, out_channels, kernel_size=4,
                                         stride=2, padding=1) if self.down_sample else nn.Identity() # $I$ -> no-op.
    
    def forward(self, x, t_emb):
        pass

