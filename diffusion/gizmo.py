import torch

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