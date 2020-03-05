import torch
import torch.nn as nn
import numpy as np
from ..bottleneck.gaussian_kernel import SpatialGaussianKernel


class AttributionBottleneck(nn.Module):

    @staticmethod
    def _sample_z(mu, log_noise_var):
        """ return mu with additive noise """
        log_noise_var = torch.clamp(log_noise_var, -10, 10)
        noise_std = (log_noise_var / 2).exp()
        eps = mu.data.new(mu.size()).normal_()
        return mu + noise_std * eps

    @staticmethod
    def _calc_capacity(mu, log_var) -> torch.Tensor:
        # KL[Q(z|x)||P(z)]
        # 0.5 * (tr(noise_cov) + mu ^ T mu - k  -  log det(noise)
        return -0.5 * (1 + log_var - mu**2 - log_var.exp())


class PerSampleBottleneck(AttributionBottleneck):
    """
    The Attribution Bottleneck.
    Is inserted in a existing model to suppress information, parametrized by a suppression mask alpha.
    """
    def __init__(self, mean: np.ndarray, std: np.ndarray, sigma, device=None, relu=False):
        """
        :param mean: The empirical mean of the activations of the layer
        :param std: The empirical standard deviation of the activations of the layer
        :param sigma: The standard deviation of the gaussian kernel to smooth the mask, or None for no smoothing
        :param device: GPU/CPU
        :param relu: True if output should be clamped at 0, to imitate a post-ReLU distribution
        """
        super().__init__()
        self.device = device
        self.relu = relu
        self.initial_value = 5.0
        self.std = torch.tensor(std, dtype=torch.float, device=self.device, requires_grad=False)
        self.mean = torch.tensor(mean, dtype=torch.float, device=self.device, requires_grad=False)
        self.alpha = nn.Parameter(torch.full((1, *self.mean.shape), fill_value=self.initial_value, device=self.device))
        self.sigmoid = nn.Sigmoid()
        self.buffer_capacity = None  # Filled on forward pass, used for loss

        if sigma is not None and sigma > 0:
            # Construct static conv layer with gaussian kernel
            kernel_size = int(round(2 * sigma)) * 2 + 1  # Cover 2.5 stds in both directions
            channels = self.mean.shape[0]
            self.smooth = SpatialGaussianKernel(kernel_size, sigma, channels, device=self.device)
        else:
            self.smooth = None

        self.reset_alpha()

    def reset_alpha(self):
        """ Used to reset the mask to train on another sample """
        with torch.no_grad():
            self.alpha.fill_(self.initial_value)
        return self.alpha

    def forward(self, r):
        """ Remove information from r by performing a sampling step, parametrized by the mask alpha """
        # Smoothen and expand a on batch dimension
        lamb = self.sigmoid(self.alpha)
        lamb = lamb.expand(r.shape[0], r.shape[1], -1, -1)
        lamb = self.smooth(lamb) if self.smooth is not None else lamb

        # We normalize r to simplify the computation of the KL-divergence
        #
        # The equation in the paper is:
        # Z = λ * R + (1 - λ) * ε)
        # where ε ~ N(μ_r, σ_r**2)
        #  and given R the distribution of Z ~ N(λ * R, ((1 - λ) σ_r)**2)
        #
        # In the code μ_r = self.mean and σ_r = self.std.
        #
        # To simplify the computation of the KL-divergence we normalize:
        #   R_norm = (R - μ_r) / σ_r
        #   ε ~ N(0, 1)
        #   Z_norm ~ N(λ * R_norm, (1 - λ))**2)
        #   Z =  σ_r * Z_norm + μ_r
        #
        # We compute KL[ N(λ * R_norm, (1 - λ))**2) || N(0, 1) ].
        #
        # The KL-divergence is invariant to scaling, see:
        # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Properties

        r_norm = (r - self.mean) / self.std

        # Get sampling parameters
        noise_var = (1-lamb)**2
        scaled_signal = r_norm * lamb
        noise_log_var = torch.log(noise_var)

        # Sample new output values from p(z|r)
        z_norm = self._sample_z(scaled_signal, noise_log_var)
        self.buffer_capacity = self._calc_capacity(scaled_signal, noise_log_var)

        # Denormalize z to match magnitude of r
        z = z_norm * self.std + self.mean

        # Clamp output, if input was post-relu
        if self.relu:
            z = torch.clamp(z, 0.0)

        return z
