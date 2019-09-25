from __future__ import print_function

import torch.autograd
from attribution_bottleneck.attribution.base import AttributionMethod

# from ..attribution.base import AttributionMethod
from attribution_bottleneck.attribution.backprop import ModifiedBackpropMethod
from tqdm import tqdm


from ..utils.baselines import Mean, Baseline
from ..utils.misc import *


class AveragingGradient(AttributionMethod):
    """
    Something than can make a attribution heatmap from several inputs and a backpropagating attribution method.
    The resulting heatmap is the sum of all the other methods.
    """
    def __init__(self, backprop: ModifiedBackpropMethod):
        super().__init__()
        self.verbose = False
        self.progbar = False
        self.backprop = backprop

    def heatmap(self, input_t, target):
        # generate sample list (different per method)
        images = self._get_samples(input_t)
        target_t = target if isinstance(target, torch.Tensor) else torch.tensor(target, device=input_t.device)
        assert isinstance(target_t, torch.Tensor)
        assert len(images[0].shape) == 4, "{} makes dim {} !".format(images[0].shape, len(images[0].shape))  # C x N x N

        grads = self._backpropagate_multiple(images, target_t)

        # Reduce sample dimension
        grads_mean = np.mean(grads, axis=0)
        # Reduce batch dimension
        grads_rgb = np.mean(grads_mean, axis=0)
        # Reduce color dimension
        heatmap = np.mean(grads_rgb, axis=0)
        return heatmap

    def _backpropagate_multiple(self, inputs: list, target_t: torch.Tensor):
        """
        returns an array with all the computed gradients
        shape: N_Inputs x Batches=1 x Color Channels x Height x Width
        """
        # Preallocate empty gradient stack
        grads = np.zeros((len(inputs), *inputs[0].shape))
        # Generate gradients
        iterator = tqdm(range(len(inputs)), ncols=100, desc="calc grad") if len(inputs) > 1 and self.progbar else range(len(inputs))
        for i in iterator:
            grad = self.backprop._calc_gradient(input_t=inputs[i], target_t=target_t)
            # If required, add color dimension
            if len(grad.shape) == 3:
                np.expand_dims(grad, axis=0)
            grads[i, :, :, :, :] = grad

        return grads

    def _get_samples(self, img_t: torch.Tensor) -> list:
        """ yield the samples to analyse """
        raise NotImplementedError


class SmoothGrad(AveragingGradient):
    def __init__(self, backprop: ModifiedBackpropMethod, std=0.15, steps=50):
        super().__init__(backprop=backprop)
        self.std = std
        self.steps = steps

    def _get_samples(self, img_t: torch.Tensor) -> list:
        relative_std = (img_t.max().item() - img_t.min().item()) * self.std
        noises = [torch.randn(*img_t.shape).to(img_t.device) * relative_std for _ in range(0, self.steps)]
        noise_images = [img_t + noises[i] for i in range(0, self.steps)]
        return noise_images


class IntegratedGradients(AveragingGradient):

    def __init__(self, backprop: ModifiedBackpropMethod, baseline: Baseline = None, steps=50):
        """
        :param baseline: start point for interpolation (0-1 grey, or "inv", or "avg")
        :param steps: resolution
        """
        super().__init__(backprop=backprop)
        self.baseline = baseline if baseline is not None else Mean()
        self.steps = steps

    def _get_samples(self, img_t: torch.Tensor) -> np.array:
        bl_img = self.baseline.apply(to_np_img(img_t))
        bl_img_t = to_img_tensor(bl_img, device=img_t.device)
        return [((i / self.steps) * (img_t - bl_img_t) + bl_img_t) for i in range(1, self.steps + 1)]

