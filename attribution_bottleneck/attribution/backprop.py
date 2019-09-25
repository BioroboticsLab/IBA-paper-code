""" Base classes and implementations for backpropagation-based methods """

import torch
import numpy as np
from ..utils.misc import to_np_img
from ..utils.transforms import Compose, CropPercentile, SetInterval
from ..utils.misc import to_np
from ..attribution.base import AttributionMethod


class ModifiedBackpropMethod(AttributionMethod):
    """
    Base class for attribution methods based on a modified gradient rule.
    The model is modified with forward and backward hooks, then the input is passed and the gradient is calculated
    and post-processed to obtain a 2D attribution map. Finally, the model is restored to impact it for other usages.
    """

    def __init__(self, model):
        self.hooks = []
        self.model = model

    def _hook_forward(self, module):
        """ To be overridden if forward pass should be modified for attribution """
        return None

    def _hook_backward(self, module):
        """ To be overridden if backward pass should be modified for attribution """
        return None

    def _transform_gradient(self, gradient: np.ndarray):
        """ How the 4-axis gradient should be transformed to a 2-axis attribution map """
        raise NotImplementedError

    def __restore_model(self):
        """ Remove hooks to restore the original model """
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def __prepare_model(self):
        """ Attach hooks to modify the gradient flow """

        def hook_layer(module):
            """ Check if this layer should get modified """
            forward = self._hook_forward(module)
            if forward is not None:
                self.hooks.append(module.register_forward_hook(forward))
            backward = self._hook_backward(module)
            if forward is not None:
                self.hooks.append(module.register_backward_hook(backward))

        self.model.apply(hook_layer)

    def heatmap(self, input_t, target_t):
        """ call the implementation and then do formatting and postprocessing """

        # Calculate raw gradient on the input features
        self.model.eval()
        self.__prepare_model()
        grad_t = self._calc_gradient(input_t=input_t, target_t=target_t).detach()
        self.__restore_model()

        assert isinstance(grad_t, torch.Tensor)
        assert len(grad_t.shape) == 4
        assert grad_t.shape == tuple(input_t.shape), f"Backprop shape mismatch: {grad_t.shape} != {input_t.shape}"

        # Apply transforms to post-process the gradient and yield a 2D map
        grad = to_np_img(grad_t)
        heatmap = self._transform_gradient(grad)

        return heatmap

    def _calc_gradient(self, input_t: torch.Tensor, target_t: torch.Tensor):
        """ Calculate the gradient of the logits w.r.t. the input """

        # Pass input through the model
        self.model.zero_grad()  # Reset grad for recalculation
        img_var = torch.autograd.Variable(input_t, requires_grad=True)  # Store input as variable with autograd
        logits = self.model(img_var)  # Pass the input

        # Calculate gradient w.r.t target
        target_idx = target_t.item() if isinstance(target_t, torch.Tensor) else target_t
        grad_eval_point = torch.zeros(device=input_t.device, size=logits.shape)
        grad_eval_point[0][target_idx] = 1.0  # "One-hot"
        logits.backward(gradient=grad_eval_point)

        return img_var.grad


class Gradient(ModifiedBackpropMethod):
    """ Just regular gradient, by averaging the color channel values """
    def _transform_gradient(self, gradient: np.ndarray):
        return Compose(
            lambda x: x.mean(axis=-1),
            CropPercentile(0.5, 99.5),
            SetInterval(1),
        )(gradient)


class Saliency(ModifiedBackpropMethod):
    """ Pixel-wise maximal absolute color channel gradient """
    def _transform_gradient(self, gradient: np.ndarray):
        return Compose(
            lambda x: np.abs(x).max(axis=-1),
            CropPercentile(0.5, 99.5),
            SetInterval(1),
        )(gradient)


class GradientTimesInput(ModifiedBackpropMethod):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_input = None

    """ Gradient * Input """
    def heatmap(self, input_t, target_t):
        # Remember input for future usage
        self.last_input = to_np(input_t)
        return super().heatmap(input_t, target_t)

    def _transform_gradient(self, gradient: np.ndarray):
        return Compose(
            lambda x: x.mean(axis=-1),
            CropPercentile(0.5, 99.5),
            SetInterval(1),
        )(self.last_input * gradient)

