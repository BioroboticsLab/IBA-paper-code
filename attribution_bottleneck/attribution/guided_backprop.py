from torch.nn import ReLU

from ..attribution.backprop import *

class GuidedBackprop(ModifiedBackpropMethod):
    """
    Uses the modified backpropagation method to calculate the guided backprop
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_relu_acts = []  # list of forward activations, #0=first layer, #N=final layer
        self.deconv_instead = False

    def _transform_gradient(self, gradient: np.ndarray):
        return np.abs(gradient).max(axis=-1)

    def _hook_backward(self, module):
        def relu_backward_hook(module, grad_in, grad_out):
            """ Only pass back positive gradients who had positive activation. """
            if self.deconv_instead:
                # Only propagate POSITIVE gradients
                modified_grad_out = torch.clamp(grad_in[0], min=0.0)
            else:
                # Only propagate POSITIVE gradients with POSITIVE activations
                backward_mask = self.forward_relu_acts[-1]  # Negative atcs are 0 already b/c of ReLU
                backward_mask[backward_mask > 0] = 1  # where act is positive, set mask to 1
                modified_grad_out = backward_mask * torch.clamp(grad_in[0], min=0.0)

            del self.forward_relu_acts[-1]  # Forget last forward output

            return modified_grad_out,

        if isinstance(module, ReLU):
            return relu_backward_hook

    def _hook_forward(self, module):
        def relu_forward_hook(module, act_in, act_out):
            """ remember forward activations """
            self.forward_relu_acts.append(act_out)

        if isinstance(module, ReLU):
            return relu_forward_hook


class DeconvNet(GuidedBackprop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deconv_instead = True
