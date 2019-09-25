
from attribution_bottleneck.attribution.base import AttributionMethod
from attribution_bottleneck.attribution.guided_backprop import GuidedBackprop
import torch
# from scipy.misc import imresize
import numpy as np
from attribution_bottleneck.utils.misc import resize


class GradCAM(AttributionMethod):

    def __init__(self, model: torch.nn.Module, layer: torch.nn.Module, interp="bilinear"):
        """
        :param model: model containing the softmax-layer
        :param device: dev
        :param layer: evaluation layer - object or name or id
        """
        self.layer = layer
        self.model = model
        self.interp = interp
        self.grads = None
        self.probs = None
        self.eps = 1e-5

    def pass_through(self, img):
        self.model.eval()
        return

    def heatmap(self, input_t: torch.Tensor, target_t: torch.Tensor):
        shape = tuple(input_t[0, 0].shape)

        # feed in
        self.model.eval()

        fmaps, grads = None, None

        def hook_forward(module, input, output):
            nonlocal fmaps
            fmaps = output.detach()

        def hook_backward(module, grad_in, grad_out):
            nonlocal grads
            grads = grad_out[0].detach()

        # pass and collect activations + gradient of feature map
        forward_handle = self.layer.register_forward_hook(hook_forward)
        backward_handle = self.layer.register_backward_hook(hook_backward)
        self.model.zero_grad()
        preds = self.model(input_t)
        forward_handle.remove()
        backward_handle.remove()

        # calc grads
        grad_eval_point = torch.Tensor(1, preds.size()[-1]).zero_()
        grad_eval_point[0][preds.argmax().item()] = 1.0
        grad_eval_point = grad_eval_point.to(input_t.device)
        preds.backward(gradient=grad_eval_point, retain_graph=True)

        # weight maps
        maps = fmaps.detach().cpu().numpy()[0, ]
        weights = grads.detach().cpu().numpy().mean(axis=(2, 3))[0, :]

        # avg maps
        gcam = np.zeros(maps.shape[0:], dtype=np.float32)
        # sum up weighted fmaps
        for i, w in enumerate(weights):
            gcam += w * maps[i, :, :]

        # avg pool over feature maps
        gcam = np.mean(gcam, axis=0)
        # relu
        gcam = np.maximum(gcam, 0)
        # to input shape
        gcam = resize(gcam, shape, interp=self.interp)
        # rescale to max 1
        gcam = gcam / (gcam.max() + self.eps)

        return gcam


class GuidedGradCAM(AttributionMethod):
    def __init__(self, model: torch.nn.Module, gradcam_layer: torch.nn.Module,
                 gradcam_interp="nearest"):
        """
        :param model: model containing the softmax-layer
        :param device: dev
        :param layer: evaluation layer - object or name or id
        """
        self.grad_cam = GradCAM(model, gradcam_layer, gradcam_interp)
        self.guided_backprop = GuidedBackprop(model)

    def heatmap(self, input_t: torch.Tensor, target_t: torch.Tensor):
        gradcam_heatmap = self.grad_cam.heatmap(input_t, target_t)
        gbp_heatmap = self.guided_backprop.heatmap(input_t, target_t)
        return gradcam_heatmap * gbp_heatmap
