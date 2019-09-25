from attribution_bottleneck.attribution.backprop import Gradient, GradientTimesInput
from attribution_bottleneck.attribution.guided_backprop import GuidedBackprop, DeconvNet
from attribution_bottleneck.attribution.averaging_gradient import IntegratedGradients, SmoothGrad
from attribution_bottleneck.attribution.grad_cam import GradCAM, GuidedGradCAM
from attribution_bottleneck.attribution.occlusion import Occlusion
from attribution_bottleneck.attribution.lrp import LRP
from attribution_bottleneck.attribution.misc import Random, Zero
from attribution_bottleneck.utils.baselines import Mean
import torch
from torch.nn import Softmax


class Factory:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.device = next(model.parameters()).device
        # Check if the given model's last layer are the logits
        for m in model.modules():
            if isinstance(m, Softmax):
                raise RuntimeError("model has to be pre-softmax!")

    def Random(self):
        return Random()

    def Zero(self):
        return Zero()

    def Gradient(self):
        return Gradient(self.model)

    def GradientTimesInput(self):
        """ https://arxiv.org/abs/1605.01713 """
        return GradientTimesInput(self.model)

    def PatternAttribution(self):
        from attribution_bottleneck.attribution.pattern import PatternAttribution
        assert hasattr(self.model, "features") and hasattr(self.model, "classifier"), \
            "PatternAttribution requires VGG"
        return PatternAttribution(self.model)

    def LRP(self):
        return LRP(self.model, eps=5, beta=-1, device=self.device)

    def Saliency(self):
        """ https://arxiv.org/abs/1312.6034 """
        return Gradient(self.model)

    def GuidedBackprop(self):
        """ https://arxiv.org/abs/1412.6806 """
        return GuidedBackprop(self.model)

    def DeconvNet(self):
        """ TODO CITATION """
        return DeconvNet(self.model)

    def IntegratedGradients(self):
        """ https://arxiv.org/abs/1703.01365 """
        # todo ist nicht gut, etwas fehlt wohl
        return IntegratedGradients(Gradient(self.model), baseline=Mean(), steps=50)

    def Occlusion(self, patch_size):
        """ TODO CITATION """
        return Occlusion(self.model, size=patch_size, baseline=Mean())

    def SmoothGrad(self):
        """ https://arxiv.org/abs/1706.03825 """
        # welche transforms?
        return SmoothGrad(Gradient(self.model), std=0.15, steps=50)

    def GradCAM(self, layer):
        """ https://arxiv.org/abs/1610.02391 """
        return GradCAM(self.model, layer=layer, interp="bilinear")

    def GuidedGradCAM(self, gradcam_layer):
        """ https://arxiv.org/abs/1412.6806 """
        return GuidedGradCAM(self.model, gradcam_layer=gradcam_layer, gradcam_interp='bilinear')
