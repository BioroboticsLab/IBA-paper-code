import torch
from attribution_bottleneck.attribution.base import AttributionMethod
from attribution_bottleneck.attribution.lrp.utils import Identity
from attribution_bottleneck.attribution.lrp.innvestigator import InnvestigateModel
from attribution_bottleneck.utils.misc import to_np


class LRP(AttributionMethod):
    def __init__(self, model, eps=1, beta=0, method='b-rule', device=torch.device('cpu')):
        # Only works for vgg for now!
        self.model = model

        self.eps = eps
        self.beta = beta
        self.method = method
        self.device = device

    def heatmap(self, input_t, target):
        target_t = target if isinstance(target, torch.Tensor) else torch.tensor(target, device=input_t.device)

        assert input_t.shape[0] == 1, "We can only fit on one sample"
        assert target_t.shape[0] == 1, "We can only fit on one label"
        assert input_t.shape[-1] == 224, "with must be 224, otherwise avgpool is not the identity"
        avgpool = self.model.avgpool
        self.model.avgpool = Identity()

        lrp_model = InnvestigateModel(self.model, epsilon=self.eps,
                                      beta=self.beta, method=self.method)
        lrp_model.to(self.device)

        _, relevance = lrp_model.innvestigate(input_t, target_t)

        self.model.avgpool = avgpool

        heatmap = to_np(relevance[0]).sum(0)
        return heatmap
