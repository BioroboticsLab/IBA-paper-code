import torch
import numpy as np


class AttributionMethod:
    """ Something than can make a attribution heatmap """

    def heatmap(self, input_t: torch.Tensor, target_t) -> np.ndarray:
        """
        Generate a attribution map. If the model is modified in the meantime, it is restored after the attribution.
        :param input_t: The input (image) as a BxHxWxC torch tensor with B=1. Already transformed to be passed to the model.
        :param target_t: A target label as 1x1 tensor, indicating the correct label. Alternatively an integer.
        :return: A HxW numpy array representing the resulting heatmap.
        """
        raise NotImplementedError

