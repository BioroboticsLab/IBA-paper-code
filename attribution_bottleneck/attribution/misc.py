from ..attribution.base import AttributionMethod
import numpy as np


class Random(AttributionMethod):
    """ random heatmap from -1 to 1, no duplicate values. sum ~ 0 """
    def __init__(self, vmin=-1, vmax=1):
        self.vmin = vmin
        self.vmax = vmax

    def heatmap(self, input_t, target_t):
        shape = input_t[0, 0].shape
        hmap = np.linspace(start=self.vmin, stop=self.vmax, num=shape[0] * shape[1])
        np.random.shuffle(hmap)
        hmap = hmap.reshape((shape[0], shape[1]))
        return hmap


class Zero(AttributionMethod):
    """ random heatmap from -1 to 1, no duplicate values. sum ~ 0 """

    def heatmap(self, input_t, target_t):
        shape = input_t[0, 0].shape
        hmap = np.zeros((shape[0], shape[1]))
        return hmap


