
import numpy as np

class HeatmapTransform:

    def __call__(self, hmap: np.ndarray):
        return self.apply_transform(hmap)

    def apply_transform(self, hmap: np.ndarray):
        raise NotImplementedError

    @staticmethod
    def set_integral(hmap, integral=1):
        """ rescale the map so that its sum equals a value"""
        return hmap / hmap.sum() * integral

    @staticmethod
    def set_absmax(hmap, absmax=1.0):
        """ set the maximal amplitude below or above zero """
        current_absmax = max(abs(hmap.max()), abs(hmap.min()))
        if current_absmax:  # dont divide by 0
            scale = 1 / current_absmax * absmax
            return hmap * scale
        else:
            print("WARNING current_absmax == 0! mean {}, max {}, min {}, std {}", hmap.mean(), hmap.max(), hmap.min())
            return hmap

    @staticmethod
    def crop_percentile(hmap: np.ndarray, bottom_p, top_p):
        return np.maximum(np.minimum(np.percentile(hmap, top_p), hmap), np.percentile(hmap, bottom_p))

    @staticmethod
    def fit(hmap):
        """ fit to [0,1] : put min input on 0 output, put max input on 1 output """
        positive = hmap - hmap.min()
        if positive.max() == 0:
            print("Warning: heatmap is uniform, mean {}, std {}".format(hmap.mean(), hmap.std()))
        return positive / positive.max()

    @staticmethod
    def to_index_map(hmap):
        """ return a heatmap, in which every pixel has its value-index as value """
        order_map = np.zeros_like(hmap)
        for i, idx in enumerate(HeatmapTransform.to_index_list(hmap)):
            order_map[idx] = -i
        return order_map

    @staticmethod
    def to_index_list(hmap, reverse=False):
        """ return the list of indices that would sort this map - highest pixel first, lowest last """
        order = np.argsort((hmap if reverse else -hmap).ravel())
        idxes = np.unravel_index(order, hmap.shape)  # array of two tuples
        idxes = np.transpose(np.stack(idxes))  # np.ndarray of sorted indices: 2x824358...
        idxes = [tuple(i) for i in np.stack(idxes)]  # TODO double np.stack ???#
        return idxes

class MaxMagnitude(HeatmapTransform):
    """ at each pixel, take the value of the color with the highest absolute value """
    def apply_transform2(self, hmap: np.ndarray):
        """ OLD """
        max_color = hmap.argmax(axis=-3)
        new_maps = np.empty(hmap.shape[:-1])
        s = new_maps.shape
        for r in range(s[-2]):
            for c in range(s[-1]):
                new_maps[..., r, c] = hmap[..., r, c, max_color[..., r, c]]
        maps = new_maps

    def apply_transform(self, hmap: np.ndarray):
        sign = np.ones_like(hmap)
        sign[hmap < 0] = -1
        hmap = np.abs(hmap)
        hmap = np.max(hmap, axis=-1)
        return hmap * sign


class Compose(HeatmapTransform):

    def __init__(self, *transforms):
        self.transforms = transforms

    def apply_transform(self, hmap: np.ndarray):
        for transform in self.transforms:
            hmap = transform(hmap)
        return hmap


class Max(HeatmapTransform):
    def apply_transform(self, hmap: np.ndarray):
        return self.method(hmap, *self.args, **self.kwargs)


class Functional(HeatmapTransform):
    def __init__(self, method, *args, **kwargs):
        self.method = method
        self.args = args
        self.kwargs = kwargs

    def apply_transform(self, hmap: np.ndarray):
        return self.method(hmap, *self.args, **self.kwargs)


class CropPercentile(Functional):
    def __init__(self, *args, **kwargs):
        super().__init__(self.crop_percentile, *args, **kwargs)


class SetInterval(Functional):
    def __init__(self, *args, **kwargs):
        super().__init__(self.set_integral, *args, **kwargs)
