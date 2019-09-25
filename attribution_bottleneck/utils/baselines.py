from scipy.ndimage import gaussian_filter
import numpy as np
from attribution_bottleneck.evaluate.perturber import GridView, NumpyGridView
from ..utils.misc import resize


class Baseline:
    def apply(self, img):
        raise NotImplementedError

    def __call__(self, img):
        return self.apply(img)


class Blur(Baseline):
    """ Gaussian blur """
    def __init__(self, sigma=10):
        self.sigma = sigma

    def apply(self, img):
        baseline_img = np.empty_like(img)
        for c in range(img.shape[-1]):
            baseline_img[..., c] = gaussian_filter(img[..., c], sigma=self.sigma)
        return baseline_img


class Uniform(Baseline):
    """ Uniform color """
    def __init__(self, val):
        self.val = val

    def apply(self, img):
        return np.full_like(img, self.val)


class Minimum(Baseline):
    """ Minimal color """
    def apply(self, img):
        return np.full_like(img, img.min())


class Maximum(Baseline):
    """ Maximal color """
    def apply(self, img):
        return np.full_like(img, img.max())


class Shuffle(Baseline):
    """ Shuffling the pixels spatially """
    def apply(self, img):
        baseline_img = img.copy().reshape(-1, img.shape[-1])
        np.random.shuffle(baseline_img)
        return baseline_img.reshape(img.shape)


class ShuffleGrid(Baseline):
    """ Shuffling by tiles and the blurring the result """
    def __init__(self, sigma=10, tile_len=28):
        self.sigma = sigma
        self.tile_len = tile_len

    def apply(self, img):
        tile_size = (self.tile_len, self.tile_len)
        view = NumpyGridView(img, tile_size)
        shuffled = view.shuffled()
        blurred = Blur(self.sigma).apply(shuffled)
        return blurred


class Mean(Baseline):
    """ Mean of color channels """
    def apply(self, img):
        baseline_img = np.empty_like(img)
        for c in range(img.shape[-1]):
            baseline_img[...,c] = np.mean(img[...,c])
        return baseline_img


class ZeroBaseline(Baseline):
    def apply(self, img):
        return np.zeros_like(img)


class TileMean(Baseline):
    """ Tile-wise color mean """
    def __init__(self, tile_size):
        self.tile_size = tile_size

    def apply(self, img):
        tile_size = (self.tile_size, self.tile_size)
        view = GridView(img, tile_size)
        result = np.zeros_like(img)
        for r in range(view.tiles_r):
            for c in range(view.tiles_c):
                slc = view.tile_slice(r, c)
                result[slc[0], slc[1]] = img[slc].mean(axis=(0,1))
        return result


class LowlevelShuffle(Baseline):
    """ Shuffle a downsampled version of the image, then upsample it again """
    def apply(self, img):
        f = 10
        s = img.shape
        low = resize(img, (f, f))
        low_shuffled = Shuffle().apply(low)
        shuffled = resize(low_shuffled, (s[0], s[1]))
        return shuffled
