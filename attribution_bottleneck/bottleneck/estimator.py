from math import inf
import torch
from tqdm import tqdm
from scipy.stats import norm
import numpy as np
import torch.nn as nn
from attribution_bottleneck.utils.misc import to_np


class Estimator:

    def get_layer(self):
        raise NotImplementedError

    def shape(self):
        """ Get the shape of mean and std tensors """
        raise NotImplementedError

    def mean(self):
        """ Get accumulated mean per cell """
        raise NotImplementedError

    def std(self, stabilize=True):
        """ Get accumulated standard deviation per cell """
        raise NotImplementedError

    def p_zero(self):
        """ Get ratio of activation equal to zero per cell """
        raise NotImplementedError


class GenericEstimator(Estimator):
    """
    Is fed single points of any-dimensional data and computes the running mean and std per cell.
    Useful to calculate the empirical mean and variance of intermediate feature maps.
    In the case of relu=True, the (virtual) mean is fixed at 0 and an additional ratio of zero-values per cell is kept.
    """
    def __init__(self, layer, relu):
        self.relu = relu
        self.layer = layer
        self.M = None  # running mean for each entry
        self.S = None  # running std for each entry
        self.N = None  # running num_seen for each entry
        self.num_seen = 0  # total samples seen
        self.eps = 1e-5

    def feed(self, z: np.ndarray):

        # Initialize if this is the first datapoint
        if self.N is None:
            self.M = np.zeros_like(z, dtype=float)
            self.S = np.zeros_like(z, dtype=float)
            self.N = np.zeros_like(z, dtype=float)

        self.num_seen += 1

        if self.relu:
            nz_idx = z.nonzero()
            self.N[nz_idx] += 1
            self.S[nz_idx] += z[nz_idx] * z[nz_idx]

        else:
            diff = (z - self.M)
            self.N += 1
            self.M += diff / self.num_seen
            self.S += diff * (z - self.M)

    def feed_batch(self, batch: np.ndarray):
        for point in batch:
            self.feed(point)

    def shape(self):
        return self.M.shape

    def is_complete(self):
        return self.num_seen > 0 and (not self.relu or np.all(self.N))

    def get_layer(self):
        return self.layer

    def mean(self):
        return self.M

    def p_zero(self):
        return 1 - self.N / (self.num_seen + 1)  # Adding 1 for stablility, so that p_zero > 0 everywhere

    def std(self, stabilize=True):
        if stabilize:
            # Add small numbers, so that dead neurons are not a problem
            return np.sqrt(np.maximum(self.S, self.eps) / np.maximum(self.N, 1.0))

        else:
            return np.sqrt(self.S / self.N)

    def estimate_density(self, z):
        z_norm = (z - self.mean()) / self.std()
        p = norm.pdf(z_norm, 0, 1)
        return p

    def normalize(self, z):
        return (z - self.mean()) / self.std()

    def load(self, what):
        state = what if not isinstance(what, str) else torch.load(what)
        # Check if estimator classes match
        if self.__class__.__name__ != state["class"]:
            raise RuntimeError("This Estimator is {}, cannot load {}".format(self.__class__.__name__, state["class"]))
        # Check if layer classes match
        if self.layer.__class__.__name__ != state["layer_class"]:
            raise RuntimeError("This Layer is {}, cannot load {}".format(self.layer.__class__.__name__, state["layer_class"]))
        self.N = state["N"]
        self.S = state["S"]
        self.M = state["M"]
        self.num_seen = state["num_seen"]


class ReluEstimator(GenericEstimator):
    def __init__(self, layer):
        super().__init__(layer, True)


class GaussianEstimator(GenericEstimator):
    def __init__(self, layer):
        super().__init__(layer, False)


class EstimatorGroup:
    """
    A wrapper for feeding data through estimators at the same time.
    This prevents unnecessary overhead when observing multiple layers at the same time
    """
    def __init__(self, model, estimators, data_gen=None):
        self.model = model
        self.estimators = estimators
        if data_gen:
            self.feed(data_gen)

    @staticmethod
    def auto(model, layers, data_gen=None):
        estimators = []
        for l in layers:
            if isinstance(l, nn.ReLU) or (isinstance(l, nn.Sequential) and isinstance(l[-1], nn.ReLU)):
                print("ReluEstimator for "+l.__class__.__name__)
                estimators.append(ReluEstimator(l))
            else:
                print("GaussianEstimator for "+l.__class__.__name__)
                estimators.append(GaussianEstimator(l))
        group = EstimatorGroup(model, estimators=estimators, data_gen=data_gen)
        return group

    def _make_feed_hook(self, i):
        def hook(m, x, z):
            self.estimators[i].feed_batch(to_np(z))
        return hook

    def feed(self, gen):
        print("Feeding estimator from generator...")
        hook_handles = [e.layer.register_forward_hook(self._make_feed_hook(i)) for i, e in enumerate(self.estimators)]

        for batch, labels in tqdm(gen):
            self.model(batch)

        for handle in hook_handles:
            handle.remove()

    def save(self, path):
        torch.save({
            "estimators": [{
                "class": e.__class__.__name__,
                "layer_class": e.layer.__class__.__name__,
                "N": e.N,
                "M": e.M,
                "S": e.S,
                "num_seen": e.num_seen,
            } for e in self.estimators]
        }, path)

    def load(self, what):
        state = what if not isinstance(what, str) else torch.load(what)
        assert len(self.estimators) == len(state["estimators"])
        for e, state in zip(self.estimators, state["estimators"]):
            e.load(state)

    def stds(self, stabilize=True):
        return [e.std(stabilize) for e in self.estimators]

    def means(self):
        return [e.mean()for e in self.estimators]

    def shapes(self):
        return [e.shape for e in self.estimators]

    def layers(self):
        return [e.layer for e in self.estimators]
