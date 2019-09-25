import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm

from attribution_bottleneck.bottleneck.estimator import ReluEstimator
from attribution_bottleneck.attribution.base import AttributionMethod
from attribution_bottleneck.bottleneck.per_sample_bottleneck import PerSampleBottleneck
from attribution_bottleneck.bottleneck.estimator import Estimator
from attribution_bottleneck.utils.misc import resize, replace_layer, to_np


class PerSampleBottleneckReader(AttributionMethod):
    def __init__(self, model, estim: Estimator, beta=10, steps=10, lr=1, batch_size=10,
                 sigma=1, progbar=False):
        self.model = model
        self.original_layer = estim.get_layer()
        self.shape = estim.shape()
        self.beta = beta
        self.batch_size = batch_size
        self.progbar = progbar
        self.device = list(model.parameters())[0].device
        self.lr = lr
        self.train_steps = steps
        self.bottleneck = PerSampleBottleneck(estim.mean(), estim.std(), device=self.device,
                                              sigma=sigma, relu=isinstance(estim, ReluEstimator))
        self.sequential = nn.Sequential(self.original_layer, self.bottleneck)

    def heatmap(self, input_t, target):
        target_t = torch.tensor([target]) if not isinstance(target, torch.Tensor) else target
        self._run_training(input_t, target_t)
        return self._current_heatmap(shape=input_t.shape[2:])

    def _run_training(self, input_t, target_t):
        # Attach layer and train the bottleneck
        replace_layer(self.model, self.original_layer, self.sequential)
        self._train_bottleneck(input_t, target_t)
        replace_layer(self.model, self.sequential, self.original_layer)

    def _current_heatmap(self, shape=None):
        # Read bottleneck
        heatmap = self.bottleneck.buffer_capacity
        heatmap = to_np(heatmap[0])
        heatmap = heatmap.sum(axis=0)  # Sum over channel dim
        heatmap = heatmap - heatmap.min()  # min=0
        heatmap = heatmap / heatmap.max()  # max=0

        if shape is not None:
            heatmap = resize(heatmap, shape)

        return heatmap

    def _train_bottleneck(self, input_t: torch.Tensor, target_t: torch.Tensor):

        assert input_t.shape[0] == 1, "We can only fit on one sample"
        assert target_t.shape[0] == 1, "We can only fit on one label"

        batch = input_t.expand(self.batch_size, -1, -1, -1), target_t.expand(self.batch_size)
        optimizer = torch.optim.Adam(lr=self.lr, params=self.bottleneck.parameters())

        # Reset from previous run or modifications
        self.bottleneck.reset_alpha()

        # Train
        self.model.eval()
        for _ in tqdm(range(self.train_steps), desc="Training Bottleneck",
                      disable=not self.progbar):
            optimizer.zero_grad()
            out = self.model(batch[0])
            loss_t = self.calc_loss(outputs=out, labels=batch[1])
            loss_t.backward()
            optimizer.step(closure=None)

    def calc_loss(self, outputs, labels):
        """ Calculate the combined loss expression for optimization of lambda """
        information_loss = self.bottleneck.buffer_capacity.mean()  # Taking the mean is equivalent of scaling with 1/K
        cross_entropy = F.cross_entropy(outputs, target=labels)
        total = cross_entropy + self.beta * information_loss
        return total
