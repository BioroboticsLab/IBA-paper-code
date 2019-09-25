import torch
import numpy as np
from IPython.display import display
from ipywidgets import interact, widgets, IntSlider

from ..utils.misc import to_np_img, show_img, to_np
import matplotlib.pyplot as plt


class TensorView:
    """ show a 3D-tensor in an ipython widget """
    def __init__(self, t, title="Tensor View", things: list = None):

        if len(t.shape) == 4:
            print("Warning: only showing sample 0 (batch dim is {})".format(t.shape[0]))
            t = t[0]

        self.handle = None
        self.things = things if things is not None else ["feature", "mean", "abs-mean", "hist"]
        self.t_np = t if isinstance(t, np.ndarray) else to_np(t)
        self.t_abs = np.abs(self.t_np)
        self.n_bins = 50
        self.title = title
        self.mean_2d = self.t_np.mean(axis=0)
        self.abs_2d = self.t_abs.mean(axis=0)

        self.hist, self.bins = np.histogram(self.t_np, bins=self.n_bins)
        self.abs_hist, self.abs_bins = np.histogram(self.t_abs, bins=self.n_bins)

        def show_layer(feature_map):

            # create plot
            # TODO only update it for better performance
            self.rows = 1
            self.axs = []
            plt.figure(figsize=(18, 4.5), dpi=80)
            for ti, thing in enumerate(self.things):
                ax = plt.subplot(1, 4, ti+1)
                self.axs.append(ax)

            # preprocess
            global_min = self.t_np.min()
            global_max = self.t_np.max()
            # HeatmapTransform.crop_percentile(np_tensor, top_p=99, bottom_p=1)
            feature_np = self.t_np[feature_map]

            for i, thing in enumerate(self.things):
                ax = self.axs[i]

                if thing == "feature":
                    # map of selected feature map
                    ax.imshow(to_np_img(feature_np), vmax=global_max, vmin=global_min, cmap="Greys_r")
                    ax.set_title(title + " - f-map {} \nsum: {:4f}\nmean: {:4f}".format(feature_map, feature_np.sum(), feature_np.mean()))

                elif thing == "f-hist":
                    # histogram of selected feature map
                    feature_hist, feature_bins = np.histogram(feature_np, bins=self.n_bins)
                    width = 0.7 * (feature_bins[1] - feature_bins[0])
                    center = (feature_bins[:-1] + feature_bins[1:]) / 2
                    ax.bar(center, feature_hist, align='center', width=width)
                    ax.set_title(title + " histogram of f-map {}".format(feature_map))

                elif thing == "f-sums":
                    # plot of sum of each feature map
                    means = [np.sum(self.t_np[feature_map]) for feature_map in range(self.t_np.shape[0])]
                    ax.plot(means)
                    ax.set_title(title + " f-map sums")

                elif thing == "mean":
                    # 2D-map: mean of feature maps
                    ax.imshow(self.mean_2d, vmax=self.mean_2d.max(), vmin=self.mean_2d.min(), cmap="Greys_r")
                    ax.set_title(title + " mean on f-axis \nmean sum per f-map: {:.4f} \nmean per neuron: {:.4f}".format(self.t_np.sum(),self.t_np.mean()))

                elif thing == "abs-mean":
                    # 2D-map: mean of absolute feature map values
                    ax.imshow(self.abs_2d, vmax=self.abs_2d.max(), vmin=self.abs_2d.min(), cmap="Greys_r")
                    ax.set_title(title + " mean abs values\nmean abs sum per f-map: {:.4f} \nmean abs per neuron: {:.4f}".format(self.t_abs.sum(axis=(1,2)).mean(),self.t_abs.mean()))

                elif thing == "hist":
                    # global histogram of activations
                    width = 0.7 * (self.bins[1] - self.bins[0])
                    center = (self.bins[:-1] + self.bins[1:]) / 2
                    ax.bar(center, self.hist, align='center', width=width)
                    ax.set_title(title + " global histogram\nmean: {:.4f}, std: {:.4f}\nmin: {:.4f}, max: {:.4f}".format(self.t_np.mean(), self.t_np.std(), self.t_np.min(), self.t_np.max()))

                elif thing == "abs-hist":
                    # global histogram of activations
                    width = 0.7 * (self.abs_bins[1] - self.abs_bins[0])
                    center = (self.abs_bins[:-1] + self.abs_bins[1:]) / 2
                    ax.bar(center, self.abs_hist, align='center', width=width)
                    ax.set_title(title + " abs vals histogram\nabs mean: {:.4f}\nabs std: {:.4f}".format(self.t_abs.mean(), self.t_abs.std()))

        self.handle = display(interact(show_layer, feature_map=IntSlider(value=0, min=0, max=self.t_np.shape[0]-1, continuous_update=False)))

