import abc

from attribution_bottleneck.attribution.base import HeatmapTransform
from attribution_bottleneck.utils.misc import *
from attribution_bottleneck.utils.misc import to_np_img


class Plotter:
    """
    anything that takes a heatmap and paints it, optionally together with an input image
    """
    @abc.abstractmethod
    def plot(self, hmap: np.array, fig, ax, img: np.array=None):
        pass

    def show_plot(self, hmap: np.array, img: np.array=None):
        fig, ax = plt.figure(), plt.gca()
        self.plot(hmap, fig, ax, img)
        plt.show()

    @staticmethod
    def put_scale(hmap, vmin=0.0, vmax=1.0):
        """
        show small boxes in the corner to show the colors of min/max values
        """
        tile = int(np.round(hmap.shape[0] / 20))
        hmap[0:tile, 0:tile] = vmax
        hmap[0:tile, tile:2 * tile] = vmin
        return hmap

    def combine(self, img, overlay, overlay_alpha):
        """
        combine two images in a certain ratio
        :param overlay_alpha: either a number (0-1) or a full-resolution alpha map
        """
        # overlay_alpha: same dims or scalar
        combined = img.copy()  # dont override img
        if len(overlay.shape) - len(img.shape) == -1:
            # add color channel
            overlay = np.stack((overlay, overlay, overlay), axis=2)
        for i in range(img.shape[2]):
            combined[:,:,i] = img[:,:,i] * (1-overlay_alpha) + overlay[:,:,i] * overlay_alpha
        return combined

class ColorPlotter(Plotter):
    """
    linear color plotter, optionally with image underlay
    """
    def __init__(self, img_factor=0.0):
        self.img_factor = img_factor
        self.cmap = plt.cm.RdBu_r

    def plot(self, hmap: np.array, fig, ax, img: np.array=None):
        # TODO show negative and positive contribution!
        hmap = HeatmapTransform.fit(hmap)
        if img is None:
            if self.img_factor != 0:
                raise RuntimeError("img_factor != 0, but no image given!")
            overlay = hmap
        else:
            overlay = self.combine(img, hmap, 1 - self.img_factor)
        im = ax.imshow(overlay.astype(int), cmap=self.cmap, vmin=0, vmax=1)
        return im

class CamOverlayPlotter(Plotter):
    """
    plotter like in the grad-cam paper. minima are visually supressed
    # TODO add transparency to the color bar
    """
    def __init__(self, min_alpha=0.2,  max_alpha=0.7, gray=True):
        self.max_alpha = max_alpha
        self.min_alpha = min_alpha
        self.gray = gray

    def plot(self, hmap: np.array, fig, ax, img: np.array = None):
        assert len(hmap.shape) == 2, "hmap has not the right shape: {}".format(hmap.shape)

        hmap = HeatmapTransform.set_absmax(hmap.astype(float))  # in [-1,1]
        im = ax.imshow(hmap, vmin=-1, vmax=1, cmap="jet")

        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # cbar = plt.colorbar(im, cax=cax, ticks=[-1,0,1])
        # cbar.set_ticklabels(["min", "0", "max"])

        if img is not None:
            hmap_alpha = np.abs(hmap)  # from 0 to 1
            hmap_alpha = np.minimum(self.max_alpha, hmap_alpha)  # at most max_alpha
            hmap_alpha = np.maximum(self.min_alpha, hmap_alpha)  # at least min_alpha
            if self.gray and len(img.shape) == 3:
                img = img.mean(2, keepdims=True)
            img = to_rgb(img)  # just ensure 3 channels
            img_final = np.concatenate([img, 1-hmap_alpha[...,np.newaxis]], axis=2)
            ax.imshow(img_final)
        else:
            pass

        return im

class GridShower:
    """
    takes a list of methods and a plotter, then evaluates an input image with these methods and plots them in a grid
    """
    def __init__(self, model: nn.Module, methods: dict, plotter: Plotter, label_provider=None):
        self.methods = methods
        self.plotter = plotter
        self.model = model
        self.label_provider = label_provider
        self.cols = 4
        self.rows = int(np.ceil((len(methods) + 1) / self.cols))

    def show_fig(self, sample, eval_class=None, blur=None):
        fig, axes = plt.subplots(self.rows, self.cols, figsize=(22, 4.5 * self.rows), dpi=80, sharex=True, sharey=True)
        axes = axes.flatten()

        img_t, target_t = sample

        # Show real img in first cell
        np_img = to_np_img(img_t, denorm=True)
        label_id = int(to_np(target_t))
        label = self.label_provider.get_label(label_id) if self.label_provider is not None else label_id
        axes[0].imshow(np_img, vmin=0, vmax=1)
        axes[0].set_title("Input: '{}'".format(label))

        # Calc heatmaps and show them
        hmaps = []
        ims = []
        for i, (name, method) in enumerate(self.methods.items()):
            result_label = torch.LongTensor(eval_class) if eval_class else target_t
            heatmap = method.heatmap(input_t=img_t, target_t=result_label)
            if blur:
                from scipy.ndimage.filters import gaussian_filter
                heatmap = gaussian_filter(heatmap, sigma=blur)

            # Show heatmap
            ax = axes[i + 1]
            im = self.plotter.plot(heatmap, fig, ax, np_img)
            ims.append(im)
            ax.set_title(name)
            hmaps.append(heatmap)

        # Show "empty" on the rest of plots
        for i in range(len(self.methods), len(axes) - 1):
            ax = axes[i+1]
            show_img(img=np.ones_like(np_img), title="empty", place=ax)
            ax.text(0.5, 0.5, 'empty',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=20, color='grey',
                    transform=ax.transAxes)

        # cax, kw = mpl.colorbar.make_axes([ax for ax in axes])
        # fig.colorbar(ims[0], cax=cax, pad=0.02, shrink=0.225)

        shrink = 0.913 if self.rows == 1 else 0.999
        cbar = fig.colorbar(ims[0], ax=axes.tolist(), pad=0.02, shrink=shrink, ticks=[-1,0,1])
        cbar.set_ticklabels(["min", "0", "max"])
        plt.show()

        # adjust imgs
        plt.show()
        return hmaps
