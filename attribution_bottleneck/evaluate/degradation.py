from collections import OrderedDict
from datetime import datetime
import time
from math import sqrt
import numpy as np
from tqdm import tqdm

from attribution_bottleneck.attribution.base import *
from attribution_bottleneck.evaluate.base import *
from attribution_bottleneck.evaluate.perturber import *
from attribution_bottleneck.utils.baselines import Mean
from attribution_bottleneck.utils.misc import *


class DegradationEval(Evaluation):
    """ Evaluates single heatmaps by perturbing the image on high rated regions """
    def __init__(self, model, n_steps=100, part=1., baseline=None, tile_size=None, reverse=False, eval_mode="probs"):
        self.model = model
        self.show_original = False
        self.show_heatmap = False
        self.show_order = False
        self.show_baseline = False
        self.show_lowest_score = False
        self.show_step = None
        self.progbar = False
        self.warn_baseline_prob = 0.05
        self.reverse = reverse
        self.part = part
        self.n_steps = n_steps
        self.tile_size = tile_size
        self.eval_mode = eval_mode
        self.baseline = baseline if baseline is not None else Mean()

        class SoftMaxWrapper(nn.Module):
            def __init__(self, logits):
                super().__init__()
                self.logits = logits
                self.softmax = nn.Softmax(dim=1)

            def forward(self, input):
                return self.softmax(self.logits(input))
        if self.eval_mode == "logits":
            pass
        elif self.eval_mode == "probs":
            self.model = SoftMaxWrapper(self.model)
        else:
            raise ValueError

    def eval(self, hmap: np.ndarray, img_t: torch.Tensor, target) -> dict:
        """
        Iteratively perturbe an image, measure classification drop, and return the path of ([num perturbations] -> [class drop]) tuples
        TODO do this all in torch - it may be faster
        """
        assert img_t.shape[0] == 1, "batch dimension has to be one - we analyze one input sample"
        assert img_t.shape[1] == 3, "RGB images required"
        assert isinstance(img_t, torch.Tensor), "img_t has to be a torch tensor"
        assert isinstance(hmap, np.ndarray), "the heatmap has to be a np.ndarray"
        assert hmap.shape == tuple(img_t[0,0].shape), "heatmap and image have to be the same size: {} != {}".format(hmap.shape, tuple(img_t[0,0].shape))

        self.model.eval()

        # construct the perturbed image
        img = to_np_img(img_t)
        baseline_img = self.baseline.apply(img)
        baseline_t = to_img_tensor(baseline_img, device=img_t.device)

        # calculate intial score and baseline
        with torch.no_grad():
            initial_out = self.eval_np(img_t)
        top1 = np.argmax(initial_out)
        initial_val = initial_out[top1]
        baseline_val = self.eval_np(baseline_t)[top1]
        if baseline_val > self.warn_baseline_prob:
            print("Warning: score is still {}".format(baseline_val))
            show_img(denormalize(baseline_img))

        # process heatmap
        perturber = PixelPerturber(img_t, baseline_img) if (self.tile_size is None or self.tile_size == (1, 1)) else GridPerturber(img_t, baseline_t, self.tile_size)
        idxes = perturber.get_idxes(hmap, reverse=self.reverse)

        # iterate with perturbing
        max_steps = len(idxes)
        do_steps = int(max_steps * self.part)
        parts = np.linspace(0, 1, self.n_steps)
        parts_int = [int(p) for p in np.round(parts*max_steps)]
        min_value = initial_val
        min_degraded_t = img_t

        parts = [0]
        perturbed_ts = [img_t]
        for step in tqdm(range(do_steps), desc="Perturbing", disable=not self.progbar):
            perturber.perturbe(*idxes[step])
            if step in parts_int:
                perturbed_ts.append(perturber.get_current().clone())


        perturbed_ts = torch.cat(perturbed_ts, 0)
        with torch.no_grad():
            model_results = to_np(self.model(perturbed_ts))


        # maybe show results
        if self.show_original:
            show_img(normalize_img(img), title="original image, score {}".format(initial_val))
        if self.show_baseline:
            show_img(normalize_img(baseline_img), title="baseline image, score {}".format(baseline_val))
        if self.show_heatmap:
            show_img(normalize_img(hmap), title="heatmap")
        # if self.show_heatmap and isinstance(perturber, GridPerturber):
        #     show_img(normalize_img(GridView(np.expand_dims(hmap, 2), *perturber.get_tile_shape()).get_tile_means()), title="heatmap grid")
        if self.show_order:
            order_map = np.zeros(perturber.get_grid_shape())
            for i, idx in enumerate(idxes):
                order_map[idx] = -i
            show_img(normalize_img(order_map), title="order of perturbation, w=first")
        if self.show_lowest_score:
            show_img(normalize_img(to_np_img(min_degraded_t)), title="lowest score of: {:.3f} ({:.3f})".format(min_value, min_value - initial_val))

        return {
            "degradation_target": model_results[:, target].astype(np.float32),
            "degradation_top1": model_results[:, top1].astype(np.float32),
            "percentage": np.array(parts).astype(np.float16),
            "initial_value": initial_out,
            "top1": top1,
            "target": target.cpu().numpy(),
        }

    def eval_np(self, img_t):
        """ pass the tensor through the network and return the scores as a numpyarray w/o batch dimension (1D shape)"""
        return to_np(self.model(img_t))[0]

class SensitivityN(Evaluation):
    pass

class Collector:
    """ Use a evaluator+samples to evaluate multiple methods """

    def __init__(self, evaluator, methods):
        self.evaluator = evaluator
        self.method_settings = methods
        self.show_heatmap = False
        self.progbar = True
        self.device = next(iter(evaluator.model.parameters())).device

    def make_eval(self, samples, n_samples=None):
        #assert isinstance(samples, list)
        #assert len(samples) > 0
        #assert isinstance(samples[0], tuple)
        #assert isinstance(samples[0][0], torch.Tensor)  # img_t
        #assert isinstance(samples[0][1], torch.Tensor)  # target_t

        # Collect results for each method
        total_ms = {name: 0.0 for name in self.method_settings}
        results = {name: [] for name in self.method_settings}
        for sample in tqdm(samples, desc="Evaluating", disable=not self.progbar, total=n_samples, ascii=True):
            for name, meth in self.method_settings.items():
                sample = sample[0].to(self.device), sample[1].to(self.device)
                start = time.time()
                hmap = meth.heatmap(sample[0], sample[1])
                total_ms[name] += time.time() - start
                if self.show_heatmap:
                    show_img(hmap)

                result = self.evaluator.eval(hmap, sample[0].clone(), sample[1])
                results[name].append(result)
        results_stacked = {}
        for name, method_results in results.items():
            stacked = {}
            for item in method_results[0].keys():
                stacked[item] = []
                for result in method_results:
                    stacked[item].append(result[item])
            results_stacked[name] = {item: np.stack(res) for item, res in stacked.items()}

        return {name: {
            "results": results_stacked[name],
            "time_ms": int(total_ms[name] * 1000.0 / len(results[name])),
        } for name in self.method_settings}


class GraphDrawer:
    """ Draws the evaluation of multiple methods over multiple images """

    def __init__(self, figsize=(18, 10)):
        self.figsize = figsize
        self.show_title = True
        self.show_times = False

    @staticmethod
    def integrals(results):
        diffs = {k: results[0][k]["paths"].mean(0) - results[1][k]["paths"].mean(0) for k in results[0]}
        return {k: diffs[k][:,1].mean() for k in diffs}

    def draw_fig(self, method_results: dict, ax=None, *args, **kwargs):

        if ax is None:
            plt.figure()
            _, ax = plt.subplots(figsize=self.figsize, dpi=80)

        self.draw(method_results, ax=ax, *args, **kwargs)
        n = next(iter(method_results.values()))["results"]['degradation'].shape[0]
        ax.set_ylabel("Drop in accuracy over {} samples".format(n))
        ax.set_xlabel("Proportion of input replaced")
        ax.set_title("Comparison over attribution methods for {} samples".format(n))
        #plt.show()

    def draw(self, method_results: dict, ax, mode="mean-std", colors=None, lines=None, labels=None, order=None):

        n = next(iter(method_results.values()))["results"]['degradation'].shape[0]

        # Make style arguments
        kwargs = [{} for _ in method_results]  # Initialize empty
        colors = colors if colors else (plt.get_cmap("tab10").colors if len(method_results) < 9 else plt.get_cmap("tab20").colors)  # Default color map
        if lines is not None:
            for i, _ in enumerate(method_results):
                kwargs[i]["linestyle"] = lines[i]
        if labels is not None:
            assert len(labels) == len(method_results)
            for i, name in enumerate(method_results):
                print(f"setting {name}  -> {labels[i]}")
                kwargs[i]["label"] = labels[i]
        else:
            for i, name in enumerate(method_results):
                kwargs[i]["label"] = name

        for kwarg in kwargs:
            kwarg["label"] = kwarg["label"].replace("_", "\_") if kwarg["label"] is not None else kwarg["label"]

        # Remove ignored lines
        if order is not None:
            keys = list(method_results.keys())
            vals = list(method_results.values())
            method_results = OrderedDict({keys[o]: vals[o] for o in order})
            kwargs = [kwargs[o] for o in order]

        # Assign colors
        if colors is not None:
            for i, _ in enumerate(method_results):
                kwargs[i]["color"] = colors[i]

        # Plot results
        # Paths are [sample, blur_steps, prob_steps]  (blur step is a bit redundant)
        if mode == "mean-std":
            for i, (name, result) in enumerate(method_results.items()):
                results = method_results[name]["results"]
                eval_point = np.argmax(results['initial_value'], axis=1)

                paths = np.stack([results['degradation'][i, :, e] for i, e in enumerate(eval_point)])
                print(eval_point.shape, results['degradation'].shape, paths.shape)
                m, s = paths.mean(axis=0), paths.std(axis=0) / sqrt(n)
                x = results['percentage'][0]
                p = ax.plot(x, m, **kwargs[i])[0]
                ax.fill_between(x, m + s, m - s, color=p.get_color(), alpha=0.2)

        elif mode == "mean":
            for i, (name, result) in enumerate(method_results.items()):
                paths = method_results[name]["paths"]
                xm = paths.mean(axis=0)
                m = xm[:,1]
                x = xm[:,0]
                ax.plot(x, m, **kwargs[i])

        elif mode == "all":
            colors = plt.cm.rainbow(np.linspace(0,1,len(method_results)))
            for meth_i, (name, result) in enumerate(method_results.items()):
                results = method_results[name]["results"]
                initial_val = np.max(results['initial_value'], axis=1)
                eval_point = np.argmax(results['initial_value'], axis=1)
                paths = np.stack([results['degradation'][i, :, e] for i, e in enumerate(eval_point)]) - initial_val[:, None]
                for path_i, (percentage, path) in enumerate(zip(results['percentage'], paths)):
                    kwargs[meth_i]["label"] = path_i # kwargs[meth_i]["label"] if path_i == 0 else None
                    kwargs[meth_i]["color"] = plt.cm.rainbow(np.linspace(0,1,len(paths)))[path_i]
                    #kwargs[meth_i]["color"] = colors[meth_i]

                    ax.plot(percentage, path, **kwargs[meth_i])
        else:
            raise ValueError

        ax.legend(prop={'size': 9}, loc=1)

        # print timing
        if self.show_times:
            for i, (name, result) in enumerate(method_results.items()):
                print("{}: {}ms".format(name, method_results[name]["time_ms"]))

