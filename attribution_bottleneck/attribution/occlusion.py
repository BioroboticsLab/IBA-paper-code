import numpy as np
from tqdm import tqdm
from attribution_bottleneck.attribution.base import AttributionMethod
from attribution_bottleneck.utils.baselines import Baseline
from attribution_bottleneck.utils.misc import to_np_img, to_img_tensor, resize, call_batched
import torch


class Occlusion(AttributionMethod):
    """
    slide a window over the input and measure the drop in score.
    a hmap value is the sum of the drop in all boxes that overlapped
    """
    def __init__(self, model, size, baseline: Baseline):
        self.model = model
        self.size = size
        self.stride = size  # TODO
        self.interp = "nearest"
        self.baseline = baseline
        self.progbar = False

    def heatmap(self, input_t: torch.Tensor, target_t: torch.Tensor):
        self.model.eval()

        # create baseline to get patches from
        baseline_t = to_img_tensor(self.baseline.apply(to_np_img(input_t)))

        # img: 1xCxNxN
        rows = input_t.shape[2]
        cols = input_t.shape[3]
        rsteps = 1 + int(np.floor((rows-self.size) / self.stride))
        csteps = 1 + int(np.floor((cols-self.size) / self.stride))
        steps = csteps * rsteps
        target = target_t.cpu().numpy()


        with torch.no_grad():
            initial_score = self.eval_np(input_t, target)
            hmap = np.zeros((rsteps, csteps))

            rstep_list = []
            cstep_list = []
            occluded = input_t.repeat(steps, 1, 1, 1).clone()
            for step in tqdm(range(steps), ncols=100, desc="calc score", disable=not self.progbar):

                # calc patch position
                cstep = step % csteps
                rstep = int((step - cstep) / csteps)
                r = rstep * self.stride
                c = cstep * self.stride
                assert((r + self.size) <= rows)
                assert((c + self.size) <= cols)

                # occlude
                # occluded.copy_(input_t)

                occluded[step, :, r:r + self.size, c:c + self.size] = baseline_t[0, :, r:r + self.size, c:c + self.size]
                rstep_list.append(rstep)
                cstep_list.append(cstep)

            # measure score drop
            #print(occluded.shape)

            score = call_batched(self.model, occluded, batch_size=100)[:, target]
            for i, (rstep, cstep) in enumerate(zip(rstep_list, cstep_list)):
                hmap[rstep, cstep] += initial_score - score[i]
            #for step in tqdm(range(steps), ncols=100, desc="calc score", disable=not self.progbar):

                # calc patch position
            #    cstep = step % csteps
            #    rstep = int((step - cstep) / csteps)
            #    r = rstep * self.stride
            #    c = cstep * self.stride
            #    assert((r + self.size) <= rows)
            #    assert((c + self.size) <= cols)

                # occlude
            #    occluded.copy_(input_t)
            #    occluded[0, :, r:r + self.size, c:c + self.size] = baseline_t[0, :, r:r + self.size, c:c + self.size]

                # measure score drop
            #    score = self.eval_np(occluded, target)
            #    hmap[rstep, cstep] += initial_score - score

        hmap = resize(hmap, (rows, cols), interp=self.interp)
        return hmap

    def eval_np(self, img_t, target):
        return self.model(img_t).squeeze()[target]

    def name(self):
        return "OC [{}]".format(self.baseline)
