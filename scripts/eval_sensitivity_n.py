#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
from tqdm import tqdm
from attribution_bottleneck.evaluate.script_utils import get_model_and_attribution_method, \
    get_default_config
from attribution_bottleneck.evaluate.degradation import GridPerturber
from attribution_bottleneck.attribution.occlusion import Occlusion
from attribution_bottleneck.utils.baselines import ZeroBaseline
import sys
import os
import time
from datetime import datetime
import pprint

sys.setrecursionlimit(10000)
torch.backends.cudnn.benchmark = True

try:
    testing = (sys.argv[4] == 'test')
except IndexError:
    testing = False


if testing:
    n_samples = 10
    n_log_steps = 3
    print("testing run. reducing samples to", n_samples)
else:
    n_samples = 1000
    n_log_steps = 16

model_name = sys.argv[1]

assert model_name in ['resnet50', 'vgg16']
tile_size = int(sys.argv[2])
attribution_name = sys.argv[3]

config = get_default_config()
config.update({
    'model_name': model_name,
    'attribution_name': attribution_name,
    'n_samples': n_samples,
    'n_log_steps': n_log_steps,
    'n_different_indices': 100,
    'tile_size': tile_size,
    'testing': False,
    'result_dir': 'results/sensitivityn',
})

print('running sensitivity-n')
print("config:")
pp = pprint.PrettyPrinter()
pp.pprint(config)
print()
print()
# Setup net
dev = torch.device(config['device'])
print("Loading setup ", model_name)
print()
sys.stdout.flush()


start_time = time.time()
# Setup net

model, attribution, test_set = get_model_and_attribution_method(config)

if config['attribution_name'] == 'Occlusion-14x14':
    attribution = Occlusion(model, size=14, baseline=ZeroBaseline())
elif config['attribution_name'] == 'Occlusion-8x8':
    attribution = Occlusion(model, size=8, baseline=ZeroBaseline())

np.random.seed(config['seed'])
sample_idxs = np.random.choice(len(test_set), config['n_samples'], replace=False)

print("loading {} samples ...".format(len(np.unique(sample_idxs))))
samples = [test_set[i] for i in sample_idxs]
samples = [(img[None], torch.LongTensor([target])) for img, target in samples]


print("loading attribution: ", config['attribution_name'])
sys.stdout.flush()


def get_pertubation_indices(perturber, n_tiles):
    h, w = perturber.current.shape[-2:]
    idxs = perturber.get_idxes(torch.randn(h, w))
    return idxs[:n_tiles]


def get_masks(n_masks, img):
    masks = []
    h, w = img.shape[-2:]
    for _ in range(n_masks):
        if config['tile_size'] == 1:
            idxs = np.unravel_index(np.random.choice(h*w, n_tiles), (h, w))
            indices.append(idxs)
            mask = np.zeros((h, w))
            mask[idxs] = 1
            masks.append(torch.from_numpy(mask).float())
        else:
            perturber = GridPerturber(torch.zeros_like(img), torch.ones_like(img), (tdim, tdim))
            idxs = perturber.get_idxes(torch.randn(h, w))

            for idx in idxs[:n_tiles]:
                perturber.perturbe(*idx)
            mask = perturber.get_current()[0, 0].clone()
            masks.append(mask)
    return masks


heatmaps = []
for img, target in tqdm(samples, ascii=True, desc='computing heatmaps'):
    heatmaps.append(attribution.heatmap(img.to(dev), target.to(dev)))
heatmaps = torch.from_numpy(np.stack(heatmaps)).float()

if config['tile_size'] == 1:
    n_replacements = 224*224
else:
    img, _ = samples[0]
    h, w = img.shape[-2:]
    tdim = config['tile_size']
    perturber = GridPerturber(torch.zeros_like(img), torch.ones_like(img), (tdim, tdim))
    idx = perturber.get_idxes(torch.randn(h, w))
    n_replacements = len(idx)

n_tiles_eval_points = np.round(np.exp(np.linspace(
    np.log(1), np.log(0.8*n_replacements), num=config['n_log_steps']))).astype(np.int)
print("for tile size {} selected n: {}".format(config['tile_size'], n_tiles_eval_points))
sys.stdout.flush()
results = []

for n_tiles in n_tiles_eval_points:

    indices = []
    img, _ = samples[0]
    np.random.seed(config['seed'])
    masks = get_masks(config['n_different_indices'], img)
    h, w = img.shape[-2:]

    score_diffs = []
    sum_masked_relevance_all = []
    for (img, target), heatmap in tqdm(zip(samples, heatmaps), ascii=True, total=len(samples),
                                       desc='sensitivity-{}'.format(n_tiles)):
        pertubated_imgs = []
        sum_masked_relevance = []
        for mask in masks:
            b, c, h, w = img.shape
            img_mean = img.view(c, h*w).mean(1)
            mask = mask[None, None]
            pertubated_imgs.append(img * (1 - mask))
            sum_masked_relevance.append((heatmap * mask).sum())

        sum_masked_relevance = torch.stack(sum_masked_relevance)
        input_imgs = pertubated_imgs + [img]
        with torch.no_grad():
            input_imgs = torch.cat(input_imgs).to(dev)
            output = model(input_imgs)
        output_pertubated = output[:-1]
        output_clean = output[-1:]

        diff = output_clean[:, target] - output_pertubated[:, target]
        score_diffs.append(diff[:, 0].cpu().numpy())
        sum_masked_relevance_all.append(sum_masked_relevance.cpu().numpy())

    score_diff = np.stack(score_diffs)
    sum_masked_relevance = np.stack(sum_masked_relevance_all)
    corrcoef = np.corrcoef(sum_masked_relevance.flatten(), score_diff.flatten())
    results.append({
        'n_tiles': n_tiles,
        'score_diff': score_diff,
        'sum_masked_relevance': sum_masked_relevance,
        'corrcoef': corrcoef,
    })
    print('correlation for {}: {:.3f}'.format(n_tiles, corrcoef[1, 0]))
    sys.stdout.flush()


result_dir = config['result_dir']
os.makedirs(result_dir, exist_ok=True)

slurm_job_id = int(os.getenv("SLURM_JOB_ID", 0))
result_filename = "sensitivityn_{}_{}_{}_{}.torch".format(
    config['model_name'],
    config['attribution_name'].replace(" ", "_").replace(".", "_"),
    slurm_job_id,
    datetime.utcnow().isoformat()
)

output_filename = os.path.abspath(os.path.join(result_dir, result_filename))
torch.save({
    'config': config,
    'start_time': start_time,
    'end_time': time.time(),
    'slurm_job_id': slurm_job_id,
    'results': results,
    'sample_idxs': sample_idxs,
}, output_filename)


print('saved to: ', output_filename)
