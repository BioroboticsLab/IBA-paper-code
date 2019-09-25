#!/usr/bin/env python
# coding: utf-8

import sys
import torch
from attribution_bottleneck.evaluate.script_utils import stream_samples, \
    get_model_and_attribution_method, get_default_config
from attribution_bottleneck.evaluate.degradation import DegradationEval, Collector
from time import strftime, gmtime
torch.backends.cudnn.benchmark = True

try:
    testing = (sys.argv[4] == 'test')
except IndexError:
    testing = False


if testing:
    print("testing run. reducing samples to 50!")
    n_samples = 1
else:
    n_samples = 50000

model_name = sys.argv[1]
patch_size = int(sys.argv[2])
attribution_name = sys.argv[3]

config = get_default_config()
config.update({
    'model_name': model_name,
    'attribution_name': attribution_name,
    'n_samples': n_samples,
})


dev = torch.device(config['device'])
print("Evaluation {} on model {} with patch size {}x{}:".format(attribution_name,
                                                                model_name, patch_size, patch_size))
print("config is:", config)


model, attribution, test_set = get_model_and_attribution_method(config)
model.eval()

t = patch_size
evaluations = {
    f"{t}{t}": DegradationEval(model, tile_size=(t, t)),
    f"{t}{t} reversed": DegradationEval(model, tile_size=(t, t), reverse=True),
}

result_list = []

for name, ev in evaluations.items():
    collector = Collector(ev, {attribution_name: attribution})
    data_gen = stream_samples(test_set, config['n_samples'])
    result_list.append(collector.make_eval(data_gen, config['n_samples']))


time = strftime("%m-%d_%H-%M-%S", gmtime())
if testing:
    fname = f"results/test_{model_name}_{attribution_name}_{t}x{t}_{time}.torch"
else:
    fname = f"results/{model_name}_{attribution_name}_{t}x{t}_{time}.torch"
torch.save(result_list, fname)
print("Saved:", fname)
