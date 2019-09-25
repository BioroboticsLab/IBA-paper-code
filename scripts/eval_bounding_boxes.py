#! /usr/bin/env python

import numpy as np
import torch
import pprint
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor

import xml.etree.ElementTree as ET
from tqdm import tqdm

from attribution_bottleneck.evaluate.script_utils import get_model_and_attribution_method, \
    get_default_config
import sys
import glob
import time
import os
from PIL import Image
from datetime import datetime
from attribution_bottleneck.utils.transforms import HeatmapTransform
sys.setrecursionlimit(10000)

start_time = time.time()

try:
    testing = (sys.argv[3] == 'test')
except IndexError:
    testing = False


if testing:
    print("testing run. reducing samples to 50!")
    n_samples = 500
else:
    n_samples = 50000

model_name = sys.argv[1]

assert model_name in ['resnet50', 'vgg16']
attribution_name = sys.argv[2]

config = get_default_config()
config.update({
    'model_name': model_name,
    'attribution_name': attribution_name,
    'n_samples': n_samples,
    'testing': testing,
    'result_dir': 'results/bbox',
    'min_bbox_ratio': 0.33,
})

print()
print("config:")
pp = pprint.PrettyPrinter()
pp.pprint(config)
print()
print()
# Setup net
dev = torch.device(config['device'])
print("Loading setup ", model_name)
print()


model, attribution, test_set = get_model_and_attribution_method(config)


def get_synset(filename):
    return filename.split('_')[0]


def get_image_full_filename(filename, train=True):
    synset = get_synset(filename)
    full_filename = os.path.join(config['imagenet_train'], synset, filename)
    return full_filename


def get_image(filename):
    return Image.open(get_image_full_filename(filename))


def scale_bbox(bbox, width, height):
    bbox_x_min = int(width * bbox[0])
    bbox_y_min = int(height * bbox[1])
    bbox_x_max = int(width * bbox[2])
    bbox_y_max = int(height * bbox[3])
    return bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max


def get_bbox_mask(image, bboxs):
    width, height = image.size
    mask = np.zeros((height, width), dtype=np.bool)
    for bbox in bboxs:
        xi, yi, xa, ya = scale_bbox(bbox, width, height)
        mask[yi:ya, xi:xa] = 1
    return mask


def parse_bbox_xml(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    root
    bboxs = []
    width = int(root.find('.size/.width').text)
    height = int(root.find('.size/.height').text)
    image_filename = root.find('.filename').text
    for obj in root.findall('.object'):
        xml_bbox = obj.find('.bndbox')
        xmin = int(xml_bbox.find('.xmin').text)
        xmax = int(xml_bbox.find('.xmax').text)
        ymin = int(xml_bbox.find('.ymin').text)
        ymax = int(xml_bbox.find('.ymax').text)
        bboxs.append([xmin / width, ymin / height, xmax / width, ymax / height])
    return image_filename, bboxs


def get_ration_top_in_bbox(mask, heatmap):
    heatmap_idxs = HeatmapTransform.to_index_map(heatmap).astype(np.int64)
    mask_np = mask > 0.5
    heatmap_bbox_idxs = heatmap_idxs.copy()
    heatmap_bbox_idxs[mask_np == 0] = heatmap_idxs.min()
    n_pixel_in_mask = mask_np.sum()
    return (heatmap_bbox_idxs > (-n_pixel_in_mask)).sum() / n_pixel_in_mask.sum()


def stream_imagenet_val_set_with_masks(image_dir, bbox_dir, if_obj_smaller=1, n_samples=50000):
    imagenet_transform = Compose([
        Resize(256),
        CenterCrop((224, 224)),
        ToTensor()
    ])

    image_filenames = sorted(glob.glob(os.path.join(image_dir, "*")))
    synnet_to_target = {name.split('/')[-1]: i for i, name in enumerate(image_filenames)}
    val_bbox_filenames = glob.glob(os.path.join(bbox_dir, "*.xml"))
    full_image_filename = glob.glob(os.path.join(image_dir, "**", "*.JPEG"), recursive=True)
    name_to_full_image_filename = {}
    for filename in full_image_filename:
        name = os.path.splitext(os.path.basename(filename))[0]
        name_to_full_image_filename[name] = filename

    for bbox_filename in sorted(val_bbox_filenames)[:n_samples]:
        image_name, bboxs = parse_bbox_xml(bbox_filename)
        synnet = name_to_full_image_filename[image_name].split('/')[-2]
        image = Image.open(name_to_full_image_filename[image_name]).convert('RGB')
        mask = get_bbox_mask(image, bboxs)

        mask_img = Image.fromarray(np.uint8(mask * 255))
        image_torch = imagenet_transform(image)
        mask_torch = imagenet_transform(mask_img)
        mask_ratio = (mask_torch.sum() / torch.ones_like(mask_torch).sum()).item()
        if mask_ratio <= if_obj_smaller and (mask_torch >= 0.5).sum() > 0:
            target = torch.LongTensor([synnet_to_target[synnet]])
            yield image_torch, mask_torch, target


mask_stream = stream_imagenet_val_set_with_masks(
    config['imagenet_test'], config['imagenet_test_bbox'],
    if_obj_smaller=config['min_bbox_ratio'], n_samples=n_samples)

ratio_attribution_in_mask = []
ratio_mask_to_image = []
ratio_top_in_bbox = []
progbar = tqdm(mask_stream, ascii=True)

for image, mask, target in progbar:
    heatmap = attribution.heatmap(image[None].to(dev), torch.LongTensor([target]).to(dev))
    ratio_attribution_in_mask.append(((heatmap * mask.numpy()).sum() / heatmap.sum()).item())
    ratio_mask_to_image.append((mask.sum() / torch.ones_like(mask).sum()).item())
    ratio_top_in_bbox.append(get_ration_top_in_bbox(mask[0].numpy(), heatmap))
    progbar.set_postfix(
        method=config['attribution_name'],
        ratio_attribution_in_mask=np.mean(ratio_attribution_in_mask),
        ratio_mask_to_image=np.mean(ratio_mask_to_image),
        ratio_top_in_bbox=np.mean(ratio_top_in_bbox),
    )
ratio_attribution_in_mask = np.array(ratio_attribution_in_mask)
ratio_mask_to_image = np.array(ratio_mask_to_image)
ratio_top_in_bbox = np.array(ratio_top_in_bbox)

result_dir = config['result_dir']
os.makedirs(result_dir, exist_ok=True)

slurm_job_id = int(os.getenv("SLURM_JOB_ID", 0))
result_filename = "bbox_{}_{}_{}_{}.torch".format(
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
    'ratio_attribution_in_mask': ratio_attribution_in_mask,
    'ratio_mask_to_image': ratio_mask_to_image,
    'ratio_top_in_bbox': ratio_top_in_bbox,
}, output_filename)

print("{}@{} with min bbox {}, {} images".format(
    config['attribution_name'], config['model_name'],
    config['min_bbox_ratio'], len(ratio_attribution_in_mask),
    ratio_attribution_in_mask.mean()))
print("ratio heatmap in bbox: {}".format(ratio_attribution_in_mask.mean()))
print("ratio top heatmap indicies in bbox: {}".format(ratio_top_in_bbox.mean()))
print("bbox covered: {}".format(ratio_mask_to_image.mean()))
print("results saved at: {}".format(output_filename))
