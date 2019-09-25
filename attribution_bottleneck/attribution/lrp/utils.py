import torch
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split
import numpy as np


def pprint(*args):
    out = [str(argument) + "\n" for argument in args]
    print(*out, "\n")


class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, in_tensor):
        return in_tensor.view((in_tensor.size()[0], -1))


def scale_mask(mask, shape):

    if shape == mask.shape:
        print("No rescaling necessary.")
        return mask

    nmm_map = np.zeros(shape)
    print("Rescaling mask")
    for lbl_idx in np.unique(mask):
        nmm_map_lbl = mask.copy()
        nmm_map_lbl[lbl_idx != nmm_map_lbl] = 0
        nmm_map_lbl[lbl_idx == nmm_map_lbl] = 1
        zoomed_lbl = zoom(nmm_map_lbl, 1.5, order=3)
        zoomed_lbl[zoomed_lbl != 1] = 0
        remain_diff = np.array(nmm_map.shape) - np.array(zoomed_lbl.shape)
        pad_left = np.array(np.ceil(remain_diff / 2), dtype=int)
        pad_right = np.array(np.floor(remain_diff / 2), dtype=int)
        nmm_map[pad_left[0]:-pad_right[0], pad_left[1]:-pad_right[1], pad_left[2]:-pad_right[2]] += zoomed_lbl * lbl_idx

    return nmm_map


class Identity(torch.nn.Module):
    def forward(self, x):
        return x
