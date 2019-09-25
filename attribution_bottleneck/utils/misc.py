
import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.axes import Axes
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


def replace_layer(model: nn.Module, target: nn.Module, replacement: nn.Module):
    """
    Replace a given module within a parent module with some third module
    Useful for injecting new layers in an existing model.
    """
    def replace_in(model: nn.Module, target: nn.Module, replacement: nn.Module):
        # print("searching ", model.__class__.__name__)
        for name, submodule in model.named_children():
            # print("is it member?", name, submodule == target)
            if submodule == target:
                # we found it!
                if isinstance(model, nn.ModuleList):
                    # replace in module list
                    model[name] = replacement

                elif isinstance(model, nn.Sequential):
                    # replace in sequential layer
                    model[int(name)] = replacement
                else:
                    # replace as member
                    model.__setattr__(name, replacement)

                # print("Replaced " + target.__class__.__name__ + " with "+replacement.__class__.__name__+" in " + model.__class__.__name__)
                return True

            elif len(list(submodule.named_children())) > 0:
                # print("Browsing {} children...".format(len(list(submodule.named_children()))))
                if replace_in(submodule, target, replacement):
                    return True
        return False

    if not replace_in(model, target, replacement):
        raise RuntimeError("Cannot substitute layer: Layer of type " + target.__class__.__name__ + " is not a child of given parent of type " + model.__class__.__name__)

def resize(arr, shape, interp="bilinear"):
    if interp == "nearest":
        interp = cv2.INTER_NEAREST
    elif interp == "bilinear" or interp == "linear":
        interp = cv2.INTER_LINEAR
    else:
        raise ValueError(interp)
    return cv2.resize(arr, dsize=shape, interpolation=interp)

def mono_to_rgb(img):
    if len(img.shape) == 2:
        return np.stack((img, img, img), axis=2)
    elif img.shape[2] == 1:
        return np.dstack((img, img, img))
    else:
        # nothing to do
        return img

def show_img(img, title="", place=None):
    img = to_np_img(img)
    if place is None:
        place = plt
    try:
        if len(img.shape) == 3 and img.shape[2] == 1:
            # remove single grey channel
            img = img[...,0]

        if len(img.shape) == 2:
            place.imshow(img, cmap="Greys_r")
        else:
            place.imshow(img)
    except TypeError:
        print("type error: shape is {}".format(img.shape))
        raise TypeError

    if not isinstance(place, Axes):
        place.title(title)
        plt.show()
    else:
        place.set_title(title)

def prepare_image(img):
    return Compose([
        Resize(224),
        CenterCrop((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])(img).unsqueeze(0)

def normalize_img(img: np.ndarray):
    img = img - np.min(img)
    img = img / max(np.max(img), 0.001)
    return img

def chw_to_hwc(img: np.ndarray):
    return np.swapaxes(np.swapaxes(img, 2, 0), 1, 0)

def hwc_to_chw(img: np.ndarray):
    return np.swapaxes(np.swapaxes(img, 1, 0), 2, 0)

def to_img_tensor(img, device=None):

    # add color channel
    if len(img.shape) == 2:
        img = np.stack((img, img, img), axis=2)

    # add batch dimension 1
    if len(img.shape) == 3:
        img = np.expand_dims(img, 0)

    # move color channel to front
    img = np.swapaxes(np.swapaxes(img, 2, 1), 3, 1)

    if device is not None:
        t = torch.tensor(img, device=device)
    else:
        t = torch.from_numpy(img)

    return t

def denormalize(img: np.ndarray):
    img = img - img.min()  # force min 0
    img = img / np.max(img)  # force max 1
    return img

    mean3 = [0.485, 0.456, 0.406]
    std3 = [0.229, 0.224, 0.225]
    mean1 = [0.5]
    std1 = [0.5]
    mean, std = (mean3, std3) if img.shape[2] == 3 else (mean1, std1)
    for d in range(len(mean)):
        img[:, :, d] += mean[d]
        if np.max(img) > 1:
            img = img / np.max(img)  # force max 1
    return img

def to_np_img(img: torch.Tensor, denorm=False):

    # force 2-3 dims
    if len(img.shape) == 4:
        img = img[0]

    # tensor to np
    if isinstance(img, torch.Tensor):
        img = img.detach()
        if img.is_cuda:
            img = img.cpu()
        img = img.numpy()

    # if color is not last
    if len(img.shape) > 2 and img.shape[0] < img.shape[2]:
        img = np.swapaxes(np.swapaxes(img, 2, 0), 1, 0)

    if denorm:
        img = denormalize(img)

    return img

def to_np(t: torch.Tensor):
    t = t.detach()
    if t.is_cuda:
        t = t.cpu()
    return t.numpy()

def make_batch_list(inp):
    if isinstance(inp, torch.Tensor):
        inp = inp, None  # pseudo-tuple
    if not isinstance(inp, list):
        inp = [inp]
    assert isinstance(inp, list), type(inp)
    assert isinstance(inp[0], tuple), type(inp[0])
    assert isinstance(inp[0][0], torch.Tensor), type(inp[0][0])
    return inp

def analyze_img(img: np.array):
    return "img {}: \n   min={:04f}, \n   max={:04f}, \n   mean={:04f}, \n   std={:04f}".format(img.shape, img.min(), img.max(), img.mean(), img.std())


def reset_layers(layers):

    def init(m):
        if isinstance(m, nn.Conv2d):
            print("reinit " + m.__class__.__name__)
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            print("reinit " + m.__class__.__name__)
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif len(m.parameters()) > 0:
            print("Warning: no init for "+m.__class__.__name__)

    for l in layers:
        l.apply(init)

def grad_only(model, layer_s):
    """ set requires_grad to true only for the given layers """
    layers = layer_s if isinstance(layer_s, (list, tuple)) else [layer_s]

    def grad_on(m):
        for p in m.parameters():
            p.requires_grad = True

    def grad_off(m):
        for p in m.parameters():
            p.requires_grad = False

    # all off
    model.apply(grad_off)
    for l in layers:
        l.apply(grad_on)

def toggle_eval(self, classe_s, val):
    """ toggle eval mode to val for every layer of certain types """
    classes = classe_s if isinstance(classe_s, tuple) else classe_s,

    def toggle(m):
        if isinstance(m, classes):
            print("putting "+m.__class__.__name__+" to "+("eval" if val else "train"))
            if val:
                m.eval()
            else:
                m.train()

    self.setup.model.apply(toggle)


def call_batched(model, tensor, batch_size):
    n = len(tensor)
    k = n
    outs = []
    for i in range(n // batch_size + 1):
        b = i*batch_size
        e = min((i+1)*batch_size, len(tensor))
        if b == e:
            break
        outs.append(model(tensor[b:e]))
    return torch.cat(outs, 0)