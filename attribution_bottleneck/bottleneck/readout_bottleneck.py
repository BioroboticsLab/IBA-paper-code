import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

from ..bottleneck.gaussian_kernel import SpatialGaussianKernel
from ..bottleneck.per_sample_bottleneck import AttributionBottleneck
from ..utils.misc import replace_layer


class ReadoutBottleneck(AttributionBottleneck):
    """ a bottleneck which noises with emprical dists """
    def __init__(self, model, layers, means, stds, kernel_size=1, relu_op=False):
        super().__init__()
        self.model = [model]
        self.device = next(model.parameters()).device
        self.readout_layers = layers

        self.device = next(model.parameters()).device
        self.feat_out = means[0].shape[0]
        self.relu_op = relu_op
        self.feat_in = 0
        self.limit_value = 5.0
        self.store_buffers = False
        self.sigmoid = nn.Sigmoid()
        self._buffer_capacity = None

        self.input = None  # the last input tensor (image)
        self.observed_acts = []  # the last intermediate activations
        self.forward_hooks = []  # registered hooks
        self.input_hook = None  # registered hooks
        self.is_nested_pass = False
        self.active = True
        self.attach_hooks()

        for i, (mean, std) in enumerate(zip(means, stds)):
            self.register_buffer("std_{}".format(i), torch.tensor(std, dtype=torch.float, device=self.device, requires_grad=False))
            self.register_buffer("mean_{}".format(i), torch.tensor(mean, dtype=torch.float, device=self.device, requires_grad=False))
            self.feat_in += mean.shape[0]

        # Smoothing layer
        self.kernel_size = kernel_size
        if kernel_size is not None:
            # Construct static conv layer with gaussian kernel
            sigma = kernel_size * 0.25  # Cover 2 stds in both directions
            channels = self.std_0.shape[0]
            self.smooth = [SpatialGaussianKernel(kernel_size, sigma, channels, self.device)]
        else:
            self.smooth = None

    def attach_hooks(self):
        """ attach hooks """
        def forward_hook(m, t_in, t_out):
            if self.active and self.is_nested_pass:
                # print("Recording act with shape {} from {}".format(t_out.shape, m.__class__.__name__))
                self.observed_acts.append(t_out.clone())

        def input_hook(m, t_in):
            if self.active and not self.is_nested_pass:
                # print("Captured input: {}".format(t_in[0].shape))
                self.input = t_in[0].clone()

        # attach hooks to intermediate layers
        for m in self.readout_layers:
            self.forward_hooks.append(m.register_forward_hook(forward_hook))

        # attach hook to model
        self.input_hook = self.model[0].register_forward_pre_hook(input_hook)

    def detach_hooks(self):
        """ detach hooks """
        for h in self.forward_hooks:
            h.remove()

        self.input_hook.remove()

        self.forward_hooks = []
        self.input_hook = None

    def forward(self, x_in):

        if self.is_nested_pass:
            return x_in

        assert self.input is not None, "no input registered - activated?"

        # clear memory
        self.observed_acts = []

        with autograd.no_grad():
            # Pass input again and collect readout
            self.is_nested_pass = True
            self.model[0](self.input)
            self.is_nested_pass = False

        # Done with readout. Use it to obtain map
        return self.forward_augmented(x_in, self.observed_acts)

    def attach(self, model, layer):
        replace_layer(model, layer, nn.Sequential(layer, self))
        return self

    def forward_augmented(self, x, readouts):

        # Preprocess readout
        target_shape = x.shape[-2:]
        buffers = dict(self.named_buffers())
        readouts = [r / buffers["std_{}".format(i)] for i, r in enumerate(readouts)]
        readouts = [r.unsqueeze(-1).unsqueeze(-1).expand(*r.shape, *target_shape[-2:]) if len(r.shape) == 2 else r for r in readouts]
        readouts = [F.interpolate(input=r, size=target_shape, mode="bilinear", align_corners=True) for r in readouts]

        # Stack readout to one tensor
        readout = torch.cat(readouts, dim=1)

        # Pass through readout net to obtain mask
        alpha = self.infer_mask(readout)
        alpha = torch.clamp(alpha, -self.limit_value, self.limit_value)
        lamb = self.sigmoid(alpha)

        # Smoothing step
        lamb = self.smooth[0](lamb) if self.smooth is not None else lamb

        # Normalize x
        x_norm = (x - self.mean_0) / self.std_0
        mu, log_var = x_norm * lamb, torch.log(1-lamb)

        # Sampling step
        # Sample new output values from p(z|x)
        z_norm = self._sample_z(mu, log_var)
        self.buffer_capacity = self._calc_capacity(mu, log_var)

        # Denormalize x
        z = z_norm * self.std_0 + self.mean_0

        # Maybe relu
        if self.relu_op:
            z = torch.clamp(z, 0.0)

        return z

    def save(self, path):
        """ Save the state of the bottleneck to restore it later without an estimator rerun """
        state = {
            "model_state": self.state_dict(),
            "shapes": self.shapes(),
            "kernel_size": self.kernel_size,
        }
        return torch.save(state, path)

    @classmethod
    def load(cls, model, layers, layer_state, shapes, kernel_size):
        means = [np.zeros(s) for s in shapes]
        stds = [np.ones(s) for s in shapes]
        # Initialize with dummy mean/stds
        bottleneck = cls(model, layers, means, stds, kernel_size).to(list(model.parameters())[0].device)
        # Override with saved state
        bottleneck.load_state_dict(layer_state)
        return bottleneck

    @classmethod
    def load_path(cls, model, layers, path):
        """ Load a bottleneck from a file """
        device = next(model.parameters()).device
        state = torch.load(path, map_location=device)
        return cls.load(model, layers, state["model_state"], state["shapes"], state["kernel_size"])

    def infer_mask(self, readout):
        raise NotImplementedError


class ReadoutBottleneckA(ReadoutBottleneck):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(in_channels=self.feat_in, out_channels=self.feat_in//2, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=self.feat_in//2, out_channels=self.feat_out//4, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=self.feat_out//4, out_channels=self.feat_out, kernel_size=1)

        with torch.no_grad():
            # Initialize with identity mapping
            nn.init.constant_(self.conv3.bias, 5.0)
            self.conv3.weight *= 1e-3

    def infer_mask(self, readout):
        f = readout
        f = self.conv1(f)
        f = self.relu(f)
        f = self.conv2(f)
        f = self.relu(f)
        f = self.conv3(f)
        return f


class ReadoutBottleneckB(ReadoutBottleneck):
    """ By error some models were trained with this architecture """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(in_channels=self.feat_in, out_channels=self.feat_in//2, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=self.feat_in//2, out_channels=self.feat_out*2, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=self.feat_out*2, out_channels=self.feat_out, kernel_size=1)

        with torch.no_grad():
            # Initialize with identity mapping
            nn.init.constant_(self.conv3.bias, 5.0)
            self.conv3.weight *= 1e-3

    def infer_mask(self, readout):
        f = readout
        f = self.conv1(f)
        f = self.relu(f)
        f = self.conv2(f)
        f = self.relu(f)
        f = self.conv3(f)
        return f
