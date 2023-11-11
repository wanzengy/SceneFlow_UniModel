"""
Adapted from UZH-RPG https://github.com/uzh-rpg/rpg_e2vid
"""
import torch
import copy
import math

import numpy as np
import torch.nn as nn
import torch.nn.functional as f

from abc import abstractmethod

def skip_concat(x1, x2):
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]
    padding = nn.ZeroPad2d((diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
    x1 = padding(x1)
    return torch.cat([x1, x2], dim=1)


def skip_sum(x1, x2):
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]
    padding = nn.ZeroPad2d((diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
    x1 = padding(x1)
    return x1 + x2

def nonzero_normalize(x):
    mean, stddev = (
        x[x != 0].mean(),
        x[x != 0].std(),
    )
    x[x != 0] = (x[x != 0] - mean) / stddev
    return x

def optimal_crop_size(max_size, max_subsample_factor, safety_margin=0):
    """
    Find the optimal crop size for a given max_size and subsample_factor.
    The optimal crop size is the smallest integer which is greater or equal than max_size,
    while being divisible by 2^max_subsample_factor.
    """
    crop_size = int(pow(2, max_subsample_factor) * math.ceil(max_size / pow(2, max_subsample_factor)))
    crop_size += safety_margin * pow(2, max_subsample_factor)
    return crop_size

class CropParameters:
    """
    Helper class to compute and store useful parameters for pre-processing and post-processing
    of images in and out of E2VID.
    Pre-processing: finding the best image size for the network, and padding the input image with zeros
    Post-processing: Crop the output image back to the original image size
    """

    def __init__(self, width, height, num_encoders, safety_margin=0):

        self.height = height
        self.width = width
        self.num_encoders = num_encoders
        self.width_crop_size = optimal_crop_size(self.width, num_encoders, safety_margin)
        self.height_crop_size = optimal_crop_size(self.height, num_encoders, safety_margin)

        self.padding_top = math.ceil(0.5 * (self.height_crop_size - self.height))
        self.padding_bottom = math.floor(0.5 * (self.height_crop_size - self.height))
        self.padding_left = math.ceil(0.5 * (self.width_crop_size - self.width))
        self.padding_right = math.floor(0.5 * (self.width_crop_size - self.width))
        self.pad = nn.ZeroPad2d(
            (
                self.padding_left,
                self.padding_right,
                self.padding_top,
                self.padding_bottom,
            )
        )

        self.cx = math.floor(self.width_crop_size / 2)
        self.cy = math.floor(self.height_crop_size / 2)

        self.ix0 = self.cx - math.floor(self.width / 2)
        self.ix1 = self.cx + math.ceil(self.width / 2)
        self.iy0 = self.cy - math.floor(self.height / 2)
        self.iy1 = self.cy + math.ceil(self.height / 2)

    def crop(self, img):
        return img[..., self.iy0 : self.iy1, self.ix0 : self.ix1]


def recursive_clone(tensor):
    """
    Assumes tensor is a torch.tensor with 'clone()' method, possibly
    inside nested iterable.
    E.g., tensor = [(pytorch_tensor, pytorch_tensor), ...]
    """
    if hasattr(tensor, "clone"):
        return tensor.clone()
    try:
        return type(tensor)(recursive_clone(t) for t in tensor)
    except TypeError:
        print("{} is not iterable and has no clone() method.".format(tensor))

def copy_states(states):
    """
    Simple deepcopy if list of Nones, else clone.
    """
    if states[0] is None:
        return copy.deepcopy(states)
    return recursive_clone(states)

class BaseModel(nn.Module):
    """
    Base class for all models
    """

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)

class ConvLayer(nn.Module):
    """
    Convolutional layer.
    Default: bias, ReLU, no downsampling, no batch norm.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        activation="relu",
        norm=None,
        BN_momentum=0.1,
        w_scale=None,
        conv_func=nn.Conv2d
    ):
        super(ConvLayer, self).__init__()

        # bias = False if norm == "BN" else True
        bias = False
        padding = kernel_size // 2
        self.conv2d = conv_func(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if w_scale is not None:
            nn.init.uniform_(self.conv2d.weight, -w_scale, w_scale)
            nn.init.zeros_(self.conv2d.bias)

        if activation is not None:
            self.activation = getattr(torch, activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == "BN":
            self.norm_layer = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
        elif norm == "IN":
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.conv2d(x)

        if self.norm in ["BN", "IN"]:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out

class ConvLayer_(ConvLayer):
    """
    Clone of ConvLayer that acts like it has state, and allows residual.
    """

    def forward(self, x, prev_state, residual=0):
        # generate empty prev_state, if None is provided
        if prev_state is None:
            prev_state = torch.tensor(0)  # not used

        out = self.conv2d(x)

        if self.norm in ["BN", "IN"]:
            out = self.norm_layer(out)

        out += residual
        if self.activation is not None:
            out = self.activation(out)

        return out, prev_state

class TransposedConvLayer(nn.Module):
    """
    Transposed convolutional layer to increase spatial resolution (x2) in a decoder.
    Default: bias, ReLU, no downsampling, no batch norm.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=1,
        activation="relu",
        norm=None,
        scale_factor=2,
    ):
        super(TransposedConvLayer, self).__init__()

        bias = False if norm == "BN" else True
        self.transposed_conv2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=2,
            padding=padding,
            output_padding=1,
            bias=bias,
        )

        if activation is not None:
            self.activation = getattr(torch, activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == "BN":
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == "IN":
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)
        
        self.scale_factor=scale_factor

    def forward(self, x):
        out = self.transposed_conv2d(x)

        if self.norm in ["BN", "IN"]:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class UpsampleConvLayer(nn.Module):
    """
    Upsampling layer (bilinear interpolation + Conv2d) to increase spatial resolution (x2) in a decoder.
    Default: bias, ReLU, no downsampling, no batch norm.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        activation="relu",
        norm=None,
        scale_factor=2,
        conv_func=nn.Conv2d,
    ):
        super(UpsampleConvLayer, self).__init__()

        bias = False if norm == "BN" else True
        # bias = False
        padding = kernel_size // 2
        self.conv2d = conv_func(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        if activation is not None:
            self.activation = getattr(torch, activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == "BN":
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == "IN":
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)
        
        self.scale_factor = scale_factor

    def forward(self, x):
        x_upsampled = f.interpolate(x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        out = self.conv2d(x_upsampled)

        if self.norm in ["BN", "IN"]:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class RecurrentConvLayer(nn.Module):
    """
    Layer comprised of a convolution followed by a recurrent convolutional block.
    Default: bias, ReLU, no downsampling, no batch norm, ConvLSTM.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        recurrent_block_type="convlstm",
        recurrent_func = None,
        activation_ff="relu",
        activation_rec=None,
        norm=None,
        BN_momentum=0.1,
        conv_func=nn.Conv2d,
        **kwargs
    ):
        super(RecurrentConvLayer, self).__init__()

        assert recurrent_block_type in ["convlstm", "convgru"]
        self.recurrent_block_type = recurrent_block_type
        if self.recurrent_block_type == "convlstm":
            RecurrentBlock = ConvLSTM
        elif self.recurrent_block_type == "convgru":
            RecurrentBlock = ConvGRU
        else:
            RecurrentBlock = recurrent_func
        self.conv = ConvLayer(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            activation_ff,
            norm,
            BN_momentum=BN_momentum,
            conv_func=conv_func
        )
        self.recurrent_block = RecurrentBlock(
            input_size=out_channels, hidden_size=out_channels, kernel_size=3, activation=activation_rec, conv_func=nn.Conv2d
        )

    def forward(self, x, prev_state):
        x = self.conv(x)
        x, state = self.recurrent_block(x, prev_state)
        if isinstance(self.recurrent_block, ConvLSTM):
            state = (x, state)
        return x, state


class ResidualBlock(nn.Module):
    """
    Residual block as in "Deep residual learning for image recognition", He et al. 2016.
    Default: bias, ReLU, no downsampling, no batch norm, ConvLSTM.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        activation="relu",
        downsample=None,
        norm=None,
        BN_momentum=0.1,
        conv_func=nn.Conv2d,
        **kwargs
    ):
        super(ResidualBlock, self).__init__()
        bias = False if norm == "BN" else True
        # bias = False
        self.conv1 = conv_func(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
        )
        if activation is not None:
            if hasattr(torch, activation):
                self.activation = getattr(torch, activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == "BN":
            self.bn1 = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
            self.bn2 = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
        elif norm == "IN":
            self.bn1 = nn.InstanceNorm2d(out_channels)
            self.bn2 = nn.InstanceNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_func(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.norm in ["BN", "IN"]:
            out = self.bn1(out)
        if self.activation is not None:
            out = self.activation(out)
    
        out = self.conv2(out)
        if self.norm in ["BN", "IN"]:
            out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        if self.activation is not None:
            out = self.activation(out)
        return out


class ConvLSTM(nn.Module):
    """
    Convolutional LSTM module.
    Adapted from https://github.com/Atcold/pytorch-CortexNet/blob/master/model/ConvLSTMCell.py
    """

    def __init__(self, input_size, hidden_size, kernel_size, conv_func=nn.Conv2d):
        super(ConvLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        pad = kernel_size // 2

        # cache a tensor filled with zeros to avoid reallocating memory at each inference step if --no-recurrent is enabled
        self.zero_tensors = {}

        self.Gates = conv_func(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=pad)

    def forward(self, input_, prev_state=None):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:

            # create the zero tensor if it has not been created already
            state_size = tuple([batch_size, self.hidden_size] + list(spatial_size))
            if state_size not in self.zero_tensors:
                # allocate a tensor with size `spatial_size`, filled with zero (if it has not been allocated already)
                self.zero_tensors[state_size] = (
                    torch.zeros(state_size, dtype=input_.dtype).to(input_.device),
                    torch.zeros(state_size, dtype=input_.dtype).to(input_.device),
                )

            prev_state = self.zero_tensors[tuple(state_size)]

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell


class ConvGRU(nn.Module):
    """
    Convolutional GRU cell.
    Adapted from https://github.com/jacobkimmel/pytorch_convgru/blob/master/convgru.py
    """

    def __init__(self, input_size, hidden_size, kernel_size, activation=None, conv_func=nn.Conv2d):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = conv_func(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = conv_func(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = conv_func(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        # assert activation is None, "ConvGRU activation cannot be set (just for compatibility)"
        self.activation = nn.Tanh()

        # nn.init.orthogonal_(self.reset_gate.weight)
        # nn.init.orthogonal_(self.update_gate.weight)
        # nn.init.orthogonal_(self.out_gate.weight)
        # nn.init.constant_(self.reset_gate.bias, 0.0)
        # nn.init.constant_(self.update_gate.bias, 0.0)
        # nn.init.constant_(self.out_gate.bias, 0.0)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = torch.zeros(state_size, dtype=input_.dtype).to(input_.device)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        # out_inputs = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        out_inputs = self.activation(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))

        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state, new_state

# class Conv3d_Space(nn.Conv3d):
#     def __init__(self, in_channels, out_channels, kernel_size, padding):
#         kernel_size = (1, kernel_size, kernel_size)
#         padding = (1, padding, padding)
#         super().__init__(in_channels, out_channels, kernel_size = kernel_size, padding = padding)

class ConvGRU_3D(nn.Module):
    """
    Convolutional GRU cell.
    Adapted from https://github.com/jacobkimmel/pytorch_convgru/blob/master/convgru.py
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = kernel_size // 2 if type(kernel_size) == int else list(map(lambda x: x //2, kernel_size))
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv3d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv3d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv3d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

        nn.init.orthogonal_(self.reset_gate.weight)
        nn.init.orthogonal_(self.update_gate.weight)
        nn.init.orthogonal_(self.out_gate.weight)
        nn.init.constant_(self.reset_gate.bias, 0.0)
        nn.init.constant_(self.update_gate.bias, 0.0)
        nn.init.constant_(self.out_gate.bias, 0.0)

    def forward(self, input_, prev_state=None):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = torch.zeros(state_size, dtype=input_.dtype).to(input_.device)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state