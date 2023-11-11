import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import Models.Modules.base as B
import mmcv.ops as MOP

def diff_conv(x1, x2, conv_func):
    weight = conv_func.weight.sum((2, 3), keepdim=True)
    x2 = F.conv2d(input=x2, weight=weight, stride=conv_func.stride, padding=0, groups=conv_func.groups)
    return x1 - x2

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                stride=1, bias=False, activation="relu", norm=None):
        super(ConvLayer, self).__init__()

        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.act = getattr(torch, str(activation), nn.Identity())

        if norm == "BN":
            self.norm = nn.BatchNorm2d(out_channels, track_running_stats=True)
        elif norm == "IN":
            self.norm = nn.InstanceNorm2d(out_channels, track_running_stats=True)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class InvBottle(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, exp=4, activation=None, norm=None, diff=False):

        super(InvBottle, self).__init__()
        mid_channels = in_channels * exp
        # D = 1
        # self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, groups=groups, bias=False)

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=in_channels, bias=False)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, groups=groups, bias=bias)
        
        self.act = getattr(torch, str(activation), nn.Identity())
 
        if norm == "BN":
            self.norm = nn.BatchNorm2d(out_channels, track_running_stats=True)
        elif norm == "IN":
            self.norm = nn.InstanceNorm2d(out_channels, track_running_stats=True)
        else:
            self.norm = nn.Identity()
        
        self.diff = diff

    def forward(self, x):
        # self.conv1.weight.data = self.conv1.weight * self.mask
        if self.diff:
            _x = self.conv1(x)
            x = diff_conv(_x, x, self.conv1)
        else:
            x = self.conv1(x)

        x = self.conv2(x)
        # x = self.norm(x)
        x = self.act(x)
        return x

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2, dilation=dilation, groups=groups, bias=bias)
        self.conv2 = nn.Conv2d(out_channels * 5, out_channels, 1, groups=out_channels, bias=False)
        # self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2, dilation=dilation, groups=groups, bias=bias)
        self.act = nn.ReLU()
        # self.avgpool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=1)

    def forward(self, x):
        if x.size(-1) % 2 == 1:
            x = F.pad(x, pad=(1, 0, 0, 0))
        if x.size(-2) % 2 == 1:
            x = F.pad(x, pad=(0, 0, 1, 0))

        x_pad = F.pad(x, pad=(1, 0, 1, 0))
        xc = self.conv1(x)
        xt = self.conv1(x_pad[..., :-1, 1:])
        xb = self.conv1(x[..., 1:, :])
        xl = self.conv1(x_pad[..., 1:, :-1])
        xr = self.conv1(x[..., :, 1:])

        x = torch.cat([xc, xt, xb, xl, xr], dim=1)
        x = self.conv2(x)
        return self.act(x)

class Traj_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):

        super(Traj_Conv2d, self).__init__()
        assert kernel_size > 1, "Not support 1 * 1 Conv"
        D = 2 * kernel_size - 2
        # self.conv1 = nn.Conv2d(in_channels, in_channels * D, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=in_channels, bias=False)
        self.conv_offset = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, bias=True)
        self.conv1 = MOP.DeformConv2d(in_channels, in_channels * D, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=in_channels)
        self.conv2 = nn.Conv2d(in_channels * D, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=groups, bias=bias)


    def init_weights(self):
        super().init_weights()
        if hasattr(self, 'conv_offset'):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()

    def forward(self, x, x2=None):
        # self.conv1.weight.data = self.conv1.weight * self.mask
        # offset_range = x1.size(-1) // 4
        if x2 == None:
            x2 = x
        
        t = self.conv_offset(x)
        t = diff_conv(t, x2, self.conv_offset)

        x = self.conv1(x, t)
        x = diff_conv(x, x2, self.conv1)
        x = self.conv2(x)
        # ts_mut = self.conv3(ux * x).view(B, -1, H, W)
        return x

class ResidualBlock(nn.Module):
    """
    Residual block as in "Deep residual learning for image recognition", He et al. 2016.
    Default: bias, ReLU, no downsampling, no batch norm, ConvLSTM.
    """

    def __init__(self, in_channels, out_channels,
                stride=1, activation="relu", norm=None, **kwargs):
        super(ResidualBlock, self).__init__()
        bias = False if norm == "BN" else True
        self.conv1 = InvBottle(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            activation=activation,
            norm=norm
        )
        # self.conv2 = InvBottle(
        #     out_channels,
        #     out_channels,
        #     kernel_size=3,
        #     stride=1,
        #     padding=1,
        #     bias=bias,
        #     activation=activation,
        #     norm=norm
        # )
        self.act = getattr(torch, str(activation), nn.Identity())

        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channels, 4 * in_channels, kernel_size=1, padding=0, bias=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(4 * in_channels, in_channels, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x.clone()
        x = self.conv1(x)
        # x = self.conv2(x)
        # x = x + residual
        x = self.act(x)
        x = self.squeeze(x) * x + residual
        return x

class ConvGRU(nn.Module):
    """
    Convolutional GRU cell.
    Adapted from https://github.com/jacobkimmel/pytorch_convgru/blob/master/convgru.py
    """

    def __init__(self, input_size, hidden_size, kernel_size, activation='tanh', conv_func=nn.Conv2d):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_size = hidden_size
        self.reset_gate = conv_func(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = conv_func(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = conv_func(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        # assert activation is None, "ConvGRU activation cannot be set (just for compatibility)"
        self.act = getattr(torch, str(activation), nn.Identity())

    def forward(self, x, state):
        # generate empty prev_state, if None is provided
        if state is None:
            size = (x.size(0), self.hidden_size,) +  x.size()[2:]
            state = torch.zeros(size).to(x)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([x, state], dim=1)
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        x = self.act(self.out_gate(torch.cat([x, state * reset], dim=1)))
        # x = torch.tanh(self.out_gate(torch.cat([x, state * reset], dim=1)))

        state = state * (1 - update) + x * update

        return state, state

class ConvLSTM(nn.Module):
    """
    Convolutional LSTM module.
    Adapted from https://github.com/Atcold/pytorch-CortexNet/blob/master/model/ConvLSTMCell.py
    """

    def __init__(self, input_size, hidden_size, kernel_size, activation, conv_func=nn.Conv2d):
        super(ConvLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        pad = kernel_size // 2

        # cache a tensor filled with zeros to avoid reallocating memory at each inference step if --no-recurrent is enabled
        self.zero_tensors = {}

        self.Gates = conv_func(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=pad)
        self.act = getattr(torch, str(activation), nn.Identity())

    def forward(self, x, prev_state=None):

        # generate empty prev_state, if None is provided
        if prev_state is None:
            # create the zero tensor if it has not been created already
            size = tuple([x.size(0), self.hidden_size] + list(x.size()[2:]))
            if size not in self.zero_tensors:
                # allocate a tensor with size `spatial_size`, filled with zero (if it has not been allocated already)
                self.zero_tensors[size] = (
                    torch.zeros(size).to(x),
                    torch.zeros(size).to(x),
                )

            prev_state = self.zero_tensors[size]

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((x, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = self.act(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * self.act(cell)

        return hidden, (hidden, cell)


class Decouple(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, bias=False):

        super(Decouple, self).__init__()
        assert kernel_size > 1, "Not support 1 * 1 Conv"

        D = kernel_size * 2 - 2
        # self.conv1 = nn.Conv2d(in_channels, in_channels * D, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=in_channels, bias=False)
        self.kernel_size = kernel_size
        # self.conv_offset = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size,  kernel_size=3, padding=1, stride=stride, dilation=dilation, bias=True)

        self.conv_offset = InvBottle(in_channels, 4 * 2 * kernel_size * kernel_size,  kernel_size=3, padding=1, stride=stride, dilation=dilation, bias=True, diff=True)

        self.conv_m = nn.ModuleDict(
            {
                'conv1': MOP.DeformConv2d(in_channels, in_channels, kernel_size, padding=padding, stride=stride, dilation=dilation, groups=in_channels, deform_groups=4),
                'conv2': nn.Conv2d(in_channels, out_channels, 1, bias=False), 
            }
        )
        
    def init_weights(self):
        super().init_weights()
        if hasattr(self, 'conv_offset'):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()


    def forward(self, x1, x2=None):
        # self.conv1.weight.data = self.conv1.weight * self.mask
        # offset_range = x1.size(-1) // 4

        if x2 == None:
            x2 = x1

        t = self.conv_offset(x1)
        # t = diff_conv(t, x2, self.conv_offset)
        
        def traj_encode(x, t, f):
            x = f['conv1'](x, t)
            x = diff_conv(x, x2, f['conv1'])
            x = f['conv2'](x)
            return x

        m = traj_encode(x1, t, self.conv_m)
        return x1, m

class RecurrentDecoupleLayer(nn.Module):
    """
    Layer comprised of a convolution followed by a recurrent convolutional block.
    Default: bias, ReLU, no downsampling, no batch norm, ConvLSTM.
    """

    def __init__(self, in_channels, out_channels,
                kernel_size=3, stride=1, activation_ff="relu", activation_rec=None, norm=None, **kwargs):
        super(RecurrentDecoupleLayer, self).__init__()

        self.decouple = nn.ModuleDict(
            {   
                'down': DownSample(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2),
                # 'conv': InvBottle(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2, exp=4, activation=activation_ff, norm=norm),
                'dec': Decouple(out_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1),
            }
        )

        self.traj_encoder = nn.ModuleDict(
            {   
                # 'conv': InvBottle_neck(out_channels, out_channels, 3, padding=1, exp=4, activation=activation_ff, norm=norm),
                'rec': ConvGRU(out_channels, out_channels, kernel_size=3, activation=activation_rec, conv_func=InvBottle),
                # 'rec': ConvLSTM(out_channels, out_channels, kernel_size=3, activation=activation_rec, conv_func=InvBottle),
                'res': ResidualBlock(out_channels, out_channels, 1, activation=activation_ff, norm=norm),
            }
        )

    def forward(self, x, state):
        x = self.decouple['down'](x)
        # x = self.decouple['conv'](x)
        x, t = self.decouple['dec'](x)

        def encode(x, s, encoder):
            # x = coder['conv'](x)
            x, s = encoder['rec'](x, s)
            x = encoder['res'](x)
            return x, s

        t, state = encode(t, state, self.traj_encoder)
        return x, t, state

class MultiResUNetRecurrent(nn.Module):
    """
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block.
    Symmetric, skip connections on every encoding layer.
    Predictions at each decoding layer.
    Predictions are added as skip connection (concat) to the input of the subsequent layer.
    """
    def __init__(self, 
                base_num_channels,
                num_encoders,
                num_residual_blocks,
                num_output_channels,
                norm,
                num_bins,
                kernel_size=5,
                channel_multiplier=2,
                activations_ff="relu",
                activations_rec=None,
                activation_out="tanh",
                skip_func=None):

        super(MultiResUNetRecurrent, self).__init__()
        self.__dict__.update({
            'num_encoders': num_encoders,
            'num_residual_blocks': num_residual_blocks,
            'num_output_channels': num_output_channels,
            'norm': norm,
            'num_bins': num_bins,
            'kernel_size': kernel_size,
            'ff_act': activations_ff,
            'rec_act': activations_rec,
            'final_activation': activation_out,
            'skip_func': skip_func,
        })
        assert num_output_channels > 0

        self.encoder_input_sizes = [
            int(base_num_channels * pow(channel_multiplier, i)) for i in range(num_encoders)
        ]
        self.encoder_output_sizes = [
            int(base_num_channels * pow(channel_multiplier, i + 1)) for i in range(num_encoders)
        ]

        self.max_num_channels = self.encoder_output_sizes[-1]

        self.num_states = num_encoders
        self.states = [None] * self.num_states
        
        self.encoders = self.build_encoders()
        self.decoders = self.build_decoders()
        self.preds = self.build_multires_prediction()

    def build_encoders(self):
        encoders = nn.ModuleList()
        for i, (input_size, output_size) in enumerate(zip(self.encoder_input_sizes, self.encoder_output_sizes)):
            if i == 0:
                input_size = self.num_bins
            stride = 2
            encoders.append(
                RecurrentDecoupleLayer(
                    input_size,
                    output_size,
                    kernel_size=self.kernel_size,
                    stride=stride,
                    activation_ff=self.ff_act,
                    activation_rec=self.rec_act,
                    norm=self.norm,
                )
            )
        return encoders

    def build_multires_prediction(self):
        preds = nn.ModuleList()
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        for i, input_size in enumerate(decoder_output_sizes):
            preds.append(
                nn.Sequential(
                    ConvLayer(
                        input_size,
                        self.num_output_channels,
                        1,
                        bias=False,
                        activation=self.final_activation
                    ),
                )
                
            )
        return preds

    def build_decoders(self):
        decoder_input_sizes = reversed(self.encoder_output_sizes)
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        decoders = nn.ModuleList()
        for i, (input_size, output_size) in enumerate(zip(decoder_input_sizes, decoder_output_sizes)):
            input_size = input_size if i == 0 else input_size * 2 + 2
            decoders.append(
                nn.Sequential(
                    # nn.ConvTranspose2d(
                    #     input_size,
                    #     input_size,
                    #     self.kernel_size,
                    #     stride=2,
                    #     padding=self.kernel_size // 2,
                    #     output_padding=1,
                    #     bias=False),
                    InvBottle(
                        input_size,
                        output_size,
                        self.kernel_size,
                        padding=self.kernel_size//2,
                        bias=False,
                        activation=self.ff_act,
                        norm=self.norm),

                )
            )
        return decoders

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: [N x num_output_channels x H x W for i in range(self.num_encoders)]
        """

        # encoder
        traj = []
        for i, encoder in enumerate(self.encoders):
            x, t, state = encoder(x, self.states[i])
            traj.append(t)
            self.states[i] = state

        predictions = []
        for i, (decoder, pred) in enumerate(zip(self.decoders, self.preds)):
            if i > 0:
                t = self.skip_func(p, self.skip_func(traj[-i - 1], t))
            t = F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)
            t = decoder(t)
            p = pred(t)
            predictions.append(p)

        return predictions

class Model(MultiResUNetRecurrent):
    """
    Recurrent version of the EV-FlowNet architecture from the paper "EV-FlowNet: Self-Supervised Optical
    Flow for Event-based Cameras", Zhu et al., RSS 2018.
    """
    def __init__(self,
                norm=None,
                base_num_channels=32, 
                num_encoders=4,
                num_residual_blocks=2,
                num_output_channels=2,
                num_bins=2,
                norm_input=True,
                kernel_size=3,
                channel_multiplier=2,
                activations_ff="relu",
                activations_rec=None,
                activation_out="tanh",
                mask_output=True,
                skip_func=B.skip_concat,
                **kwargs
                ):
        super().__init__(base_num_channels, num_encoders, num_residual_blocks, num_output_channels,
            norm, num_bins, kernel_size, channel_multiplier, activations_ff, activations_rec, activation_out, skip_func)

        self.crop = None
        self.mask = mask_output
        self.norm_input = norm_input
        self.num_bins = num_bins
        # self.traj = Traj_Conv2d(1, 4, 3, bias=False)

    def detach_states(self):        
        def detach(state):
            if type(state) is tuple:
                tmp = []
                for hidden in state:
                    hidden = detach(hidden)
                    tmp.append(hidden)
                return tuple(tmp)
            else:
                return state.detach()
        
        self.states = [detach(state) for state in self.states]

    def reset_states(self):
        self.states = [None] * self.num_states

    def init_cropping(self, width, height, safety_margin=0):
        self.crop = B.CropParameters(width, height, self.num_encoders, safety_margin)

    def flow_resize(self, x, multires_flow):
         # upsample flow estimates to the original input resolution
        flow_list = []
        for flow in multires_flow:
            flow_list.append(
                torch.nn.functional.interpolate(
                    flow,
                    scale_factor=(
                        x.shape[2] / flow.shape[2],
                        x.shape[3] / flow.shape[3],
                    ),
                    mode='nearest-exact'
                )
            )

        # crop output
        if self.crop is not None:
            for i, flow in enumerate(flow_list):
                flow_list[i] = flow[:, :, self.crop.iy0 : self.crop.iy1, self.crop.ix0 : self.crop.ix1]
                flow_list[i] = flow_list[i].contiguous()
        
        return flow_list

    def reblur(self, x, flow):
        if flow == None:
            flow = F.pad(x, (1, 0, 1, 0), "constant", 0)
            flow = flow.repeat(1, 2, 1, 1)
            flow[:, 0, 1:, :] = flow[:, 0, 1:, :] - flow[:, 0, :-1, :]
            flow[:, 1, :, 1:] = flow[:, 1, :, 1:] - flow[:, 0, :, :-1]
            flow = flow[..., 1:, 1:]

        B, C, H, W = x.size()
        x = x.view(-1, 1, H, W)
        x = F.unfold(x, kernel_size=3, padding=1)
        x = x.view(-1, 9, H, W)
        blur_kernel = self.blur_conv(flow)
        blur_kernel = F.relu(blur_kernel)
        x = torch.sum(x * blur_kernel, dim=1, keepdim=True).view(B, C, H, W)
        return x

    def forward(self, x):
        """
        :param event_voxel: N x num_bins x H x W
        :param event_cnt: N x 4 x H x W per-polarity event cnt and average timestamp
        :return: output dict with list of [N x 2 X H X W] (x, y) displacement within event_tensor.
        """

        # normalize input
        if self.norm_input:
           x = B.nonzero_normalize(x)

        # pad input
        if self.crop is not None:
            x = self.crop.pad(x)

        # x = self.reblur(x, self.flow)
        # cnt, ts = x[:, :-1], x[:, -1:]
        # cnt = (cnt > 0).float()
        # ts = self.traj(ts)
        # forward pass
        multires_flow = super().forward(x)
        multires_flow = self.flow_resize(x, multires_flow)

        return {"flow": multires_flow}