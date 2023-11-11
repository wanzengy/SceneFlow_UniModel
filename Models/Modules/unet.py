"""
Adapted from UZH-RPG https://github.com/uzh-rpg/rpg_e2vid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import (
    ConvLayer,
    RecurrentConvLayer,
    ConvGRU,
    ResidualBlock,
    TransposedConvLayer,
    UpsampleConvLayer,
)

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

class BaseUNet(nn.Module):
    """
    Base class for conventional UNet architecture.
    Symmetric, skip connections on every encoding layer.
    """

    ff_type = ConvLayer
    res_type = ResidualBlock
    upsample_type = UpsampleConvLayer
    transpose_type = TransposedConvLayer
    w_scale_pred = None

    def __init__(
        self,
        base_num_channels,
        num_encoders,
        num_residual_blocks,
        num_output_channels,
        skip_type,
        norm,
        use_upsample_conv,
        num_bins,
        recurrent_block_type=None,
        kernel_size=5,
        channel_multiplier=2,
        activations=["relu", None],
        spiking_feedforward_block_type=None,
        spiking_neuron=None,
    ):
        super(BaseUNet, self).__init__()
        self.base_num_channels = base_num_channels
        self.num_encoders = num_encoders
        self.num_residual_blocks = num_residual_blocks
        self.num_output_channels = num_output_channels
        self.kernel_size = kernel_size
        self.skip_type = skip_type
        self.norm = norm
        self.num_bins = num_bins
        self.recurrent_block_type = recurrent_block_type
        self.channel_multiplier = channel_multiplier
        self.ff_act, self.rec_act = activations

        self.spiking_kwargs = {}
        if spiking_feedforward_block_type is not None:
            self.spiking_kwargs["spiking_feedforward_block_type"] = spiking_feedforward_block_type
        if type(spiking_neuron) is dict:
            self.spiking_kwargs.update(spiking_neuron)

        self.skip_ftn = eval("skip_" + skip_type)
        if use_upsample_conv:
            self.UpsampleLayer = self.upsample_type
        else:
            self.UpsampleLayer = self.transpose_type
        assert self.num_output_channels > 0

        self.encoder_input_sizes = [
            int(self.base_num_channels * pow(self.channel_multiplier, i)) for i in range(self.num_encoders)
        ]
        self.encoder_output_sizes = [
            int(self.base_num_channels * pow(self.channel_multiplier, i + 1)) for i in range(self.num_encoders)
        ]
        self.max_num_channels = self.encoder_output_sizes[-1]

    def build_encoders(self):
        encoders = nn.ModuleList()
        for (input_size, output_size) in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            encoders.append(
                self.ff_type(
                    input_size,
                    output_size,
                    kernel_size=self.kernel_size,
                    stride=2,
                    activation=self.ff_act,
                    norm=self.norm,
                    w_scale=5,
                    **self.spiking_kwargs
                )
            )
        return encoders

    def build_resblocks(self):
        resblocks = nn.ModuleList()
        for i in range(self.num_residual_blocks):
            resblocks.append(
                self.res_type(
                    self.max_num_channels,
                    self.max_num_channels,
                    activation=self.ff_act,
                    norm=self.norm,
                    **self.spiking_kwargs
                )
            )
        return resblocks

    def build_decoders(self):
        decoder_input_sizes = reversed(self.encoder_output_sizes)
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        decoders = nn.ModuleList()
        for input_size, output_size in zip(decoder_input_sizes, decoder_output_sizes):
            decoders.append(
                self.UpsampleLayer(
                    input_size if self.skip_type == "sum" else 2 * input_size,
                    output_size,
                    kernel_size=self.kernel_size,
                    activation=self.ff_act,
                    norm=self.norm,
                    **self.spiking_kwargs
                )
            )
        return decoders

    def build_prediction_layer(self, num_output_channels, norm=None):
        return self.ff_type(
            self.base_num_channels if self.skip_type == "sum" else 2 * self.base_num_channels,
            num_output_channels,
            1,
            activation=None,
            norm=norm,
            w_scale=5,
        )
    
    def detach_states(self):
        pass

    def reset_states(self):
        pass


class UNetRecurrent(BaseUNet):
    """
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block.
    Symmetric, skip connections on every encoding layer.
    """

    rec_type = RecurrentConvLayer

    def __init__(self, unet_kwargs):
        final_activation = unet_kwargs.pop("final_activation", "none")
        self.final_activation = getattr(torch, final_activation, None)
        super().__init__(**unet_kwargs)

        self.head = ConvLayer(
            self.num_bins,
            self.base_num_channels,
            kernel_size=self.kernel_size,
            stride=1,
        )

        self.encoders = self.build_recurrent_encoders()
        self.resblocks = self.build_resblocks()
        self.decoders = self.build_decoders()
        self.pred = self.build_prediction_layer(self.num_output_channels, self.norm)
        self.num_states = self.num_encoders
        self.states = [None] * self.num_states

    def build_recurrent_encoders(self):
        encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            encoders.append(
                self.rec_type(
                    input_size,
                    output_size,
                    kernel_size=self.kernel_size,
                    stride=2,
                    recurrent_block_type=self.recurrent_block_type,
                    activation_ff=self.ff_act,
                    activation_rec=self.rec_act,
                    norm=self.norm,
                )
            )
        return encoders

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: N x num_output_channels x H x W
        """

        # head
        x = self.head(x)
        head = x

        # encoder
        blocks = []
        for i, encoder in enumerate(self.encoders):
            x, state = encoder(x, self.states[i])
            blocks.append(x)
            self.states[i] = state

        # residual blocks
        for resblock in self.resblocks:
            x, _ = resblock(x)

        # decoder
        for i, decoder in enumerate(self.decoders):
            x = decoder(self.skip_ftn(x, blocks[self.num_encoders - i - 1]))

        # tail
        img = self.pred(self.skip_ftn(x, head))
        if self.final_activation is not None:
            img = self.final_activation(img)
        return img


class MultiResUNet(BaseUNet):
    """
    Conventional UNet architecture.
    Symmetric, skip connections on every encoding layer.
    Predictions at each decoding layer.
    Predictions are added as skip connection (concat) to the input of the subsequent layer.
    """

    def __init__(self, unet_kwargs):
        self.final_activation = unet_kwargs.pop("final_activation", None)
        super().__init__(**unet_kwargs)

        self.encoders = self.build_encoders()
        self.resblocks = self.build_resblocks()
        self.decoders = self.build_multires_prediction_decoders()
        self.preds = self.build_multires_prediction_layer()

    def build_encoders(self):
        encoders = nn.ModuleList()
        for i, (input_size, output_size) in enumerate(zip(self.encoder_input_sizes, self.encoder_output_sizes)):
            if i == 0:
                input_size = self.num_bins
            encoders.append(
                self.ff_type(
                    input_size,
                    output_size,
                    kernel_size=self.kernel_size,
                    stride=2,
                    activation=self.ff_act,
                    norm=self.norm,
                    **self.spiking_kwargs
                )
            )
        return encoders

    def build_multires_prediction_layer(self):
        preds = nn.ModuleList()
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        for output_size in decoder_output_sizes:
            preds.append(
                self.ff_type(output_size, self.num_output_channels, 1, activation=self.final_activation, norm=self.norm)
            )
        return preds

    def build_multires_prediction_decoders(self):
        decoder_input_sizes = reversed(self.encoder_output_sizes)
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        decoders = nn.ModuleList()
        for i, (input_size, output_size) in enumerate(zip(decoder_input_sizes, decoder_output_sizes)):
            prediction_channels = 0 if i == 0 else self.num_output_channels
            decoders.append(
                self.UpsampleLayer(
                    2 * input_size + prediction_channels,
                    output_size,
                    kernel_size=self.kernel_size,
                    activation=self.ff_act,
                    norm=self.norm,
                    **self.spiking_kwargs
                )
            )
        return decoders

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: [N x num_output_channels x H x W for i in range(self.num_encoders)]
        """

        # encoder
        blocks = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            blocks.append(x)

        # residual blocks
        for resblock in self.resblocks:
            x, _ = resblock(x)

        # decoder and multires predictions
        predictions = []
        for i, (decoder, pred) in enumerate(zip(self.decoders, self.preds)):
            x = self.skip_ftn(x, blocks[self.num_encoders - i - 1])
            if i > 0:
                x = self.skip_ftn(predictions[-1], x)
            x = decoder(x)
            predictions.append(pred(x))

        return predictions


class MultiResUNetRecurrent(BaseUNet):
    """
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block.
    Symmetric, skip connections on every encoding layer.
    Predictions at each decoding layer.
    Predictions are added as skip connection (concat) to the input of the subsequent layer.
    """

    rec_type = RecurrentConvLayer

    def __init__(self, 
                base_num_channels,
                num_encoders,
                num_residual_blocks,
                num_output_channels,
                skip_type,
                norm,
                use_upsample_conv,
                num_bins,
                recurrent_block_type=None,
                kernel_size=5,
                channel_multiplier=2,
                activations=["relu", None],
                spiking_feedforward_block_type=None,
                spiking_neuron=None,
                final_activation=None,
                conv_func=nn.Conv2d):
        self.final_activation = final_activation
        
        super().__init__(base_num_channels,
                        num_encoders,
                        num_residual_blocks,
                        num_output_channels,
                        skip_type,
                        norm,
                        use_upsample_conv,
                        num_bins,
                        recurrent_block_type,
                        kernel_size,
                        channel_multiplier,
                        activations,
                        spiking_feedforward_block_type,
                        spiking_neuron,)

        self.conv_func = conv_func
        self.encoders = self.build_recurrent_encoders()
        self.resblocks = self.build_resblocks()
        self.decoders = self.build_multires_prediction_decoders()
        self.preds = self.build_multires_prediction_layer()
        self.num_states = self.num_encoders
        self.states = [None] * self.num_states

    def build_recurrent_encoders(self):
        encoders = nn.ModuleList()
        for i, (input_size, output_size) in enumerate(zip(self.encoder_input_sizes, self.encoder_output_sizes)):
            if i == 0:
                input_size = self.num_bins
            encoders.append(
                self.rec_type(
                    input_size,
                    output_size,
                    kernel_size=self.kernel_size,
                    stride=2,
                    recurrent_block_type=self.recurrent_block_type,
                    activation_ff=self.ff_act,
                    activation_rec=self.rec_act,
                    norm=self.norm,
                    conv_func=self.conv_func,
                    **self.spiking_kwargs
                )
            )
        return encoders

    def build_multires_prediction_layer(self):
        preds = nn.ModuleList()
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        for output_size in decoder_output_sizes:
            preds.append(
                self.ff_type(
                    output_size,
                    self.num_output_channels,
                    1,
                    activation=self.final_activation,
                    norm=self.norm,
                    w_scale=self.w_scale_pred,
                    conv_func=nn.Conv2d,
                )
            )
        return preds

    def build_multires_prediction_decoders(self):
        decoder_input_sizes = reversed(self.encoder_output_sizes)
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        decoders = nn.ModuleList()
        for i, (input_size, output_size) in enumerate(zip(decoder_input_sizes, decoder_output_sizes)):
            prediction_channels = 0 if i == 0 else self.num_output_channels
            decoders.append(
                self.UpsampleLayer(
                    2 * input_size + prediction_channels,
                    output_size,
                    kernel_size=self.kernel_size,
                    activation=self.ff_act,
                    norm=self.norm,
                    **self.spiking_kwargs
                )
            )
        return decoders

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: [N x num_output_channels x H x W for i in range(self.num_encoders)]
        """

        # encoder
        blocks = []
        for i, encoder in enumerate(self.encoders):
            x, state = encoder(x, self.states[i])
            blocks.append(x)
            self.states[i] = state

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # decoder and multires predictions
        predictions = []
        for i, (decoder, pred) in enumerate(zip(self.decoders, self.preds)):
            x = self.skip_ftn(x, blocks[self.num_encoders - i - 1])
            if i > 0:
                x = self.skip_ftn(predictions[-1], x)
            x = decoder(x)
            predictions.append(pred(x))

        return predictions

class MultiResTwoUNetRecurrent(BaseUNet):
    """
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block.
    Symmetric, skip connections on every encoding layer.
    Predictions at each decoding layer.
    Predictions are added as skip connection (concat) to the input of the subsequent layer.
    """

    rec_type = RecurrentConvLayer

    def __init__(self, 
                base_num_channels,
                num_encoders,
                num_residual_blocks,
                num_output_channels,
                skip_type,
                norm,
                use_upsample_conv,
                num_bins,
                kernel_size=5,
                channel_multiplier=2,
                activations=["relu", None],
                spiking_feedforward_block_type=None,
                spiking_neuron=None,
                final_activation=None,
                traj_recurrent_func=ConvGRU,
                recurrent_func=ConvGRU,
                conv_func=nn.Conv2d,
                traj_func=nn.Conv2d,
                dec_func=None):
        self.final_activation = final_activation
        
        super().__init__(base_num_channels,
                        num_encoders,
                        num_residual_blocks,
                        num_output_channels,
                        skip_type,
                        norm,
                        use_upsample_conv,
                        num_bins,
                        None,
                        kernel_size,
                        channel_multiplier,
                        activations,
                        spiking_feedforward_block_type,
                        spiking_neuron,)

        self.conv_func = conv_func
        self.traj_func = traj_func
        self.traj_recurrent_func = traj_recurrent_func
        self.recurrent_func = recurrent_func
        self.traj_recurrent_type = self.traj_recurrent_type

        self.encoders = self.build_recurrent_encoders()
        self.traj_encoders = self.build_traj_encoders()

        self.resblocks = self.build_resblocks()
        self.decoders = self.build_multires_prediction_decoders()
        self.preds = self.build_multires_prediction_layer()
        self.num_states = self.num_encoders
        self.states = [None] * self.num_states
        self.traj_states = [None] * self.num_states

    def build_traj_encoders(self):
        encoders = nn.ModuleList()
        for i, (input_size, output_size) in enumerate(zip(self.encoder_input_sizes, self.encoder_output_sizes)):
            stride = 1
            kernel_size = self.kernel_size
            encoders.append(
                self.rec_type(
                    output_size,
                    output_size,
                    kernel_size=kernel_size,
                    stride=stride,
                    recurrent_block_type=self.traj_recurrent_block_type,
                    recurrent_func=self.traj_recurrent_func,
                    activation_ff=self.ff_act,
                    activation_rec=self.rec_act,
                    norm=self.norm,
                    conv_func=self.traj_func,
                    **self.spiking_kwargs
                )
            )
        return encoders

    def build_recurrent_encoders(self):
        encoders = nn.ModuleList()
        for i, (input_size, output_size) in enumerate(zip(self.encoder_input_sizes, self.encoder_output_sizes)):
            if i == 0:
                input_size = self.num_bins
                stride = 2
            else:
                stride = 2
            # stride = 1
            encoders.append(
                self.rec_type(
                    input_size,
                    output_size,
                    kernel_size=self.kernel_size,
                    stride=stride,
                    recurrent_block_type=self.recurrent_block_type,
                    activation_ff=self.ff_act,
                    activation_rec=self.rec_act,
                    norm=self.norm,
                    conv_func=self.conv_func,
                    **self.spiking_kwargs
                )
            )
        return encoders

    def build_multires_prediction_layer(self):
        preds = nn.ModuleList()
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        for output_size in decoder_output_sizes:
            preds.append(
                self.ff_type(
                    output_size,
                    self.num_output_channels,
                    1,
                    activation=self.final_activation,
                    norm=self.norm,
                    w_scale=self.w_scale_pred,
                    conv_func=nn.Conv2d,
                )
            )
        return preds

    def build_multires_prediction_decoders(self):
        decoder_input_sizes = reversed(self.encoder_output_sizes)
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        decoders = nn.ModuleList()
        for i, (input_size, output_size) in enumerate(zip(decoder_input_sizes, decoder_output_sizes)):
            prediction_channels = 0 if i == 0 else self.num_output_channels
            scale_factor = 2
            decoders.append(
                self.UpsampleLayer(
                    2 * input_size + prediction_channels,
                    output_size,
                    kernel_size=self.kernel_size,
                    activation=self.ff_act,
                    norm=self.norm,
                    scale_factor=scale_factor,
                    **self.spiking_kwargs
                )
            )
        return decoders

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: [N x num_output_channels x H x W for i in range(self.num_encoders)]
        """

        # encoder
        blocks = []
        for i, (encoder, traj_encoder) in enumerate(zip(self.encoders, self.traj_encoders)):
            x, state = encoder(x, self.states[i])
            self.states[i] = state
            t, state = traj_encoder(x, self.traj_states[i])
            self.traj_states[i] = state
            blocks.append(t)
            # x = F.avg_pool2d(x, 2)

        x = t
        # x = self.mixture(t)
        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # decoder and multires predictions
        predictions = []
        for i, (decoder, pred) in enumerate(zip(self.decoders, self.preds)):
            x = self.skip_ftn(x, blocks[self.num_encoders - i - 1])
            if i > 0:
                x = self.skip_ftn(predictions[-1], x)
            x = decoder(x)
            predictions.append(pred(x))
            # if i < len(self.decoders) - 1:
            #     x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        return predictions