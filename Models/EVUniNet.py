import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Modules.base import CropParameters, nonzero_normalize
from .Modules.unet import BaseUNet

class MultiResUNet_Uni(BaseUNet):
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
        # self.ego_pred = self.build_ego_prediction_layer()

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
                self.ff_type(output_size,
                             self.num_output_channels,
                             1,
                             activation=self.final_activation,
                             norm="BN")
            )
        return preds

    def build_ego_prediction_layer(self):
        input_size = self.encoder_output_sizes[-1]
        ego_pred = nn.Sequential(
            nn.AdaptiveAvgPool2d((16, 16)),
            self.ff_type(input_size, 
                        input_size // 4, 
                        kernel_size=3, 
                        stride=4, 
                        activation=self.ff_act, 
                        norm=self.norm),
            self.ff_type(input_size // 4, 
                        6, 
                        kernel_size=3, 
                        stride=4, 
                        activation=self.ff_act, 
                        norm=self.norm),
        )
        return ego_pred

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
            x = resblock(x)

        # ego prediction
        ego_motion = torch.zeros(x.shape[0], 6).to(x.device)
        # ego_motion = self.ego_pred(x)
        # ego_motion = ego_motion.squeeze(dim=3).squeeze(dim=2)

        # depth prediction
        # decoder and multires predictions
        disps = []
        tmp = []
        for i, (decoder, pred) in enumerate(zip(self.decoders, self.preds)):
            x = self.skip_ftn(x, blocks[self.num_encoders - i - 1])
            if i > 0:
                x = self.skip_ftn(disps[-1], x)
            x = decoder(x)
            p = pred(x)
            disps.append(p)

        return ego_motion, disps, tmp

class Model(MultiResUNet_Uni):
    """
    Recurrent version of the EV-FlowNet architecture from the paper "EV-FlowNet: Self-Supervised Optical
    Flow for Event-based Cameras", Zhu et al., RSS 2018.
    """
    def __init__(self,
                base_num_channels=32, 
                kernel_size=3,
                num_bins=2,
                num_encoders=4,
                norm_input=True,
                norm=None,
                use_upsample_conv=True, 
                num_residual_blocks=2,
                num_output_channels=2, 
                skip_type = 'concat',
                channel_multiplier=2,
                activations=["relu", None],
                final_activation="tanh",
                mask_output=True,
                **kwargs
                ):
        
        unet_kwargs = {
            "base_num_channels": base_num_channels,
            "num_encoders": num_encoders,
            "num_residual_blocks": num_residual_blocks,
            "num_output_channels": num_output_channels,
            "skip_type": skip_type,
            "norm": norm,
            "use_upsample_conv": use_upsample_conv,
            "num_bins": num_bins,
            "recurrent_block_type": None,
            "kernel_size": kernel_size,
            "channel_multiplier": channel_multiplier,
            "activations": activations,
            "final_activation": final_activation,
            "spiking_feedforward_block_type": None,
            "spiking_neuron": None,
        }

        super().__init__(unet_kwargs)

        self.crop = None
        self.mask = mask_output
        self.norm_input = norm_input
        self.num_bins = num_bins
        self.num_encoders = num_encoders

    def init_cropping(self, width, height, safety_margin=0):
        self.crop = CropParameters(width, height, self.num_encoders, safety_margin)

    def resize(self, multires_out):
         # upsample disparity estimates to the original input resolution
        out_list = []
        for o in multires_out:
            out_list.append(
                F.interpolate(
                    o,
                    scale_factor=(
                        multires_out[-1].shape[2] / o.shape[2],
                        multires_out[-1].shape[3] / o.shape[3],
                    ),
                    mode='bilinear', 
                    align_corners=False
                )
            )

        # crop output
        if self.crop is not None:
            for i, d in enumerate(out_list):
                out_list[i] = out_list[:, :, self.crop.iy0 : self.crop.iy1, self.crop.ix0 : self.crop.ix1]
                out_list[i] = out_list[i].contiguous()
        
        return out_list

    def forward(self, x):
        """
        :param event_voxel: N x num_bins x H x W
        """

        # normalize input
        if self.norm_input:
           x = nonzero_normalize(x)

        # pad input
        if self.crop is not None:
            x = self.crop.pad(x)

        # x = self.direc(x)
        # forward pass
        ego_motion, disps, tmp = super().forward(x)
        disps = self.resize(disps)

        ego_motion = torch.concat([ego_motion[..., :3] * torch.pi  / 2 / 100, 
                                   ego_motion[..., 3:] / 100],
                                   dim=-1)

        return {"ego_motion": ego_motion,
                "disps": disps,
                "tmp": tmp}