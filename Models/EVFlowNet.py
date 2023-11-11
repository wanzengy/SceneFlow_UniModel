import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Modules.base import CropParameters, nonzero_normalize
from .Modules.unet import MultiResUNet

class Model(MultiResUNet):
    """
    Recurrent version of the EV-FlowNet architecture from the paper "EV-FlowNet: Self-Supervised Optical
    Flow for Event-based Cameras", Zhu et al., RSS 2018.
    """
    def __init__(self,
                norm=None,
                use_upsample_conv=True, 
                base_num_channels=32, 
                num_encoders=4,
                num_residual_blocks=2,
                num_output_channels=2,
                num_bins=2,
                norm_input=True,
                skip_type = 'concat',
                kernel_size=3,
                channel_multiplier=2,
                activations=["relu", None],
                final_activation="tanh",
                mask_output=True,
                **kwargs
                ):
        super().__init__(base_num_channels, num_encoders, num_residual_blocks, num_output_channels,
            skip_type, norm, use_upsample_conv, num_bins, None, kernel_size,
            channel_multiplier, activations, None, None)

        self.crop = None
        self.mask = mask_output
        self.norm_input = norm_input
        self.num_bins = num_bins
        self.num_encoders = num_encoders

    def init_cropping(self, width, height, safety_margin=0):
        self.crop = CropParameters(width, height, self.num_encoders, safety_margin)

    def flow_resize(self, multires_flow):
         # upsample flow estimates to the original input resolution
        flow_list = []
        for flow in multires_flow:
            flow_list.append(
                torch.nn.functional.interpolate(
                    flow,
                    scale_factor=(
                        multires_flow[-1].shape[2] / flow.shape[2],
                        multires_flow[-1].shape[3] / flow.shape[3],
                    ),
                )
            )

        # crop output
        if self.crop is not None:
            for i, flow in enumerate(flow_list):
                flow_list[i] = flow[:, :, self.crop.iy0 : self.crop.iy1, self.crop.ix0 : self.crop.ix1]
                flow_list[i] = flow_list[i].contiguous()
        
        return flow_list

    def forward(self, x):
        """
        :param event_voxel: N x num_bins x H x W
        :param event_cnt: N x 4 x H x W per-polarity event cnt and average timestamp
        :return: output dict with list of [N x 2 X H X W] (x, y) displacement within event_tensor.
        """

        # normalize input
        if self.norm_input:
           x = nonzero_normalize(x)

        # pad input
        if self.crop is not None:
            x = self.crop.pad(x)

        # x = self.direc(x)
        # forward pass
        multires_flow = super().forward(x)
        multires_flow = self.flow_resize(multires_flow)

        return {"flow": multires_flow}