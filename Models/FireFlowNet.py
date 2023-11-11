from .Modules.base import BaseModel, ConvLayer, ConvLayer_, ConvGRU, copy_states

class Model(BaseModel):
    """
    FireNet architecture (adapted for optical flow estimation), as described in the paper "Fast Image
    Reconstruction with an Event Camera", Scheerlinck et al., WACV 2020.
    """

    head_neuron = ConvLayer_
    ff_neuron = ConvLayer_
    rec_neuron = ConvLayer_
    residual = False
    num_recurrent_units = 7
    kwargs = [{}] * num_recurrent_units
    w_scale_pred = 0.01

    def __init__(self, num_bins, base_num_channels, kernel_size, encoding,  mask_output, activations, norm_input=False, **kwargs):
        super().__init__()
        self.num_bins = num_bins
        base_num_channels = base_num_channels
        kernel_size = kernel_size
        self.encoding = encoding
        self.norm_input = norm_input
        self.mask = mask_output
        ff_act, rec_act = activations
    
        self.head = self.head_neuron(self.num_bins, base_num_channels, kernel_size, activation=ff_act, **self.kwargs[0])

        self.G1 = self.rec_neuron(
            base_num_channels, base_num_channels, kernel_size, activation=rec_act, **self.kwargs[1]
        )
        self.R1a = self.ff_neuron(
            base_num_channels, base_num_channels, kernel_size, activation=ff_act, **self.kwargs[2]
        )
        self.R1b = self.ff_neuron(
            base_num_channels, base_num_channels, kernel_size, activation=ff_act, **self.kwargs[3]
        )

        self.G2 = self.rec_neuron(
            base_num_channels, base_num_channels, kernel_size, activation=rec_act, **self.kwargs[4]
        )
        self.R2a = self.ff_neuron(
            base_num_channels, base_num_channels, kernel_size, activation=ff_act, **self.kwargs[5]
        )
        self.R2b = self.ff_neuron(
            base_num_channels, base_num_channels, kernel_size, activation=ff_act, **self.kwargs[6]
        )

        self.pred = ConvLayer(
            base_num_channels, out_channels=2, kernel_size=1, activation="tanh", w_scale=self.w_scale_pred
        )

        self.reset_states()

    @property
    def states(self):
        return copy_states(self._states)

    @states.setter
    def states(self, states):
        self._states = states

    def detach_states(self):
        detached_states = []
        for state in self.states:
            if type(state) is tuple:
                tmp = []
                for hidden in state:
                    tmp.append(hidden.detach())
                detached_states.append(tuple(tmp))
            else:
                detached_states.append(state.detach())
        self.states = detached_states

    def reset_states(self):
        self._states = [None] * self.num_recurrent_units

    def init_cropping(self, width, height):
        pass

    def forward(self, event_voxel, event_cnt, log=False):
        """
        :param event_voxel: N x num_bins x H x W
        :param event_cnt: N x 4 x H x W per-polarity event cnt and average timestamp
        :param log: log activity
        :return: output dict with list of [N x 2 X H X W] (x, y) displacement within event_tensor.
        """

        # input encoding
        if self.encoding == "voxel":
            x = event_voxel
        elif self.encoding == "cnt" and self.num_bins == 2:
            x = event_cnt
        else:
            print("Model error: Incorrect input encoding.")
            raise AttributeError

        # normalize input
        if self.norm_input:
            mean, stddev = (
                x[x != 0].mean(),
                x[x != 0].std(),
            )
            x[x != 0] = (x[x != 0] - mean) / stddev

        # forward pass
        x1, self._states[0] = self.head(x, self._states[0])

        x2, self._states[1] = self.G1(x1, self._states[1])
        x3, self._states[2] = self.R1a(x2, self._states[2])
        x4, self._states[3] = self.R1b(x3, self._states[3], residual=x2 if self.residual else 0)

        x5, self._states[4] = self.G2(x4, self._states[4])
        x6, self._states[5] = self.R2a(x5, self._states[5])
        x7, self._states[6] = self.R2b(x6, self._states[6], residual=x5 if self.residual else 0)

        flow = self.pred(x7)

        # log activity
        if log:
            activity = {}
            name = [
                "0:input",
                "1:head",
                "2:G1",
                "3:R1a",
                "4:R1b",
                "5:G2",
                "6:R2a",
                "7:R2b",
                "8:pred",
            ]
            for n, l in zip(name, [x, x1, x2, x3, x4, x5, x6, x7, flow]):
                activity[n] = l.detach().ne(0).float().mean().item()
        else:
            activity = None

        return {"flow": [flow], "activity": activity}