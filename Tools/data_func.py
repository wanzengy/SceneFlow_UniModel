import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

def ev_to_2Dmap(xs, ys, ws, size=[180, 240], accumulate=True):
    """
    Accumulate events into an image according to the weight of each event.
    ws: the weight for each event
    """
    img = torch.zeros(size)

    x0 = xs.int()
    y0 = ys.int()
    for xlim in [x0, x0 + 1]:
        for ylim in [y0, y0 + 1]:
            mask = (ylim < size[-2]) & (xlim < size[-1]) & (xlim >= 0) & (ylim >= 0)              
            wlim = ws * (1 - abs(xlim - xs)) * (1 - abs((ylim - ys)))
            yi, xi, wi = ylim[mask], xlim[mask], wlim[mask]
            img.index_put_((yi.long(), xi.long()), wi, accumulate=accumulate)

    return img

def ev_to_3Dmap(xs, ys, ts, ws, size=[3, 180, 240], accumulate=True):
    """
    Accumulate events into an image according to the weight of each event.
    ws: the weight for each event
    """
    img = torch.zeros(size)

    x0 = xs.int()
    y0 = ys.int()
    for xlim in [x0, x0 + 1]:
        for ylim in [y0, y0 + 1]:
            mask = (ylim < size[-2]) & (xlim < size[-1]) & (xlim >= 0) & (ylim >= 0)              
            wlim = ws * (1 - abs(xlim - xs)) * (1 - abs((ylim - ys)))
            # ti, yi, xi, wi = ts[mask], ys[mask], xs[mask], ws[mask]
            ti, yi, xi, wi = ts[mask], ylim[mask], xlim[mask], wlim[mask]
            img.index_put_((ti.long(), yi.long(), xi.long()), wi, accumulate=accumulate)

    return img

def ev_to_channels(xs, ys, ps, size=[180, 240]):
    """
    Generate a two-channel event image containing event counters.
    """

    assert len(xs) == len(ys) == len(ps)

    return torch.stack([ev_to_2Dmap(xs, ys, ps * (ps > 0), size), 
                        ev_to_2Dmap(xs, ys, - ps * (ps < 0), size)])

def ev_to_voxel(xs, ys, ts, ps, num_bins, size=[180, 240], round_ts=False):
    """
    Generate a voxel grid from input events using temporal bilinear interpolation.
    """

    assert len(xs) == len(ys) == len(ts) == len(ps)

    ts = ts * (num_bins - 1)
    if round_ts:
        ts = torch.round(ts)

    vox = torch.zeros([num_bins, ] + size)
    for b_idx in range(num_bins):
        weights = torch.clamp(1.0 - torch.abs(b_idx - ts), min=0)
        vox[b_idx] = ev_to_2Dmap(xs, ys, ps * weights, size)

    return vox

def ev_to_timesurface(xs, ys, ts, ps, num_bins, size=[180, 240]):

    assert len(xs) == len(ys) == len(ts) == len(ps)
    ts = ts * num_bins
    size = (num_bins,) + tuple(size)
    
    ts[ts == num_bins] = num_bins - 1e-7
    timg = ev_to_3Dmap(xs, ys, ts, ts - ts.long(), size)
    cimg = ev_to_3Dmap(xs, ys, ts, torch.ones_like(ps), size)

    timg = timg / (cimg + 1e-9)
    return timg


def get_hot_event_mask(event_rate, idx, max_px=100, min_obvs=5, max_rate=0.8):
    """
    Returns binary mask to remove events from hot pixels.
    """

    mask = torch.ones(event_rate.shape).to(event_rate.device)
    if idx > min_obvs:
        for i in range(max_px):
            argmax = torch.argmax(event_rate)
            index = (torch.div(argmax, event_rate.shape[1], rounding_mode='trunc'), argmax % event_rate.shape[1])
            if event_rate[index] > max_rate:
                event_rate[index] = 0
                mask[index] = 0
            else:
                break
    return mask

def event_formatting(xs, ys, ts, ps):
    """
    Reset sequence-specific variables.
    :param xs: [N] numpy array with event x location
    :param ys: [N] numpy array with event y location
    :param ts: [N] numpy array with event timestamp
    :param ps: [N] numpy array with event polarity ([-1, 1])
    :return xs: [N] tensor with event x location
    :return ys: [N] tensor with event y location
    :return ts: [N] tensor with normalized event timestamp
    :return ps: [N] tensor with event polarity ([-1, 1])
    """

    xs = torch.from_numpy(xs.astype(np.float32))
    ys = torch.from_numpy(ys.astype(np.float32))
    ts = torch.from_numpy(ts.astype(np.float32))
    ps = torch.from_numpy(ps.astype(np.float32))
    if min(ps) == 0:
        ps = 2 * ps - 1
    ts = (ts - ts[0]) / (ts[-1] - ts[0])
    return xs, ys, ts, ps

def binary_search_array(array, x, left=None, right=None, side="left"):
    """
    Binary search through a sorted array.
    """

    left = 0 if left is None else left
    right = len(array) - 1 if right is None else right
    mid = left + (right - left) // 2

    if left > right:
        return left if side == "left" else right

    if array[mid] == x:
        return mid

    if x < array[mid]:
        return binary_search_array(array, x, left=left, right=mid - 1)

    return binary_search_array(array, x, left=mid + 1, right=right)

def delta_time(ts, window):
    floor_row = int(np.floor(ts))
    ceil_row = int(np.ceil(ts + window))
    if ceil_row - floor_row > 1:
        floor_row += ceil_row - floor_row - 1

    idx0_change = ts - floor_row
    idx1_change = ts + window - floor_row

    delta_idx = event_idx1 - event_idx0
    event_idx1 = int(event_idx0 + idx1_change * delta_idx)
    event_idx0 = int(event_idx0 + idx0_change * delta_idx)
    return event_idx0, event_idx1

def create_polarity_mask(ps):
    """
    Creates a two channel tensor that acts as a mask for the input event list.
    :param ps: [N] tensor with event polarity ([-1, 1])
    :return [N x 2] event representation
    """

    inp_pol_mask = torch.stack([ps, ps])
    inp_pol_mask[0, :][inp_pol_mask[0, :] < 0] = 0
    inp_pol_mask[1, :][inp_pol_mask[1, :] > 0] = 0
    inp_pol_mask[1, :] *= -1
    return inp_pol_mask

def augment_frames(img, augmentation):
    """
    Augment APS frame with horizontal and vertical flips.
    :param img: [H x W] numpy array with APS intensity
    :param batch: batch index
    :return img: [H x W] augmented numpy array with APS intensity
    """
    if "Horizontal" in augmentation:
        img = np.flip(img, 1)
    if "Vertical" in augmentation:
        img = np.flip(img, 0)
    return img

def augment_flowmap(flowmap, augmentation):
    """
    Augment ground-truth optical flow map with horizontal and vertical flips.
    :param flowmap: [2 x H x W] numpy array with ground-truth (x, y) optical flow
    :param batch: batch index
    :return flowmap: [2 x H x W] augmented numpy array with ground-truth (x, y) optical flow
    """
    if "Horizontal" in augmentation:
        flowmap = np.flip(flowmap, 2)
        flowmap[0, :, :] *= -1.0
    if "Vertical" in augmentation:
        flowmap = np.flip(flowmap, 1)
        flowmap[1, :, :] *= -1.0
    return flowmap

def augment_events(xs, ys, ps, augmentation=["Horizontal", "Vertical", "Polarity", "VariNum"], resolution=[255, 255]):
    """
    Augment event sequence with horizontal, vertical, and polarity flips, and
    artificial event pauses.
    :return xs: [N] tensor with event x location
    :return ys: [N] tensor with event y location
    :return ps: [N] tensor with event polarity ([-1, 1])
    :param batch: batch index
    :return xs: [N] tensor with augmented event x location
    :return ys: [N] tensor with augmented event y location
    :return ps: [N] tensor with augmented event polarity ([-1, 1])
    """

    for i, mechanism in enumerate(augmentation):
        if mechanism == "Horizontal":
            xs = resolution[1] - 1 - xs
        elif mechanism == "Vertical":
            ys = resolution[0] - 1 - ys
        elif mechanism == "Polarity":
            ps *= -1
            # ts = ts[-1] - ts
        elif mechanism == 'Transpose':
            xs, ys = ys, xs

        # # shared among batch elements
        # elif (
        #     batch == 0
        #     and mechanism == "Pause"
        #     and tc_idx > config["loss"]["reconstruction_tc_idx_threshold"]
        # ):
        #     if augmentation["Pause"]:
        #         if np.random.random() < config["loader"]["augment_prob"][i][1]:
        #             self.batch_augmentation["Pause"] = False
        #     elif np.random.random() < config["loader"]["augment_prob"][i][0]:
        #             self.batch_augmentation["Pause"] = True

    return xs, ys, ps

def custom_collate(batch):
    """
    Collects the different event representations and stores them together in a dictionary.
    """
    batch_dict = {k:[] for k in batch[0].keys()}
    for entry in batch:
        for k, v in entry.items():
            batch_dict[k].append(v)
    for k, v in batch_dict.items():
        if type(v[0]) is torch.Tensor:
            # if k == 'event_list':
            #     batch_dict[k] = pad_sequence(v, batch_first=True)
            # else:
            batch_dict[k] = torch.stack(v)

    # events = []
    # for i, d in enumerate(batch_dict["event_list"]):
    #     ev = np.concatenate([d, i*np.ones((len(d),1), dtype=np.float32)],1)
    #     events.append(ev)
    # batch_dict["event_list_unroll"] = torch.from_numpy(np.concatenate(events,0))
    return batch_dict


def stereo_collate(batch):
    """
    Collects the stereo event representations and stores them together in a dictionary.
    """
    batch_dict = {}
    for k in batch[0].keys():
        batch_dict[k] = []
        if k in ['left', 'right', 'gt']:
            batch_dict[k] = {sk:[] for sk in batch[0][k].keys()}

    for entry in batch:
        for k, v in entry.items():
            if k in ['left', 'right', 'gt']:
                for sk, sv in v.items():
                    batch_dict[k][sk].append(sv)
            else:
                batch_dict[k].append(v)

    for k, v in batch_dict.items():
        if k in ['left', 'right', 'gt']:
            for sk, sv in v.items():
                if type(sv[0]) is torch.Tensor:
                    batch_dict[k][sk] = torch.stack(sv)
        elif type(v[0]) is torch.Tensor:
            # if k == 'event_list':
            #     batch_dict[k] = pad_sequence(v, batch_first=True)
            # else:
            batch_dict[k] = torch.stack(v)

    return batch_dict