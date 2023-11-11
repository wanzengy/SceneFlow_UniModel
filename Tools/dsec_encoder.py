"""
Adapted from Monash University https://github.com/TimoStoff/events_contrast_maximization
"""

import numpy as np
import torch


def events_to_mask(xs, ys, ps, sensor_size=(180, 240)):
    """
    Accumulate events into a binary mask.
    """

    device = xs.device
    img_size = list(sensor_size)
    mask = torch.zeros(img_size).to(device)
    x0, y0 = xs.to(dtype=int), ys.to(dtype=int)

    for xlim in [x0, x0 + 1]:
        for ylim in [y0, y0 + 1]:
            mask = (xlim < sensor_size[1]) & (xlim >= 0) & (ylim < sensor_size[0]) & (ylim >= 0)
            mask.index_put_((ylim[mask], xlim[mask]), 1., accumulate=False)

    return mask


def events_to_image(xs, ys, ps, sensor_size=(180, 240), accumulate=True):
    """
    Accumulate events into an image.
    """

    img = torch.zeros(list(sensor_size))
    x0, y0 = xs.to(dtype=int), ys.to(dtype=int)
    
    for xlim in [x0, x0 + 1]:
        for ylim in [y0, y0 + 1]:
            mask = (xlim < sensor_size[1]) & (xlim >= 0) & (ylim < sensor_size[0]) & (ylim >= 0)
            interp_weight = 1. * (1 - np.abs(xlim - xs)) * (1 - np.abs(ylim - ys))
            img.index_put_((ylim[mask], xlim[mask]), ps[mask] * interp_weight[mask], accumulate=True)

    return img


def events_to_voxel(xs, ys, ts, ps, num_bins, sensor_size=(180, 240), round_ts=False):
    """
    Generate a voxel grid from input events using temporal bilinear interpolation.
    """

    assert len(xs) == len(ys) and len(ys) == len(ts) and len(ts) == len(ps)

    voxel = []
    ts = ts * (num_bins - 1)

    if round_ts:
        ts = torch.round(ts)

    zeros = torch.zeros(ts.size())
    for b_idx in range(num_bins):
        weights = torch.max(zeros, 1.0 - torch.abs(ts - b_idx))
        voxel_bin = events_to_image(xs, ys, ps * weights, sensor_size=sensor_size)
        # voxel_bin = events_to_channels(xs, ys, ts * ps * (weights > 0), sensor_size=sensor_size)
        # voxel_bin = events_to_image(xs, ys, ps * (weights > 0), sensor_size=sensor_size)
        # voxel_bin = events_to_channels(xs, ys, ps * (weights > 0), sensor_size=sensor_size)
        voxel.append(voxel_bin)

    return torch.stack(voxel).reshape(-1, *sensor_size)


def events_to_channels(xs, ys, ps, sensor_size=(180, 240)):
    """
    Generate a two-channel event image containing event counters.
    """

    assert len(xs) == len(ys) and len(ys) == len(ps)

    if ps.min() == -1:
        ps = (ps + 1) / 2

    inds = (ps.long(), ys.long(), xs.long())
    img = torch.zeros([2] + list(sensor_size))
    img.index_put_(inds, torch.ones_like(ps), accumulate = True)

    return img


def events_to_timesurface(xs, ys, ts, ps, num_bins=2, sensor_size=(180, 240), polarity=True):
    ts = ts * num_bins
    if ps.min() == -1:
        ps = (ps + 1) / 2

    if polarity:
        pos_t = torch.zeros(list(sensor_size))
        pos_t.index_put_((xs, ys), ts * (ps > 0), accumulate=True)
        cimg = torch.zeros_like(pos_t)
        cimg.index_put_((xs, ys), ps * (ps > 0), accumulate = True)
        pos_t /= (cimg + 1e-9)

        neg_t = torch.zeros(list(sensor_size))
        neg_t.index_put_((xs, ys), ts * (ps < 0), accumulate=True)
        cimg = torch.zeros_like(pos_t)
        cimg.index_put_((xs, ys), - ps * (ps < 0), accumulate = True)
        neg_t /= (cimg + 1e-9)

        timg = torch.stack([pos_t, neg_t])
    else:
        timg = torch.zeros([2] + list(sensor_size))
        cimg = torch.zeros([2] + list(sensor_size))
        # x0, y0 = xs.to(dtype=int), ys.to(dtype=int)
        
        # for xlim in [x0, x0 + 1]:
        #     for ylim in [y0, y0 + 1]:
        #         mask = (xlim < sensor_size[1]) & (xlim >= 0) & (ylim < sensor_size[0]) & (ylim >= 0)
                # interp_weight = 1. * (1 - np.abs(xlim - xs)) * (1 - np.abs(ylim - ys))
                # timg.index_put_((ylim[mask], xlim[mask]), ts[mask] * interp_weight[mask], accumulate=True)
                # timg.index_put_((ylim[mask], xlim[mask]), ts[mask], accumulate=False)
                # cimg.index_put_((ylim[mask], xlim[mask]), torch.ones(1), accumulate=True)
           
        inds = (ps.long(), ys.long(), xs.long())
        timg.index_put_(inds, ts, accumulate = True)
        cimg.index_put_(inds, torch.ones_like(ps), accumulate = True)


        # ind = cimg > 0
        # timg[ind] = timg[ind] / cimg[ind]
        timg = timg / (cimg + 1e-9)
        # timg = timg[None, ...]

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
