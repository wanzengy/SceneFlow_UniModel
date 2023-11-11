import os
import sys

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch3d.transforms as t3d

from Tools.iwe import get_interpolation, interpolate

def warp_event(events, flow, tref, resolution, flow_scaling,  weights=None, round_idx=False):
    warp_pos, warp_weights = get_interpolation(events, flow, tref, resolution, flow_scaling, round_idx=False)

    if weights is not None:
        warp_weights *= weights

    iwe = torch.stack([
            interpolate(warp_pos, warp_weights, resolution, events[..., -1] == 1),
            interpolate(warp_pos, warp_weights, resolution, events[..., -1] == -1)], dim=1)

    return iwe


def warp_disparity(img, disp):
    batch_size, _, height, width = img.size()

    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, width).repeat(batch_size,
                height, 1).type_as(img)
    y_base = torch.linspace(0, 1, height).repeat(batch_size,
                width, 1).transpose(1, 2).type_as(img)

    # Apply shift in X direction
    x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel and normalized
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
    # In grid_sample coordinates are assumed to be between -1 and 1

    output = F.grid_sample(img, 2 * flow_field - 1, mode='bilinear', padding_mode='zeros')
    return output

def warp_contrast(tref, flow, events, resolution=(128, 128), flow_scaling=128):
    '''
    Warp Image contrast maximization Loss
    Input:
        tref: target time
        max_ts: max timestamp
        flow: event flow [batch_size x N x 2] (vx, vy)
        events: [batch_size x N x 4] (ts, x, y, p)
        pol_mask: polarity mask of events [batch_size x N x 2]
        resolution: resolution of the image space (int, int)
        flow_scaling: scalar that multiplies the optical flow map
    '''
    
    iwe = warp_event(events, flow, tref, resolution, flow_scaling)

    ts = events[..., 0:1]
    ts_iwe = warp_event(events, flow, tref, resolution, flow_scaling, weights=ts)
    
    ts_iwe /= (iwe + 1e-9)
    loss = (ts_iwe ** 2).sum(dim=(1, 2, 3))
    
    non_zero = iwe[:, 0] + iwe[:, 1]
    non_zero[non_zero > 0] = 1
    loss /= torch.sum(non_zero, dim=(1, 2))
    return loss
    # loss = torch.sum((iwe - iwe.mean(dim=(-1, -2), keepdim=True)) ** 2, dim=(1, 2, 3))
    # if loss_scaling:
    #     loss /= torch.sum((iwe[:, 0] + iwe[:, 1]) > 0, dim=(1, 2))
    # return loss.mean()

class census_transform(nn.Module):
    '''
    Census Transform
    Input:
        img: image [batch_size x C x H x W]
        kernel_size: size of the kernel
    '''
    def __init__(self, kernel_size=3):
        super(census_transform, self).__init__()
        self.kernel_size = kernel_size
        w = torch.eye(kernel_size * kernel_size)
        w = w.reshape(kernel_size * kernel_size, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(w, requires_grad=False)
    
    def forward(self, img):
        c = img.size(1)
        weights = self.weights.repeat(c, 1, 1, 1)

        census = F.conv2d(img, weights, bias=None, stride=1, padding=self.kernel_size//2, dilation=1, groups=c)
        img = img[:, None].repeat(1, self.kernel_size * self.kernel_size, 1, 1, 1).permute(0, 2, 1, 3, 4)
        img = img.reshape(img.size(0), -1, img.size(3), img.size(4))
        census = (census - img) > 0
        return census.float()

def charbonnier_loss(x, a = 0.5, mask=None):
    x = x ** 2
    if mask is not None:
        result = torch.pow(x + 1e-6, a)
        result[~mask] = 0
        return result
    else:
        return torch.pow(x + 1e-6, a)
    # x = (x ** 2)
    # x = x * mask if mask is not None else x
    # return torch.pow(x[x > 0], a)

def hamming_distanc(t1, t2):
    dist = (t1 - t2) ** 2
    dist = dist / (dist + 1e-9)
    dist = dist.sum(dim=1)
    return dist

def spatial_variance(x):
    return torch.var(
        x.view(
            x.shape[0],
            -1,
        ),
        dim=1,
        keepdim=True,
    )

def smoothness_loss(flow, event_cnt):
    mask = event_cnt.sum(dim=2, keepdim=True) > 0
    mask = mask & torch.isfinite(flow) & ~torch.isnan(flow)

    flow_lr = (flow[..., :-1] - flow[..., 1:])
    flow_ud = (flow[..., :-1, :] - flow[..., 1:, :])
    flow_lurd = (flow[..., :-1, :-1] - flow[..., 1:, 1:])
    flow_ldru = (flow[..., :-1, 1:] - flow[..., 1:, -1:])

    loss = []
    # non_zero = torch.sum(mask)
    # loss.append(charbonnier_loss(flow_lr, mask=(mask[..., :-1] & mask[..., 1:])).sum() / non_zero)
    # loss.append(charbonnier_loss(flow_ud, mask=(mask[..., :-1, :] & mask[..., 1:, :])).sum() / non_zero)
    # loss.append(charbonnier_loss(flow_lurd, mask=(mask[..., :-1, :-1] & mask[..., 1:, 1:])).sum() / non_zero)
    # loss.append(charbonnier_loss(flow_ldru, mask=(mask[..., :-1, 1:] & mask[..., 1:, -1:])).sum() / non_zero)

    loss.append(charbonnier_loss(flow_lr).mean())
    loss.append(charbonnier_loss(flow_ud).mean())
    loss.append(charbonnier_loss(flow_lurd).mean())
    loss.append(charbonnier_loss(flow_ldru).mean())

    if flow.size(1) > 1:
        flow_dt = flow[:, :-1] - flow[:, 1:]
        loss.append(charbonnier_loss(flow_dt,)).mean()
    
    return sum(loss) / len(loss)

def consistency_loss(disp_left_maps, disp_right_maps):
    """
    Consistency loss, as described in Section 3.4 of the paper 'Unsupervised Event-based Learning
    of Optical Flow, Depth, and Egomotion', Zhu et al., CVPR'19.
    The consistency loss is the minimization of the per-pixel and per-polarity image of the squared
    disparity difference between the left and right disparity estimates.
    """
    loss = []
    for i in range(disp_left_maps.size(1)):
        disp_r_warp_l = warp_disparity(disp_right_maps[:, i], disp_left_maps[:, i])
        mask = torch.isfinite(disp_left_maps[:, i]) & torch.isfinite(disp_r_warp_l)
        loss.append(torch.sum(torch.abs(disp_r_warp_l + disp_left_maps[:, i])[mask]) / mask.sum())
    loss = sum(loss) / len(loss)
    return loss

def temporal_loss(event_list,
                  event_cnt,
                  flow_maps,
                  flow_list,
                  max_ts,
                  resolution,
                  flow_scaling,
                  flow_regul_weight,
                  smooth=False):
    loss = {'total': [],
        'fw': [],
        'bw': [],
        'smooth': []}
    
    for i in range(len(flow_maps)):
        fw = warp_contrast(max_ts, flow_list[i], event_list, resolution, flow_scaling)
        loss['fw'].append(fw.sum() / (max_ts ** 2))

        bw = warp_contrast(0, flow_list[i], event_list, resolution, flow_scaling)
        loss['bw'].append(bw.sum() / (max_ts ** 2))

        tot = loss['fw'][i] + loss['bw'][i]

        if smooth:
            fs = smoothness_loss(flow_maps[i], event_cnt)
            loss['smooth'].append(fs)
            tot += flow_regul_weight * loss['smooth'][i]
    
        loss['total'].append(tot)

    tot = sum(loss['total']) / len(loss['total'])
    return tot

def stereo_loss(event_stereo_list,
                flow_stereo_list,
                disp_stereo_maps,
                max_ts,
                resolution,
                flow_scaling,
                kernel_size=3,
                ):
    """
    Stereo loss, as descibed in section 3.4 of the paper 'Unsupervised Event-based Learning of Optical Flow, 
    Depth, and Egomotino', Zhu et al., CVPR'19.
    The stereo loss is the minimization of the per-pixel census transforms of event cnt from left and right.
    """
    
    loss = {'left':[], 'right':[]}
    ct = census_transform(kernel_size=kernel_size).to(event_stereo_list['left'].device)
    for i in range(len(flow_stereo_list['left'])):
        left_iwe = warp_event(event_stereo_list['left'], 
                              flow_stereo_list['left'][i],
                              max_ts['left'], 
                              resolution, 
                              flow_scaling)
        
        right_iwe = warp_event(event_stereo_list['right'],
                               flow_stereo_list['right'][i],
                               max_ts['right'],
                               resolution,
                               flow_scaling)
        
        census_left = ct(left_iwe)
        census_right = ct(right_iwe)

        left_nonzero = (left_iwe > 0).sum()
        right_nonzero = (right_iwe > 0).sum()


        for j in range(disp_stereo_maps['left'][i].size(1)):
            census_r_warp_l = warp_disparity(census_right, -disp_stereo_maps['left'][i][:, j])
            census_l_warp_r = warp_disparity(census_left, disp_stereo_maps['right'][i][:, j])

            loss['left'].append((charbonnier_loss(census_left - census_r_warp_l)).sum() / left_nonzero)
            loss['right'].append((charbonnier_loss(census_right - census_l_warp_r)).sum() / right_nonzero)
    
    loss['left'] = sum(loss['left']) / len(loss['left'])
    loss['right'] = sum(loss['right']) / len(loss['right'])

    return loss, left_iwe, right_iwe, census_left, census_right, census_r_warp_l


class EventWarping(nn.Module):
    """
    Contrast maximization loss, as described in Section 3.2 of the paper 'Unsupervised Event-based Learning
    of Optical Flow, Depth, and Egomotion', Zhu et al., CVPR'19.
    The contrast maximization loss is the minimization of the per-pixel and per-polarity image of averaged
    timestamps of the input events after they have been compensated for their motion using the estimated
    optical flow. This minimization is performed in a forward and in a backward fashion to prevent scaling
    issues during backpropagation.
    """

    def __init__(self, 
                left_intrinsic, 
                right_intrinsic,
                baseline=10,
                focal=199,
                resolution=[256, 256],
                flow_regul_weight=0.1,
                stereo_weight=1.0,
                consistency_weight=0.1,
                smooth_weight=0.2,
                flow_scaling=None,
                mask_output=True,
                overwrite_intermediate=False,
                **kwargs):
        super(EventWarping, self).__init__()
        self.res = (resolution, ) * 2 if type(resolution) == int else resolution
        self.flow_scaling = flow_scaling if flow_scaling else max(resolution)
        self.mask_output = mask_output
        self.overwrite_intermediate = overwrite_intermediate

        self.flow_regul_weight = flow_regul_weight
        self.stereo_weight = stereo_weight
        self.consistency_weight = consistency_weight
        self.smooth_weight = smooth_weight
        
        self.intrinsic = {'left': left_intrinsic, 'right':right_intrinsic}
        self.baseline = baseline
        self.focal = focal

        self.reset()
        self.set_xy_map()

    def reset(self):
        self._passes = {'left':0, 'right':0}
        self._event_cnt = {'left':[], 'right':[]}
        self._event_list = {'left':[], 'right':[]}
        self._flow_maps = {'left':[], 'right':[]}
        self._flow_list = {'left':[], 'right':[]}
        self._disp_maps = {'left':[], 'right':[]}
        self._ego_motion = {'left':[], 'right':[]}

        self._gt_flow = {'left':None, 'right':None}

        self._out_ts_map = []
        self._inp_ts_map = []

    def set_xy_map(self):
        self.x_map = nn.ParameterDict({'left':None, 'right':None})
        self.y_map = nn.ParameterDict({'left':None, 'right':None})

        for loc in ['left', 'right']:
            # y_inds, x_inds = torch.meshgrid(torch.arange(self.res[0]),
            #                             torch.arange(self.res[1]))
            y_inds, x_inds = torch.meshgrid(torch.arange(260),
                                        torch.arange(346))

            Pfx, Ppx, Tx, Pfy, Ppy, Ty = self.intrinsic[loc][0][0],\
                                        self.intrinsic[loc][0][2],\
                                        self.intrinsic[loc][0][3],\
                                        self.intrinsic[loc][1][1],\
                                        self.intrinsic[loc][1][2],\
                                        self.intrinsic[loc][1][3]

            x_inds = x_inds - Ppx
            x_inds = x_inds * (1. / Pfx)

            y_inds = y_inds - Ppy
            y_inds = y_inds * (1. / Pfy)

            self.x_map[loc] = nn.Parameter(x_inds[2:258, 45:301], requires_grad=False)
            self.y_map[loc] = nn.Parameter(y_inds[2:258, 45:301], requires_grad=False)

    # def compute_of(self, x_map, y_map, R, T, depth, intrinsic):
    #     '''
    #      x                    x
    #        = K pi( R d K^-1 ( y ) + T) 
    #      y                    1  
        
    #     '''
    #     # flat_depth[np.logical_or(np.isclose(flat_depth,0.0), flat_depth<0.)]

    #     Pfx, Ppx, Tx, Pfy, Ppy, Ty = intrinsic[0][0], intrinsic[0][2], intrinsic[0][3],\
    #                                 intrinsic[1][1], intrinsic[1][2], intrinsic[1][3]
        
    #     mask = torch.logical_and(torch.isfinite(depth), (depth > 0))

    #     cam_coor = torch.stack((x_map, y_map, torch.ones_like(x_map)), dim=0)[None, ...] * depth
    #     cam_coor[:, 0] = cam_coor[:, 0] - Tx / Pfx
    #     cam_coor[:, 1] = cam_coor[:, 1] - Ty / Pfy
        
    #     cam_coor = torch.einsum('bij, bjnm->binm', R, cam_coor) + T[..., None, None]

    #     x_new = (cam_coor[:, 0] + Tx / Pfx) / cam_coor[:, 2]
    #     y_new = (cam_coor[:, 1] + Ty / Pfy) / cam_coor[:, 2]

    #     dx = x_new - x_map
    #     dy = y_new - y_map

    #     dx[~mask[:, 0]] = 0
    #     dy[~mask[:, 0]] = 0
    #     # dx[abs(dx) > 1] = 0
    #     # dy[abs(dy) > 1] = 0
        
    #     flow = torch.stack([dx * Pfx, dy * Pfy], dim=-1)
        
    #     return flow

    def compute_of(self, x_map, y_map, R, T, depth, intrinsic):
        '''
        Vx      -1/Z    0     x/Z    xy  -(1+x^2)  y      V
            = [                                      ] [   ]
        Vy        0   -1/Z    y/Z  1+y^2   -xy    -x      w
        
        '''
        # flat_depth[np.logical_or(np.isclose(flat_depth,0.0), flat_depth<0.)]

        Pfx, Ppx, Tx, Pfy, Ppy, Ty = intrinsic[0][0], intrinsic[0][2], intrinsic[0][3],\
                                    intrinsic[1][1], intrinsic[1][2], intrinsic[1][3]
        
        mask = torch.logical_and(torch.isfinite(depth), (depth > 0))
        re_Z =  1 / (depth + 1e-9)
        re_Z[~mask] = 0
        re_Z = re_Z[:, 0]

        b, h, w = re_Z.size()
                         
        coeff_mat = torch.zeros((b, h, w, 2, 6)).to(depth)

        coeff_mat[..., 0, 0] = - re_Z
        coeff_mat[..., 0, 2] = x_map * re_Z
        coeff_mat[..., 0, 3] = x_map * y_map
        coeff_mat[..., 0, 4] = -(1 + x_map ** 2)
        coeff_mat[..., 0, 5] = y_map

        coeff_mat[..., 1, 1] = - re_Z
        coeff_mat[..., 1, 2] = y_map * re_Z
        coeff_mat[..., 1, 3] = 1 + y_map ** 2
        coeff_mat[..., 1, 4] = - x_map * y_map
        coeff_mat[..., 1, 5] = -x_map

        # flow = torch.matmul(coeff_mat, )[..., None])[..., 0]
        flow = torch.einsum('bi, bnmji->bnmj', torch.concat((T, R), axis=1), coeff_mat)

        flow[..., 0] = flow[..., 0] * Pfx
        flow[..., 1] = flow[..., 1] * Pfy
        
        return flow

    def event_association(self, event_list, event_cnt, disp_list, ego_motion, loc='left', **kwargs):
        """
        :param flow_list: [batch_size x N x 2] list of optical flow (x, y) map
        :param event_list: [batch_size x N x 4] input events (ts, x, y, p)
        """
    
        _event_cnt = self._event_cnt[loc]
        _event_list = self._event_list[loc]
        _flow_maps = self._flow_maps[loc]
        _flow_list = self._flow_list[loc]
        _disp_maps = self._disp_maps[loc]
        _ego_motion = self._ego_motion[loc]

        angle = ego_motion[..., :3]
        # angle = angle * 0.05
        # R = t3d.euler_angles_to_matrix(angle, convention='XYZ') # radian(pi)
        R = angle
        T = ego_motion[..., 3:]

        flow_idx = event_list[:, :, 1:3].clone()
        flow_idx[:, :, 1] *= self.res[1]  # torch.view is row-major
        flow_idx = torch.sum(flow_idx, dim=2, keepdim=True).long().repeat(1, 1, 2) # B x N x 2

        disp_maps = []
        flow_maps = []
        eventflow_list = []

        # get optical flow from ego-motion and depth
        # get flow for every event in the list and update flow maps
        flow = kwargs['of'].to(dtype=torch.float32)
        self._gt_flow[loc] = flow

        for i, disp in enumerate(disp_list):
            depth = self.focal / (disp * self.flow_scaling + 1e-9) * self.baseline
            flow = self.compute_of(self.x_map[loc], self.y_map[loc], R, T, depth, self.intrinsic[loc])

            flow_maps.append(flow[:, None])
            flow = flow.view(flow.shape[0], 2, -1).permute(0, 2, 1) # B x (HW) x 2
            # event_flow = flow[batch_idx, flow_idx, :]
            event_flow = torch.gather(flow, 1, flow_idx)
            eventflow_list.append(event_flow)
            
            # eventflow_list.append(torch.zeros_like(event_flow))

            disp_maps.append(disp[:, None])

        ''' 
        Test GT
        '''
        # for i, disp in enumerate(disp_list):
        #     depth = disp[:, None]
        #     flow = self.compute_of(self.x_map[loc], self.y_map[loc], R, T, depth, self.intrinsic[loc])
        #     flow_maps.append(flow[:, None])
            
        #     # flow = kwargs['of'].to(dtype=torch.float32)
        #     # flow[~torch.isnan(depth[:, 0])] = 0
        
        #     flow = flow.view(flow.shape[0], 2, -1).permute(0, 2, 1) # B x (HW) x 2
        #     # event_flow = flow[batch_idx, flow_idx, :]
        #     event_flow = torch.gather(flow, 1, flow_idx)
        #     eventflow_list.append(event_flow)

        #     # eventflow_list.append(torch.zeros_like(event_flow))

        #     disp = self.focal * self.baseline / depth / self.res[1]
        #     # disp = torch.ones_like(disp) / 346
        #     if loc == 'left':
        #         disp_maps.append(disp[:, None])
        #     else:
        #         y_inds, x_inds = torch.meshgrid(torch.arange(self.res[0]),
        #                                 torch.arange(self.res[1]))

        #         right_disps = torch.zeros_like(disp)
        #         right_disps[...] = torch.nan
        #         right_x = x_inds.to(disp) - disp
        #         right_x = right_x.int()
        #         mask = torch.isfinite(right_x) & (right_x >= 0) & (right_x < self.res[1])

        #         batch_idx = torch.zeros_like(right_x)
        #         batch_idx[1] = 1

        #         depth_idx = torch.zeros_like(right_x)

        #         batch_idx = batch_idx[mask].long()
        #         depth_idx = depth_idx[mask].long()
        #         x_inds = x_inds.repeat(2, 1, 1)[:, None][mask].long()
        #         y_inds = y_inds.repeat(2, 1, 1)[:, None][mask].long()
        #         right_x = right_x[mask].long()

        #         right_disps[batch_idx, depth_idx, y_inds, right_x] = disp[batch_idx, depth_idx, y_inds, x_inds]
        #         disp_maps.append(right_disps[:, None])
  
        if len(_disp_maps) == 0:
            _disp_maps = disp_maps
        else:
            _disp_maps = [torch.cat([_disp_maps[i], disp], dim=1) for i, disp in enumerate(disp_maps)]

        if len(_flow_maps) == 0:
            _flow_maps = flow_maps
        else:
            _flow_maps = [torch.cat([_flow_maps[i], flow], dim=1) for i, flow in enumerate(flow_maps)]

        if len(_ego_motion) == 0:
            _ego_motion = ego_motion[:, None]
        else:
            _ego_motion = torch.cat([_ego_motion, ego_motion[:, None]], dim=1)


        if len(_flow_list) == 0:
            _flow_list = eventflow_list
        else:
            _flow_list = [torch.cat([_flow_list[i], flow], dim=1) for i, flow in enumerate(eventflow_list)]
              
        if len(_event_cnt) == 0:
            _event_cnt = event_cnt[:, None]
        else:
            _event_cnt = torch.cat([_event_cnt, event_cnt[:, None]], dim=1)

        # update internal event list
        event_list[:, :, 0:1] += self._passes[loc] # only nonzero second time
        if len(_event_list) == 0:
            _event_list = event_list
        else:
            _event_list = torch.cat([_event_list, event_list], dim=1)

        # if len(self._out_ts_map) == 0:
        #     self._out_ts_map = out_ts[:, None]
        # else:
        #     self._out_ts_map = torch.cat([self._out_ts_map, out_ts[:, None]], dim=1)

        # if len(self._inp_ts_map) == 0:
        #     self._inp_ts_map = inp_ts[:, None]
        # else:
        #     self._inp_ts_map = torch.cat([self._inp_ts_map, inp_ts[:, None]], dim=1)

        # update timestamp index

        self._event_cnt[loc] = _event_cnt
        self._event_list[loc] = _event_list
        self._flow_maps[loc] = _flow_maps
        self._flow_list[loc] = _flow_list
        self._disp_maps[loc] = _disp_maps
        self._ego_motion[loc] = _ego_motion
        self._passes[loc] += 1

    def overwrite_intermediate(self, disp_list, ego_motion, loc='left'):
        """
        :param flow_maps: [[batch_size x 2 x H x W]] list of optical flow (x, y) maps
        """
        _flow_list = self._flow_list[loc]

        R = t3d.euler_angles_to_matrix(ego_motion[..., :3], convention='ZXY')
        T = ego_motion[..., 3:]

        flow_idx = self._event_list[loc][:, :, 1:3].clone()
        flow_idx[:, :, 1] *= self.res[1]  # torch.view is row-major
        flow_idx = torch.sum(flow_idx, dim=2) # B x N x 1

        _flow_list = []
        # get flow for every event in the list and update the flow map
        for i, disp in enumerate(disp_list):
            depth = self.focal / disp * self.baseline
            flow = self.compute_of(self.x_map[loc], self.y_map[loc], R, T, depth, self.intrinsic[loc])
            flow = flow.view(flow.shape[0], 2, -1).permute(0, 2, 1) # B x (HW) x 2
            event_flow = torch.gather(flow, 1, flow_idx.repeat(1, 1, 2))
            _flow_list.append(event_flow)

    @property
    def num_events(self):
        return self._event_list['left'].size(1)

    @property
    def max_ts(self):
        return self._passes

    def test_forward(self, loc):
        loss = []
        mask = self._gt_flow[loc][:, None] > 0
        for i in range(len(self._flow_maps[loc])):
            loss.append(torch.abs(self._flow_maps[loc][i] - self._gt_flow[loc][:, None])[mask].mean())
        
        return sum(loss) / len(loss)

    def forward(self):
        loss = {'left':{}, 'right':{}}
        
        loss['left']['temp'] = temporal_loss(event_list=self._event_list['left'],
                                            event_cnt=self._event_cnt['left'],
                                            flow_maps=self._flow_maps['left'],
                                            flow_list=self._flow_list['left'],
                                            max_ts=self.max_ts['left'],
                                            resolution=self.res,
                                            flow_scaling=1,
                                            flow_regul_weight=self.flow_regul_weight,
                                            smooth=False)
    
        loss['right']['temp'] = temporal_loss(event_list=self._event_list['right'],
                                            event_cnt=self._event_cnt['right'],
                                            flow_maps=self._flow_maps['right'],
                                            flow_list=self._flow_list['right'],
                                            max_ts=self.max_ts['right'],
                                            resolution=self.res,
                                            flow_scaling=1,
                                            flow_regul_weight=self.flow_regul_weight,
                                            smooth=False)

        loss['left']['smooth'] = [smoothness_loss(disp, self._event_cnt['left']) for disp in self._disp_maps['left']]
        loss['left']['smooth'] = sum(loss['left']['smooth']) / len(loss['left']['smooth'])
        loss['right']['smooth'] = [smoothness_loss(disp, self._event_cnt['right']) for disp in self._disp_maps['right']]
        loss['right']['smooth'] = sum(loss['right']['smooth']) / len(loss['right']['smooth'])

        # network output only the absoulute disparity
        loss['left']['consistensy'] = [consistency_loss(-l, r) for l, r in zip(self._disp_maps['left'], self._disp_maps['right'])]
        loss['left']['consistensy'] = sum(loss['left']['consistensy']) / len(loss['left']['consistensy'])
        loss['right']['consistensy'] = [consistency_loss(l, -r) for l, r in zip(self._disp_maps['right'], self._disp_maps['left'])]
        loss['right']['consistensy'] = sum(loss['right']['consistensy']) / len(loss['right']['consistensy'])
        
        stereo_val, left_iwe, right_iwe, census_left, census_right, census_r_warp_l = stereo_loss(self._event_list,
                                                                                    self._flow_list,
                                                                                    self._disp_maps,
                                                                                    self.max_ts,
                                                                                    self.res,
                                                                                    1,
                                                                                    kernel_size=3)
        loss['left']['stereo'] = stereo_val['left']
        loss['right']['stereo'] = stereo_val['right']

        # for loc in ['left', 'right']:
        #     loss[loc]['tot'] = loss[loc]['temp'] + \
        #                         self.stereo_weight *loss[loc]['stereo'] + \
        #                         self.consistency_weight * loss[loc]['consistensy'] + \
        #                         self.smooth_weight * loss[loc]['smooth']

        loss['left']['tot'] = self.test_forward('left') + self.smooth_weight * loss['left']['smooth']
        loss['right']['tot'] = self.test_forward('right') + self.smooth_weight * loss['right']['smooth']

        return {
            'loss':loss['left']['tot'] + loss['right']['tot'],
            'left_iwe':left_iwe,
            'right_iwe':right_iwe,
            'census_left':census_left,
            'census_right':census_right,
            'census_r_warp_l':census_r_warp_l,
            'loss_left_terms': loss['left'],
        }

class BaseValidationMetric(torch.nn.Module):
    """
    Base class for validation metrics.
    """

    def __init__(self, resolution, device, overwrite_intermediate=False, flow_scaling=128, **kwargs):
        super(BaseValidationMetric, self).__init__()
        self.res = resolution
        self.flow_scaling = flow_scaling  # should be specified by the user
        self.overwrite_intermediate = overwrite_intermediate
        self.device = device

        self.reset()

    @property
    def num_events(self):
        if self._event_list is None:
            return 0
        else:
            return self._event_list.shape[1]

    def event_flow_association(self, flow_map, event_list, event_mask, dt_input, dt_gt, gtflow=None):
        """
        :param flow_list: [batch_size x N x 2] list of optical flow (x, y) maps
        :param event_list: [batch_size x N x 4] input events (ts, x, y, p)
        :param gtflow
        """

        # pos, weight = get_interpolation(event_list, flow_list, event_list[0, :, 0][-1], self.res, self.flow_scaling, round_idx=True)
        # pos = event_list[..., 1:3]
        # pos[..., 0] *= self.res[0]  # torch.view is row-major
        # pos = torch.sum(pos, dim=-1)
        # weight = torch.ones_like(pos[..., :1])
        # flow_map = torch.stack([
        #     interpolate(pos.long(), weight * flow_list[..., 0], self.res),
        #     interpolate(pos.long(), weight * flow_list[..., 1], self.res)
        # ], dim=1)
        # flow_map /= event_cnt
        # flow_map[event_cnt == 0] = 0

        flow_idx = event_list[:, :, 1:3].clone()
        flow_idx[:, :, 1] *= self.res[1]  # torch.view is row-major
        flow_idx = torch.sum(flow_idx, dim=2).long() # B x N x 2

        # get flow for every event in the list and update flow maps
   
        # B = flow_idx.shape[0]
        # batch_idx = torch.arange(B).view(B, 1).expand(-1, flow_idx.shape[1])
        flow = flow_map
        flow = flow.view(flow.shape[0], 2, -1).permute(0, 2, 1) # B x (HW) x 2
        # event_flow = flow[batch_idx, flow_idx, :]
        event_flow = torch.gather(flow, 1, flow_idx[..., None].repeat(1, 1, 2))

        if self._flow_list is None:
            self._flow_list = event_flow
        else:
            self._flow_list = torch.cat([self._flow_list, event_flow], dim=1)

        if self._flow_map is None:
            self._flow_map = flow_map[:, None]
        else:
            self._flow_map = torch.cat([self._flow_map, flow_map[:, None]], dim=1)

        # update internal event list
        if self._event_list is None:
            self._event_list = event_list
        else:
            event_list = event_list.clone()  # to prevent issues with other metrics
            event_list[:, :, 0:1] += self._passes  # only nonzero second time
            self._event_list = torch.cat([self._event_list, event_list], dim=1)

        if self._event_mask is None:
            self._event_mask = event_mask[:, None]
        else:
            self._event_mask = torch.cat([self._event_mask, event_mask[:, None]], dim=1)

        # update ground-truth optical flow
        self._gtflow = gtflow

        # update timestamps
        self._dt_input = dt_input
        self._dt_gt = dt_gt

        # update timestamp index
        self._passes += 1

    def overwrite_intermediate_flow(self, flow_map):
        """
        :param flow_list: [batch_size x N x 2] list of optical flow (x, y)
        :param event_list: [batch_size x N x 4] list of events
        """
        self._flow_map = flow_map
    
        self._event_mask = torch.sum(self._event_mask, dim=1, keepdim=True)
        self._event_mask[self._event_mask > 1] = 1

    def reset(self):
        self._passes = 0
        self._event_list = None
        self._flow_list = None
        self._flow_map = None
        self._event_mask = None

    @property
    def flow_map(self):
        return self._flow_map[:, -1]

    def compute_window_events(self):
        idx = self._event_list[..., 1] * self.res[1] + self._event_list[..., 2]
        weights = torch.ones(idx.shape).to(self.device)

        return torch.stack([
                interpolate(idx.long(), weights[..., None], self.res, self._event_list[..., -1] == 1), 
                interpolate(idx.long(), weights[..., None], self.res, self._event_list[..., -1] == -1)
                ], dim=1)

    def compute_masked_window_flow(self):
        if self.overwrite_intermediate:
            return self._flow_map[-1] * self._event_mask
        else:
            avg_flow = self._flow_map * self._event_mask[:, :, None]
            avg_flow = avg_flow.sum(dim=1)
            avg_flow /= torch.sum(self._event_mask, dim=1, keepdim=True) + 1e-9
            return avg_flow

    def compute_window_iwe(self, round_idx=True):
        max_ts = self._passes
        fw_idx, fw_weights = get_interpolation(
            self._event_list, self._flow_list, max_ts, self.res, self.flow_scaling, round_idx=round_idx
        )
        iwe = torch.stack([
            interpolate(fw_idx.long(), fw_weights, self.res, self._event_list[..., -1] == 1),
            interpolate(fw_idx.long(), fw_weights, self.res, self._event_list[..., -1] == -1)
        ], dim=1)

        return iwe

class FWL(BaseValidationMetric):
    """
    Flow Warp Loss (FWL), as described in Section 4.1 of the paper 'Reducing the Sim-to-Real Gap for Event
    Cameras', Stoffregen and Scheerlinck et al., ECCV'20.
    The FWL metric is the ratio of the contrast of the image of warped events and that of the image of
    (non-warped) events; hence, the larger the value of this metric, the better the optical flow estimate.
    Contrast is measured through the spatial variance of the image of events.
    Note that this metric is sensitive to the number of input events.
    """

    def __init__(self, resolution, device, overwrite_intermediate=False, flow_scaling=128, **kwargs):
        super().__init__(resolution, device, overwrite_intermediate=overwrite_intermediate, flow_scaling=flow_scaling, **kwargs)

    
    def event_flow_association(self, flow, event_list):
        """
        :param flow_list: [batch_size x N x 2] list of optical flow (x, y) map
        :param event_list: [batch_size x N x 4] input events (ts, x, y, p)
        """

        flow_idx = event_list[:, :, 1:3].clone()
        flow_idx[:, :, 1] *= self.res[1]  # torch.view is row-major
        flow_idx = torch.sum(flow_idx, dim=2) # B x N x 1

        # get flow for every event in the list and update flow maps

        flow = flow.view(flow.shape[0], 2, -1).permute(0, 2, 1) # B x (HW) x 2
        event_flow = torch.gather(flow, 1, flow_idx[..., None].repeat(1, 1, 2).long())

        if self._flow_list is None:
            self._flow_list = event_flow
        else:
            self._flow_list = torch.cat([self._flow_list, event_flow], dim=1)

        # update internal event list
        if self._event_list is None:
            self._event_list = event_list
        else:
            event_list = event_list.clone()  # to prevent issues with other metrics
            event_list[:, :, 0:1] += self._passes  # only nonzero second time
            self._event_list = torch.cat([self._event_list, event_list], dim=1)

        # update timestamp index
        self._passes += 1

    def overwrite_intermediate_flow(self, flow_list):
        """
        :param flow_list: [batch_size x N x 2] list of optical flow (x, y)
        :param event_list: [batch_size x N x 4] list of events
        """

        self._flow_list = flow_list

    def forward(self):
        max_ts = self._passes

        # image of (forward) warped events
        fw_idx, fw_weights = get_interpolation(
            self._event_list, self._flow_list, max_ts, self.res, self.flow_scaling, round_idx=False
        )
        fw_IWE = interpolate(fw_idx, fw_weights, self.res)

        # image of non-warped events
        zero_idx, zero_weights = get_interpolation(
            self._event_list, self._flow_list * 0, max_ts, self.res, self.flow_scaling, round_idx=False
        )
        IE = interpolate(zero_idx, zero_weights, self.res)

        # Forward Warping Loss (FWL)
        FWL = spatial_variance(fw_IWE) / spatial_variance(IE)
        FWL = FWL.view((fw_IWE.shape[0]))  # one metric per batch

        return FWL, fw_IWE, IE


class RSAT(BaseValidationMetric):
    """
    Similarly to the FWL metric, the Ratio of the Squared Averaged Timestamps (RSAT) metric is the ratio of the squared sum
    of the per-pixel and per-polarity average timestamp of the image of warped events and that of
    the image of (non-warped) events; hence, the lower the value of this metric, the better the optical flow estimate.
    Note that this metric is sensitive to the number of input events.
    """

    def __init__(self, resolution, device, overwrite_intermediate=False, flow_scaling=128, **kwargs):
        super().__init__(resolution, device, overwrite_intermediate=overwrite_intermediate, flow_scaling=flow_scaling, **kwargs)

    def forward(self):
        max_ts = self._passes

        # image of (forward) warped averaged timestamps
        ts_list = self._event_list[:, :, 0:1]
        fw_idx, fw_weights = get_interpolation(
            self._event_list, self._flow_list, max_ts, self.res, self.flow_scaling, round_idx=True
        )
        fw_iwe_pos = interpolate(fw_idx.long(), fw_weights, self.res, mask=self._event_list[..., -1] == 1)
        fw_iwe_neg = interpolate(fw_idx.long(), fw_weights, self.res, mask=self._event_list[..., -1] == -1)
        fw_iwe_pos_ts = interpolate(
            fw_idx.long(), fw_weights * ts_list, self.res, mask=self._event_list[..., -1] == 1
        )
        fw_iwe_neg_ts = interpolate(
            fw_idx.long(), fw_weights * ts_list, self.res, mask=self._event_list[..., -1] == -1
        )
        fw_iwe_pos_ts /= fw_iwe_pos + 1e-9
        fw_iwe_neg_ts /= fw_iwe_neg + 1e-9
        fw_iwe_pos_ts = fw_iwe_pos_ts / max_ts
        fw_iwe_neg_ts = fw_iwe_neg_ts / max_ts

        # image of non-warped averaged timestamps
        zero_idx, zero_weights = get_interpolation(
            self._event_list, self._flow_list * 0, max_ts, self.res, self.flow_scaling, round_idx=True
        )
        zero_iwe_pos = interpolate(
            zero_idx.long(), zero_weights, self.res, mask=self._event_list[..., -1] == 1
        )
        zero_iwe_neg = interpolate(
            zero_idx.long(), zero_weights, self.res, mask=self._event_list[..., -1] == -1
        )
        zero_iwe_pos_ts = interpolate(
            zero_idx.long(), zero_weights * ts_list, self.res, mask=self._event_list[..., -1] == 1
        )
        zero_iwe_neg_ts = interpolate(
            zero_idx.long(), zero_weights * ts_list, self.res, mask=self._event_list[..., -1] == -1
        )
        zero_iwe_pos_ts /= zero_iwe_pos + 1e-9
        zero_iwe_neg_ts /= zero_iwe_neg + 1e-9
        zero_iwe_pos_ts = zero_iwe_pos_ts / max_ts
        zero_iwe_neg_ts = zero_iwe_neg_ts / max_ts

        # (scaled) sum of the squares of the per-pixel and per-polarity average timestamps
        fw_iwe_pos_ts = fw_iwe_pos_ts.view(fw_iwe_pos_ts.shape[0], -1)
        fw_iwe_neg_ts = fw_iwe_neg_ts.view(fw_iwe_neg_ts.shape[0], -1)
        fw_iwe_pos_ts = torch.sum(fw_iwe_pos_ts ** 2, dim=1)
        fw_iwe_neg_ts = torch.sum(fw_iwe_neg_ts ** 2, dim=1)
        fw_ts_sum = fw_iwe_pos_ts + fw_iwe_neg_ts

        fw_nonzero_px = fw_iwe_pos + fw_iwe_neg
        fw_nonzero_px[fw_nonzero_px > 0] = 1
        fw_nonzero_px = fw_nonzero_px.view(fw_nonzero_px.shape[0], -1)
        fw_ts_sum /= torch.sum(fw_nonzero_px, dim=1)

        zero_iwe_pos_ts = zero_iwe_pos_ts.view(zero_iwe_pos_ts.shape[0], -1)
        zero_iwe_neg_ts = zero_iwe_neg_ts.view(zero_iwe_neg_ts.shape[0], -1)
        zero_iwe_pos_ts = torch.sum(zero_iwe_pos_ts ** 2, dim=1)
        zero_iwe_neg_ts = torch.sum(zero_iwe_neg_ts ** 2, dim=1)
        zero_ts_sum = zero_iwe_pos_ts + zero_iwe_neg_ts

        zero_nonzero_px = zero_iwe_pos + zero_iwe_neg
        zero_nonzero_px[zero_nonzero_px > 0] = 1
        zero_nonzero_px = zero_nonzero_px.view(zero_nonzero_px.shape[0], -1)
        zero_ts_sum /= torch.sum(zero_nonzero_px, dim=1)

        return fw_ts_sum / zero_ts_sum


class AEE(BaseValidationMetric):
    """
    Average endpoint error (which is just the Euclidean distance) loss.
    """

    def __init__(self, resolution, device, overwrite_intermediate=False, flow_scaling=128, **kwargs):
        super().__init__(resolution, device, overwrite_intermediate, flow_scaling, **kwargs)

    @property
    def num_events(self):
        return float("inf")

    def forward(self):

        # convert flow
        flow = self._flow_map.sum(1) * self.flow_scaling
        # flow *= self._dt_gt.to(self.device) / self._dt_input.to(self.device)
        flow_mag = flow.pow(2).sum(1).sqrt()

        # compute AEE
        error = (flow - self._gtflow).pow(2).sum(1).sqrt()

        # AEE not computed in pixels without events
        event_mask = self._event_mask[:, -1, :, :].bool()

        # AEE not computed in pixels without valid ground truth
        # gtflow_mask_x = self._gtflow[:, 0, :, :] == 0.0
        # gtflow_mask_y = self._gtflow[:, 1, :, :] == 0.0
        # gtflow_mask = gtflow_mask_x * gtflow_mask_y
        # gtflow_mask = ~gtflow_mask
        
        gtflow_mask = (self._gtflow[:, 0] != 0) | (self._gtflow[:, 1] != 0)

        # mask AEE and flow
        mask = event_mask * gtflow_mask
        mask = mask.reshape(self._flow_map[:, -1].shape[0], -1)
        error = error.view(self._flow_map[:, -1].shape[0], -1)
        flow_mag = flow_mag.view(self._flow_map[:, -1].shape[0], -1)
        error = error * mask
        flow_mag = flow_mag * mask

        # compute AEE and percentage of outliers
        num_valid_px = torch.sum(mask, dim=1)
        AEE = torch.sum(error, dim=1) / (num_valid_px + 1e-9)

        outliers = (error > 3.0) * (error > 0.05 * flow_mag)  # AEE larger than 3px and 5% of the flow magnitude
        percent_AEE = outliers.sum() / (num_valid_px + 1e-9)

        return {"metric": AEE, 
                "percent": percent_AEE,}


class AveragedIWE(nn.Module):
    """
    Returns an image of the per-pixel and per-polarity average number of warped events given
    an optical flow map.
    """

    def __init__(self, config, device):
        super(AveragedIWE, self).__init__()
        self.res = config["loader"]["resolution"]
        self.flow_scaling = max(config["loader"]["resolution"])
        self.batch_size = config["loader"]["batch_size"]
        self.device = device

    def forward(self, flow, event_list, pol_mask):
        """
        :param flow: [batch_size x 2 x H x W] optical flow maps
        :param event_list: [batch_size x N x 4] input events (y, x, ts, p)
        :param pol_mask: [batch_size x N x 2] per-polarity binary mask of the input events
        """

        # original location of events
        idx = event_list[:, :, 1:3].clone()
        idx[:, :, 1] *= self.res[1]  # torch.view is row-major
        idx = torch.sum(idx, dim=2, keepdim=True)

        # flow vector per input event
        flow_idx = event_list[:, :, 1:3].clone()
        flow_idx[:, :, 1] *= self.res[1]  # torch.view is row-major
        flow_idx = torch.sum(flow_idx, dim=2)

        # get flow for every event in the list
        flow = flow.view(flow.shape[0], 2, -1)
        event_flowy = torch.gather(flow[:, 1, :], 1, flow_idx.long())  # vertical component
        event_flowx = torch.gather(flow[:, 0, :], 1, flow_idx.long())  # horizontal component
        event_flowy = event_flowy.view(event_flowy.shape[0], event_flowy.shape[1], 1)
        event_flowx = event_flowx.view(event_flowx.shape[0], event_flowx.shape[1], 1)
        event_flow = torch.cat([event_flowy, event_flowx], dim=2)

        # interpolate forward
        fw_idx, fw_weights = get_interpolation(event_list, event_flow, 1, self.res, self.flow_scaling, round_idx=True)

        # per-polarity image of (forward) warped events
        fw_iwe_pos = interpolate(fw_idx.long(), fw_weights, self.res, polarity_mask=pol_mask[:, :, 0:1])
        fw_iwe_neg = interpolate(fw_idx.long(), fw_weights, self.res, polarity_mask=pol_mask[:, :, 1:2])
        if fw_idx.shape[1] == 0:
            return torch.cat([fw_iwe_pos, fw_iwe_neg], dim=1)

        # make sure unfeasible mappings are not considered
        pol_list = event_list[:, :, 3:4].clone()
        pol_list[pol_list < 1] = 0  # negative polarity set to 0
        pol_list[fw_weights == 0] = 2  # fake polarity to detect unfeasible mappings

        # encode unique ID for pixel location mapping (idx <-> fw_idx = m_idx)
        m_idx = torch.cat([idx.long(), fw_idx.long()], dim=2)
        m_idx[:, :, 0] *= self.res[0] * self.res[1]
        m_idx = torch.sum(m_idx, dim=2, keepdim=True)

        # encode unique ID for per-polarity pixel location mapping (pol_list <-> m_idx = pm_idx)
        pm_idx = torch.cat([pol_list.long(), m_idx.long()], dim=2)
        pm_idx[:, :, 0] *= (self.res[0] * self.res[1]) ** 2
        pm_idx = torch.sum(pm_idx, dim=2, keepdim=True)

        # number of different pixels locations from where pixels originate during warping
        # this needs to be done per batch as the number of unique indices differs
        fw_iwe_pos_contrib = torch.zeros((flow.shape[0], self.res[0] * self.res[1], 1)).to(self.device)
        fw_iwe_neg_contrib = torch.zeros((flow.shape[0], self.res[0] * self.res[1], 1)).to(self.device)
        for b in range(0, self.batch_size):

            # per-polarity unique mapping combinations
            unique_pm_idx = torch.unique(pm_idx[b, :, :], dim=0)
            unique_pm_idx = torch.cat(
                [
                    torch.div(unique_pm_idx, (self.res[0] * self.res[1]) ** 2, rounding_mode='trunc'),
                    unique_pm_idx % ((self.res[0] * self.res[1]) ** 2),
                ],
                dim=1,
            )  # (pol_idx, mapping_idx)
            unique_pm_idx = torch.cat(
                [unique_pm_idx[:, 0:1], unique_pm_idx[:, 1:2] % (self.res[0] * self.res[1])], dim=1
            )  # (pol_idx, fw_idx)
            unique_pm_idx[:, 0] *= self.res[0] * self.res[1]
            unique_pm_idx = torch.sum(unique_pm_idx, dim=1, keepdim=True)

            # per-polarity unique receiving pixels
            unique_pfw_idx, contrib_pfw = torch.unique(unique_pm_idx[:, 0], dim=0, return_counts=True)
            unique_pfw_idx = unique_pfw_idx.view((unique_pfw_idx.shape[0], 1))
            contrib_pfw = contrib_pfw.view((contrib_pfw.shape[0], 1))
            unique_pfw_idx = torch.cat(
                [torch.div(unique_pfw_idx, self.res[0] * self.res[1], rounding_mode='trunc'), unique_pfw_idx % (self.res[0] * self.res[1])],
                dim=1,
            )  # (polarity mask, fw_idx)

            # positive scatter pixel contribution
            mask_pos = unique_pfw_idx[:, 0:1].clone()
            mask_pos[mask_pos == 2] = 0  # remove unfeasible mappings
            b_fw_iwe_pos_contrib = torch.zeros((self.res[0] * self.res[1], 1)).to(self.device)
            b_fw_iwe_pos_contrib = b_fw_iwe_pos_contrib.scatter_add_(
                0, unique_pfw_idx[:, 1:2], mask_pos.float() * contrib_pfw.float()
            )

            # negative scatter pixel contribution
            mask_neg = unique_pfw_idx[:, 0:1].clone()
            mask_neg[mask_neg == 2] = 1  # remove unfeasible mappings
            mask_neg = 1 - mask_neg  # invert polarities
            b_fw_iwe_neg_contrib = torch.zeros((self.res[0] * self.res[1], 1)).to(self.device)
            b_fw_iwe_neg_contrib = b_fw_iwe_neg_contrib.scatter_add_(
                0, unique_pfw_idx[:, 1:2], mask_neg.float() * contrib_pfw.float()
            )

            # store info
            fw_iwe_pos_contrib[b, :, :] = b_fw_iwe_pos_contrib
            fw_iwe_neg_contrib[b, :, :] = b_fw_iwe_neg_contrib

        # average number of warped events per pixel
        fw_iwe_pos_contrib = fw_iwe_pos_contrib.view((flow.shape[0], 1, self.res[0], self.res[1]))
        fw_iwe_neg_contrib = fw_iwe_neg_contrib.view((flow.shape[0], 1, self.res[0], self.res[1]))
        fw_iwe_pos[fw_iwe_pos_contrib > 0] /= fw_iwe_pos_contrib[fw_iwe_pos_contrib > 0]
        fw_iwe_neg[fw_iwe_neg_contrib > 0] /= fw_iwe_neg_contrib[fw_iwe_neg_contrib > 0]

        return torch.cat([fw_iwe_pos, fw_iwe_neg], dim=1)

class Depth_Benchmark(nn.Module):
    def __init__(self, 
                resolution=[256, 256],
                baseline=10,
                focal=199,
                flow_scaling=None,
                mask_output=True,
                overwrite_intermediate=False,
                device=torch.device("cuda")):
        
        super(Depth_Benchmark, self).__init__()
        self.res = resolution
        self.baseline = baseline
        self.focal = focal
        self.flow_scaling = flow_scaling  # should be specified by the user
        self.mask_output = mask_output
        self.overwrite_intermediate = overwrite_intermediate
        self.device = device
        self.reset()

    def reset(self):
        self._event_cnt = []
        self._depth_maps = []
        self._gt_depth_maps = []
    
    def event_association(self, disp, event_cnt, gt_depth):
        """
        :param depth: [batch_size x H x W x 1] list of optical flow (x, y) map
        :param event_cnt: [batch_size x H x W x 2] input events
        """
        depth = self.focal / (disp * self.flow_scaling + 1e-9) * self.baseline

        self._depth_maps.append(depth)
        self._event_cnt.append(event_cnt)
        self._gt_depth_maps.append(gt_depth[:, None])

    def overwrite_intermediate(self, disp, event_cnt):
        depth = self.focal / (disp * self.flow_scaling + 1e-9) * self.baseline
        self._event_cnt = []
        self._depth_maps = []

        self._depth_maps.append(depth)
        self._event_cnt.append(event_cnt)
    
    def forward(self):
        metrics = {}

        depth_maps = torch.cat(self._depth_maps, dim=0)
        event_cnt = torch.cat(self._event_cnt, dim=0)
        gt_depth_maps = torch.cat(self._gt_depth_maps, dim=0)

        # # mask depth
        mask = torch.isfinite(gt_depth_maps) & (gt_depth_maps > 0)
        if self.mask_output:
            mask = mask & (event_cnt.sum(dim=1, keepdim=True) > 0)
            # depth_maps = depth_maps * event_cnt
            # gt_depth_maps = gt_depth_maps * event_cnt

        # compute Abs Relative Error
        error = torch.abs((depth_maps - gt_depth_maps)) / (gt_depth_maps)
        error[~mask] = 0
        error = error.sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3))
        metrics["Abs_Relative_Error"] = error.mean()

        # compute Logarithmic RMSE
        error = torch.log(depth_maps) - torch.log(gt_depth_maps)
        error = error ** 2
        error[~mask] = 0
        error = error.sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3))
        error = error.sqrt()
        metrics["Log_RMSE"] = error.mean()

        # compute SILog
        error = torch.log(depth_maps) - torch.log(gt_depth_maps)
        error[~mask] = 0
        error = (error ** 2).sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3)) - \
                error.sum(dim=(1, 2, 3)) ** 2 / mask.sum(dim=(1, 2, 3)) ** 2
        metrics['SILog'] = error.mean()

        # compute Delta1
        error = torch.max(depth_maps / gt_depth_maps, gt_depth_maps / depth_maps)
        error[~mask] = torch.inf
        metrics['Delta_1'] = (error < 1.25).sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3))
        metrics['Delta_1'] = metrics['Delta_1'].mean()

        metrics['Delta_2'] = (error < (1.25 ** 2)).sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3))
        metrics['Delta_2'] = metrics['Delta_2'].mean()

        metrics['Delta_3'] = (error < (1.25 ** 3)).sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3))
        metrics['Delta_3'] = metrics['Delta_3'].mean()

        return metrics


class Flow_Benchmark:
    def __init__(self, infer_loss_fname, config, device):
        self.config = config
        self.infer_loss_fname = infer_loss_fname
        self.fn = {}
        self.cur = {}
        self.tot = {}
        for metric in config['metrics']["method"]:
            self.fn[metric] = eval(metric)(config, device, flow_scaling=config["metrics"]["flow_scaling"])
            self.cur[metric] = {}
            self.tot[metric] = {}
        self.reset(self.cur)
        self.reset(self.tot)
        self.idx_AEE = 0

    def reset(self, seq):
        for metric in self.cur.keys():
            seq[metric]["value"] = 0
            seq[metric]["it"] = 0
            if metric == "AEE":
                seq[metric]["outliers"] = 0

    def update(self):
        for metric in self.cur.keys():
            self.tot[metric]["value"] += self.cur[metric]["value"]
            self.tot[metric]["it"] += self.cur[metric]["it"]
            if metric == "AEE":
                self.tot[metric]["outliers"] += self.cur[metric]["outliers"]

    def __call__(self, pred_flow, inputs):       
        for metric in self.fn.keys():
            self.fn[metric].event_flow_association(pred_flow, inputs)

            if self.fn[metric].num_events >= self.config["data"]["window_eval"]:
                # overwrite intermedia flow estimates with the final ones
                if self.config["loss"]["overwrite_intermediate"]:
                    self.fn[metric].overwrite_intermediate_flow(pred_flow)
                if metric == "AEE":
                    if inputs["dt_gt"] <= 0.0:
                        continue
                    self.idx_AEE += 1
                    if self.idx_AEE != np.round(1.0 / self.config["data"]["window"]):
                        continue

                # compute metric
                val_metric = self.fn[metric]()
                if metric == "AEE":
                    self.idx_AEE = 0

                # accumulate results
                for batch in range(self.config["loader"]["batch_size"]):
                    self.cur[metric]["it"] += 1
                    if metric == "AEE":
                        self.cur[metric]["value"] += val_metric[0][batch].cpu().numpy()
                        self.cur[metric]["outliers"] += val_metric[1][batch].cpu().numpy()
                    else:
                        self.cur[metric]["value"] += val_metric[batch].cpu().numpy()

    def write(self, seq_name=None):
            # store validation config and results
        results = {}
        self.update()
        with open(self.infer_loss_fname, 'a') as f:
            # f.write('perceptual loss for each step:{}\n'.format(self.loss['perceptual_loss']))
            # f.write('mse loss for each step:{}\n'.format(self.loss['mse_loss']))
            # f.write('ssim loss for each step:{}\n'.format(self.loss['ssim_loss']))
            # f.write('******************************\n')
            seq = self.cur if seq_name else self.tot
            seq_name = "whole" if not seq_name else seq_name
            for metric in seq.keys():
                results[metric] = seq[metric]["value"] / seq[metric]["it"]
                f.write(f"mean {metric} for {seq_name} sequences:{results[metric]:.3f}\n")
                if metric == "AEE":
                    results["AEE_outliers"] = seq[metric]["outliers"] / seq[metric]["it"]
                    f.write(f"mean AEE_outliers for {seq_name} sequences:{results['AEE_outliers']:.3f}\n")
        self.reset(self.cur)
        return results

if __name__ == '__main__':
    parent_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    sys.path.append(parent_dir_name)
    from Tools.stream_loader import H5Dataset, StreamDataLoader
    config = {
        "resolution": [128, 128],
        "loss_window": 10000,
        "flow_regul_weight": 0.001,
        "clip_grad": 100,
        "overwrite_intermediate": False
    }
    loss1 = EventWarping(**config)

    data_config = {
        "path": "/home/wan97/Workspace/DVS/Optical_flow/TimeFlow/Datasets/UZHFPV/Optical_Flow",
        "mode": "events",
        "__mode": "events/time/frames",
        "window": 1000,
        "num_bins": 2,
        "resolution": [128, 128],
        "batch_size": 1,
        "encoding": "cnt",
        # "augmentation": ["Horizontal", "Vertical", "Polarity"],
        # "augment_prob": [0.5, 0.5, 0.5],
        "debug": False
    }
    dataset = H5Dataset(**data_config)
    loader = StreamDataLoader(dataset, num_workers=1)

    torch.manual_seed(2022)
    flow = [torch.rand((1, 2, 128, 128))]

    for item in loader:
        loss1.event_flow_association(flow, item['event_list'], item['event_cnt'])
        loss1_batch = loss1()
        print(loss1_batch)
        break
    exit()
