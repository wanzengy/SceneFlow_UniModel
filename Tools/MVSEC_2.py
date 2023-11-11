"""
The StreamLoader is a class built on top of DataLoader,
that fuses batches together so batches are always temporally
coherent.

Here we use a different strategy than the one described in
https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd

We just return the torch worker's id every batch, and create a fifo per worker on the main
thread side.
"""

import os
import h5py
import time
import torch
import random
import zipfile
import yaml

import numpy as np
import pytorch3d.transforms as t3d

import data_func

from torch.utils.data import DataLoader, Dataset, ConcatDataset



def split_batch_size(batch_size, num_workers):
    """Returns a list of batch_size

    Args:
        batch_size (int): total batch size
        num_workers (int): number of workers
    """
    num_workers = min(num_workers, batch_size)
    split_size = batch_size // num_workers
    total_size = 0
    split_sizes = [split_size] * (num_workers - 1)
    split_sizes += [batch_size - sum(split_sizes)]
    return split_sizes

class Map:
    """
    Utility class for reading the APS frames/ Optical flow maps encoded in the HDF5 files.
    """

    def __init__(self):
        self.ts = []
        self.names = []

    def __call__(self, name, h5obj):
        if hasattr(h5obj, "dtype") and name not in self.names:
            self.names += [name]

class H5Stream(Dataset):
    def __init__(self,
                file_name,
                stereo_idx,
                window, 
                resolution,
                num_bins,
                augmentation,
                augment_prob,
                get_gt,
                **kwargs):
        self.file = h5py.File(file_name, "r")
        self.fname = file_name
        self.sequence_name = file_name.split("/")[-1].split(".")[0]
        self.file_predict = None
        self.stereo_idx = stereo_idx

        self.left_events = self.file["left/events"]
        self.right_events = self.file["right/events"]

        self.window = window
        self.resolution = resolution
        self.num_bins = num_bins

        self.augmentation = []
        for i, mechanism in enumerate(augmentation):
            if np.random.random() < augment_prob[i]:
                self.augmentation.append(mechanism)

        self.get_gt = get_gt
        if get_gt:
            self.gt_file = h5py.File(file_name.replace("data", "gt"), "r")

    def hot_filter(self, batch, event_voxel, event_cnt, event_mask):
        hot_mask = self.create_hot_mask(event_cnt, batch)
        hot_mask_voxel = torch.stack([hot_mask] * self.num_bins, axis=2).permute(2, 0, 1)
        hot_mask_cnt = torch.stack([hot_mask] * 2, axis=2).permute(2, 0, 1)
        event_voxel = event_voxel * hot_mask_voxel
        event_cnt = event_cnt * hot_mask_cnt
        event_mask *= hot_mask.view((1, hot_mask.shape[0], hot_mask.shape[1]))
        return event_voxel, event_cnt, event_mask

    def load_events(self, events, cur_idx):
        """
        Get all the events in between two indices.
        :param file: file to read from
        :param idx0: start index
        :param idx1: end index
        :return xs: [N] numpy array with event x location
        :return ys: [N] numpy array with event y location
        :return ts: [N] numpy array with event timestamp
        :return ps: [N] numpy array with event polarity ([-1, 1])
        """
        idx0, idx1 = cur_idx[0], cur_idx[1]
        ys = events[idx0:idx1]['y'] - 2
        xs = events[idx0:idx1]['x'] - 45
        ts = events[idx0:idx1]['t']
        ps = events[idx0:idx1]['p']

        # handle case with very few events
        if xs.shape[0] <= 10:
            xs, ys, ts, ps = np.split(np.empty([40, 0]), 4)

        # event formatting and timestamp normalization
        dt_input = np.asarray(0.0)
        if ts.shape[0] > 0:
            dt_input = np.asarray(ts[-1] - ts[0], dtype=np.float32)
        
        last_ts = np.asarray(ts[-1])
        xs, ys, ts, ps = data_func.event_formatting(xs, ys, ts, ps)

        # data augmentation
        xs, ys, ps = data_func.augment_events(xs, ys, ps, self.augmentation, self.resolution)

        event_list = torch.stack([ts, xs, ys, ps], dim=-1) # t x y p
        event_cnt = data_func.ev_to_channels(xs, ys, ps, self.resolution)
        event_vox = data_func.ev_to_voxel(xs, ys, ts, ps, self.num_bins, self.resolution)
        event_mask = (event_cnt[0] + event_cnt[1]) > 0
        event_mask = event_mask.float()

        return {
            "event_list":   event_list,
            "event_cnt":    event_cnt,
            "event_vox":    event_vox,
            "event_mask":   event_mask,
            "idx":  cur_idx,
            "ts":   torch.from_numpy(last_ts),
            "dt_input": torch.from_numpy(dt_input),
        }

    def __len__(self):
        return len(self.stereo_idx[0])

    def __getitem__(self, idx):
        output = {'left':self.load_events(self.left_events, self.stereo_idx[0][idx]), 
                'right':self.load_events(self.right_events, self.stereo_idx[1][idx])}

        # # hot pixel removal
        # if self.config["hot_filter"]["enabled"]:
        #     event_voxel, event_cnt, event_mask = self.hot_filter(batch, event_voxel, event_cnt, event_mask)

        if self.get_gt:
            output['gt'] = {}
            output['gt']['depth'] = torch.from_numpy(self.gt_file[f"depth/D_{self.stereo_idx[2][idx]}"][:])
            output['gt']['lin_vel'] = torch.from_numpy(self.gt_file[f'lin_vel'][self.stereo_idx[2][idx]][:])
            output['gt']['ang_vel'] = torch.from_numpy(self.gt_file[f'ang_vel'][self.stereo_idx[2][idx]][:])
            output['gt']['of'] = torch.from_numpy(self.gt_file[f"optical_flow/F_{self.stereo_idx[2][idx]}"][:])

        output['name'] = self.sequence_name
        output["file_name"] = self.fname
        return output

class H5Dataloader(DataLoader):
    def __init__(self, 
                path, window, 
                resolution=[255, 255],
                debug=False, 
                num_bins=2, 
                batch_size=1,
                num_workers=4,
                augmentation=[],
                augment_prob=[],
                shuffle=False,
                get_gt=False,
                **kwargs):
        # input event sequences
        self.files = []
        self.events = {}
        self.idx = {}
        self.maps = {}
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith("1_data.h5"):
                    fname = os.path.join(root, file)
                    vari = 1
                    if 'VariNum' in augmentation:
                        vari = (1 - augment_prob[augmentation.index('VariNum')] * np.random.random())
                    # if 'shapes_6dof.h5' in fname:
                    # if 'reflective_materials.h5' in fname:
                    # if 'outdoor_day1_data.h5' in fname:
                    self.get_stereo_event_fixed_num(fname, window, batch_size, get_gt)
                    if debug and len(self.files) == batch_size:
                        break

                if file.endswith("_calib.zip"):
                    zip_fn = os.path.join(root, file)
                    scene = file.split(".")[0][:-6]
                    with zipfile.ZipFile(zip_fn) as calib_zip:
                        yaml_name = "camchain-imucam-" + scene +".yaml"
                        with calib_zip.open(yaml_name) as yaml_file:
                            intrinsic_extrinsic = yaml.safe_load(yaml_file)
                            self.left_intrinsics = intrinsic_extrinsic['cam0']['projection_matrix']
                            self.right_intrinsics = intrinsic_extrinsic['cam1']['projection_matrix']
        
        datasets = []
        for fn in self.files:
            idx = self.idx[fn]
            file_name = fn.split('@')[0]
            datasets.append(H5Stream(file_name, idx, window, resolution, num_bins, augmentation, augment_prob, get_gt))
        datasets = ConcatDataset(datasets)

        super().__init__(dataset=datasets, 
                        batch_size=batch_size, 
                        shuffle=shuffle, 
                        num_workers=num_workers, 
                        collate_fn=data_func.stereo_collate,
                        pin_memory=True)

    def get_stereo_event_fixed_num(self, fname, num, batch_size, get_gt=False):
        file = h5py.File(fname, "r")
     
        left_events = file["/left/events"]
        right_events = file["right/events"]

        last_left_idx = len(left_events) - 1
        last_right_idx = len(right_events) - 1

        left_idx = []
        right_idx = []
        
        cur_left_idx = 0
        cur_right_idx = 0

        idx_fname = fname[:-7] + str(num)
        if os.path.exists(idx_fname + '_left.txt'):
            left_idx = np.loadtxt(idx_fname + '_left.txt', dtype=np.uint32)
            right_idx = np.loadtxt(idx_fname + '_right.txt', dtype=np.uint32)
        else:
            # synchronous left events and right events
            if abs(left_events[cur_left_idx]['t'] - right_events[cur_right_idx]['t']) > 1e-4:
                if left_events[cur_left_idx]['t'] > right_events[cur_right_idx]['t']:
                    cur_ms = np.floor((left_events[cur_left_idx]['t'] - right_events[0]['t']) * 1000)
                    cur_right_idx = file["right/event_ms_to_idx"][int(cur_ms)]
                    while (abs(left_events[cur_left_idx]['t'] - right_events[cur_right_idx]['t']) > 1e-4) and (cur_right_idx < last_right_idx):
                        cur_right_idx += 1
                else:
                    cur_ms = np.floor((right_events[cur_right_idx]['t'] - left_events[0]['t']) * 1000)
                    cur_left_idx = file['left/event_ms_to_idx'][int(cur_ms)]
                    while (abs(left_events[cur_left_idx]['t'] - right_events[cur_right_idx]['t']) > 1e-4) and (cur_left_idx < last_left_idx):
                        cur_left_idx += 1
            
            syn_time = left_events[cur_left_idx]['t']

            while ((cur_left_idx + num) < last_left_idx) and ((cur_right_idx + num) < last_right_idx):
                left_idx.append((cur_left_idx, cur_left_idx + num))
                right_idx.append((cur_right_idx, cur_right_idx + num))

                cur_left_idx = cur_left_idx + num
                cur_right_idx = cur_right_idx + num

                syn_time = left_events[cur_left_idx]['t']
                if abs(right_events[cur_right_idx]['t'] - syn_time) > 1e-4:
                    cur_ms = np.floor((syn_time - right_events[0]['t']) * 1000)
                    cur_right_idx = file["right/event_ms_to_idx"][int(cur_ms)]
                    while (right_events[cur_right_idx]['t'] < syn_time) and (cur_right_idx < last_right_idx):
                        cur_right_idx += 1
                
                # if len(left_idx) > 100:
                #     break

            np.savetxt(idx_fname + '_left.txt', np.array(left_idx, dtype=np.uint32))
            np.savetxt(idx_fname + '_right.txt', np.array(right_idx, dtype=np.uint32))

        gt_idx = []
        if get_gt:
            if os.path.exists(idx_fname + '_gt.txt'):
                gt_idx = np.loadtxt(idx_fname + '_gt.txt', dtype=np.uint32)
            else:
                gt_fname = fname.replace('data', 'gt')
                gt_file = h5py.File(gt_fname, "r")
                cur_idx = 0
                ed_idx = len(gt_file['timestamps'])
                for ev_idx in left_idx:
                    ev_t = left_events[ev_idx[0]]['t']
                    ros_ev_t = ev_t + file['start_time']
                    while (cur_idx < ed_idx) and (gt_file['timestamps'][cur_idx] < ros_ev_t):
                        cur_idx += 1
                    gt_idx.append(cur_idx)

                np.savetxt(idx_fname + '_gt.txt', np.array(gt_idx, dtype=np.uint32))

        for i in range(batch_size):
            fname = fname + '@' + str(i)

            if get_gt:
                gt_idx_batch = gt_idx[i::batch_size]
            else:
                gt_idx_batch = gt_idx
            
            self.idx[fname] = (left_idx[i::batch_size], right_idx[i::batch_size], gt_idx_batch)
            # self.idx[fname] = (left_idx[i::batch_size], right_idx[i::batch_size], gt_idx_batch)
            self.files.append(fname)

if __name__ == '__main__':
    # data = [[10, 11, 12, 13],
    #         [20, 21, 22],
    #         [42, 43],
    #         [90],
    #         # [100]
    #         ]
    # def temp(x):
    #     for t in x:
    #         yield t
    # # dataset = StreamDataset(data, temp, batch_size=4)
    # # dataset = H5Loader()
    # # dataset=None
    # loader = StreamDataLoader(data, temp, 4, collate_fn=lambda x: x)
    # for i in loader:
    #     print(list(i))
    # file = h5py.File('/home/wan97/Workspace/Dataset/DVS/ssl_E2VID/UZHFPV/Optical_Flow/indoor_forward_3_davis_with_gt_0.h5', 'r')
    # print(file['events'].keys())

    cfg = {
        "path": "Datasets/UZHFPV/Optical_Flow",
        "mode": "events",
        "__mode": "events/time/frames",
        "window": 1000,
        "seq_len": 10,
        "num_bins": 1,
        "resolution": [128, 128],
        "batch_size": 16,
        "encoding": "timesurface",
        "__encoding": "cnt/timesurface/mixture",
        "augmentation": ["Horizontal", "Vertical", "Polarity"],
        "augment_prob": [0.5, 0.5, 0.5],
        "debug": False
    }

    loader = H5Dataloader(**cfg)

    for i in loader:
        print(loader.dataset.pos.value, '/', len(loader.dataset),
            int(100 * loader.dataset.pos.value / len(loader.dataset)), '%', end='\r')