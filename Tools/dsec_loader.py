"""
The StreamLoader is a class built on top of DataLoader,
that fuses batches together so batches are always temporally
coherent.

Here we use a different strategy than the one described in
https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd

We just return the torch worker's id every batch, and create a fifo per worker on the main
thread side.
"""
from collections import deque
import os
import h5py
import time
import torch
import random
import multiprocessing

import numpy as np

import dsec_encoder as encoder
import data_func

from itertools import chain
from torch.utils.data import IterableDataset, DataLoader



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

class StreamDataset(IterableDataset):
    """Stream Dataset
    An Iterable Dataset zipping a group of iterables streams together.

    Args:
        stream_list (list): list of streams (path/ metadata)
        streamer (object): an iterator (user defined)
        batch_size (int): total batch size
        padding_mode (str): "zeros" "data" or "none", see "get_zip" function
        padding_value (object): padding value
    """
    def __init__(self,
                stream_list,
                streamer,
                batch_size,
                padding_mode,
                padding_value,
                pos,
                num_actives,
                mutex,
                ):

        self.stream_list = stream_list
        self.batch_size = batch_size
        self.streamer = streamer
        self.padding_mode = padding_mode
        self.padding_value = padding_value
        assert padding_mode in ['zero', 'data']
        self.pos = pos
        self.mutex = mutex
        self.num_actives = num_actives
        self.cnt = 0
        self._set_seed()

    def shuffle(self):
        random.shuffle(self.stream_list)

    def _set_seed(self):
        """ so that data is different along threads and epochs"""
        worker = torch.utils.data.get_worker_info()
        worker_id = int(worker.id) if worker is not None else 0
        seed = 2023 + worker_id + self.cnt
        np.random.seed(seed)
        random.seed(seed)
        self.cnt = (self.cnt + 4) % 20000

    def init_position(self):
        self.mutex.acquire()
        self.pos.value = 0
        self.num_actives.value = 0
        self.mutex.release()
    
    def increment_pos(self):
        self.mutex.acquire()
        pos = self.pos.value
        stream = self.stream_list[pos%len(self.stream_list)]
        self.pos.value = pos + 1
        self.mutex.release()
        return stream

    def get_value(self, iterators, i, actives):
        done = False
        while not done:
            try:
                if actives[i] or self.padding_mode == 'data':
                    value = next(iterators[i])
                    assert value is not None
                elif self.padding_mode == 'zero':
                    value = self.padding_value
                done = True
            except StopIteration:
                self.mutex.acquire()
                if actives[i] and (self.pos.value >= len(self.stream_list)):
                    self.num_actives.value -= 1
                actives[i] = 1 * (self.pos.value < len(self.stream_list))
                self.mutex.release()
                stream = self.increment_pos()
                if actives[i] or self.padding_mode == 'data':
                    assert stream is not None, self.pos.value
                    iterators[i] = iter(self.streamer(stream))
                elif self.padding_mode == 'zero':
                    value = self.padding_value
        return value

    def __len__(self):
        return len(self.stream_list)
    
    def __iter__(self):
        """Iterates over stream files

        Note: Here we use a mutex (WIP, pytest not working!)

        Note: Here the scheduling of iterable is done at the beginning.
        Instead User can change this code to map lazily iterables.
        """
        assert self.mutex, "Not initialize parallize"

        #initialization this should be done in worker_init_fnx
        worker = torch.utils.data.get_worker_info()
        worker_id = int(worker.id) if worker is not None else 0

        num_workers = 1 if worker is None else worker.num_workers
        split_size = split_batch_size(self.batch_size, num_workers)[worker_id]

        if len(self) < split_size:
            print('worker#', worker_id, ': Stopping... Number of streams < split_size')
            raise StopIteration

        """
        Just-in-time mapping
        The scheduling is done as we iterate.

        EDIT 9/7/2021: The position in the stream is shared accross workers
        This allows us to avoid the non ideal pre-iteration splitting of the dataset
        """

        iterators = []
        for i in range(split_size):
            stream = self.increment_pos()
            stream = iter(self.streamer(stream))
            iterators.append(stream)

        actives = [1 for _ in range(len(iterators))]
        _num_actives = sum(actives)
        self.mutex.acquire()
        self.num_actives.value += _num_actives
        self.mutex.release()

        while True:
            values = []
            for i in range(len(iterators)):
                values.append(self.get_value(iterators, i, actives))
            if self.num_actives.value:
                yield tuple(values), worker_id
            else:
                yield tuple([None]), worker_id

class StreamDataLoader(object):
    """StreamDataLoader

    Wraps around the DataLoader to handle the asynchronous batches.
    We now handle one single list of streams read from multiple workers with a mutex.

    Args:
        iterator_fun (lambda): function to create one stream
        batch_size (int): number of streams read at the same time
        num_workers (int): number of workers
        collate_fn (function): function to collate batch parts
        padding_mode (str): "data" or "zeros", what to do when all streams have been read but you still but one thread of streaming needs to output something
        padded_value (object): object or None
    """
    def __init__(self, 
                files,
                iterator_fun,
                batch_size=1,
                padding_mode='data',
                padding_value=None,
                shuffle=True,
                num_workers=1,
                collate_fn=data_func.custom_collate, 
                ):
        pos = multiprocessing.Value('i', 0)
        num_actives = multiprocessing.Value('i', 0)
        mutex = multiprocessing.Lock()
        dataset = StreamDataset(files, iterator_fun, batch_size, padding_mode, padding_value, pos, num_actives, mutex)

        self.dataset = dataset
        num_workers = min(dataset.batch_size, num_workers)
        assert isinstance(dataset, StreamDataset)
        self.dataloader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=num_workers,
            collate_fn=lambda x: x,
            pin_memory=False,
            drop_last=False)
        self.collate_fn = collate_fn
        self.num_workers = max(1, num_workers)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.dataloader.dataset.shuffle()
        self.dataloader.dataset.init_position()

        cache = [deque([]) for _ in range(self.num_workers)]
        for data in self.dataloader:
            data, worker_id = data
            cache[worker_id].append(data)
            num = sum([len(v) > 0 for v in cache])
            if num == self.num_workers:
                batch = [item.popleft() for item in cache]
                batch = list(chain.from_iterable(iter(batch)))
                 # Check if batch is all padding_value, do not yield
                any_pad = any([item == self.dataset.padding_value for item in batch])
                if any_pad:
                    break
                batch = self.collate_fn(batch)
                yield batch

        # Empty remaining cache
        # Assert no value is a true value
        for fifo in cache:
            if not len(fifo):
                continue
            while fifo:
                # print(fifo)
                item = fifo.pop()[0]
                # assert item == self.dataset.padding_value, 'code is broken, cache contained real data'

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
            self.ts += [h5obj.attrs["timestamp"]]

class H5Stream(object):
    def __init__(self,
                file_name,
                idx,
                events,
                maps,
                mode,
                window, 
                resolution,
                orig_resolution,
                num_bins,
                encoding,
                augmentation,
                augment_prob,
                predict_load,
                predict_dir,
                round_ts,
                **kwargs):
        self.file = h5py.File(file_name, "r")
        self.fname = file_name
        self.sequence_name = file_name.split("/")[-1].split(".")[0]
        self.file_predict = None
        self.idx = idx
        self.events = events
        self.maps = maps
        self.mode = mode
        self.window = window
        self.resolution = resolution
        self.orig_resolution = orig_resolution
        self.num_bins = num_bins
        self.encoding = encoding
        if predict_load:
            self.file_predict = h5py.File(os.path.join(self.predict_dir, self.sequence_name + '.h5'), "r")
        self.predict_load = predict_load
        self.predict_dir = predict_dir
        self.round_ts = round_ts

        self.augmentation = []
        for i, mechanism in enumerate(augmentation):
            if np.random.random() < augment_prob[i]:
                self.augmentation.append(mechanism)

    def hot_filter(self, batch, event_voxel, event_cnt, event_mask):
        hot_mask = self.create_hot_mask(event_cnt, batch)
        hot_mask_voxel = torch.stack([hot_mask] * self.num_bins, axis=2).permute(2, 0, 1)
        hot_mask_cnt = torch.stack([hot_mask] * 2, axis=2).permute(2, 0, 1)
        event_voxel = event_voxel * hot_mask_voxel
        event_cnt = event_cnt * hot_mask_cnt
        event_mask *= hot_mask.view((1, hot_mask.shape[0], hot_mask.shape[1]))
        return event_voxel, event_cnt, event_mask

    def load_frames(self, file, maps, cur_ts):
        curr_idx = cur_ts
        next_idx = int(np.ceil(cur_ts + self.window))

        frames = np.zeros((2, self.resolution[1], self.resolution[0]))
        img0 = file["images"][maps.names[curr_idx]][:]
        img1 = file["images"][maps.names[next_idx]][:]
        frames[0, :, :] = data_func.augment_frames(img0, self.augmentation)
        frames[1, :, :] = data_func.augment_frames(img1, self.augmentation)
        frames = torch.from_numpy(frames.astype(np.uint8))
        return frames

    def load_flow(self, file, maps, cur_ts):
        idx = int(np.ceil(cur_ts + self.window))
        flowmap = file[self.mode][maps.names[idx]][:]
        flowmap = data_func.augment_flowmap(flowmap, self.augmentation)
        flowmap = torch.from_numpy(flowmap.copy())[(1,0), ...]
        if idx > 0:
            dt_gt = maps.ts[idx] - maps.ts[idx - 1]
        return flowmap, dt_gt

    def load_events(self, cur_idx):
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

        xs = self.events['x'][idx0:idx1]
        ys = self.events['y'][idx0:idx1]
        ts = self.events['t'][idx0:idx1] - self.file.attrs.get("t0", 0)
        ps = self.events['p'][idx0:idx1]        

        # center crop
        # cy, cx = [r // 2 for r in self.orig_resolution]
        # ry, rx = [r // 2 for r in self.resolution]
        # mask = (xs > (cx - rx)) & (xs < (cx + rx)) & (ys > (cy - ry)) & (ys < (cy + ry))
        # xs, ys, ts, ps = xs[mask], ys[mask], ts[mask], ps[mask]
        # ys = ys - (cy - ry)
        # xs = xs - (cx - rx)

        # uniform downsample
        # ind = sorted(np.random.choice(len(xs), size=int(1e4), replace=False))
        # xs, ys, ts, ps = (p[ind] for p in [xs, ys, ts, ps])



        # ts -= self.events['t0']  # sequence starting at t0 = 0

        # # reset sequence if not enough input events
        # if self.config["data"]["mode"] == "events" and xs.shape[0] < self.config["data"]["window"]:
        #     return [[]] * 5

        # handle case with very few events
        if xs.shape[0] <= 10:
            xs, ys, ts, ps = np.split(np.empty([40, 0]), 4)

        # event formatting and timestamp normalization
        dt_input = np.asarray(0.0)
        if ts.shape[0] > 0:
            dt_input = np.asarray(ts[-1] - ts[0], dtype=np.float32)
        
        last_ts = ts[-1]
        xs, ys, ts, ps = data_func.event_formatting(xs, ys, ts, ps)

        # xs, ys = data_func.event_reshaping(xs, ys, self.resolution, self.orig_resolution)

        # data augmentation
        xs, ys, ps = data_func.augment_events(xs, ys, ps, self.augmentation, self.resolution)

        predict = None
        if self.predict_load:
            predict = self.predict_file['events'][idx0:idx1, -2:] # N x 2 (vx. vy)
            predict = torch.tensor(predict)

        return xs, ys, ts, ps, dt_input, last_ts, predict

    def __iter__(self):  
        cur_ts = 0

        for cur_idx in self.idx:
            # load events
            xs, ys, ts, ps, dt_input, seq_last_ts, predict = self.load_events(cur_idx)

            event_list = torch.stack([ts, xs, ys, ps], dim=-1)

            event_cnt = encoder.events_to_channels(xs, ys, ps, self.resolution)
            timesurface = encoder.events_to_timesurface(xs, ys, ts, ps, 1, self.resolution, polarity=False)

            # timesurface = torch.nn.functional.pad(timesurface, (1, 0, 1, 0), "constant", 0)

            event_mask = (event_cnt[0] + event_cnt[1]) > 0
            # event_mask = event_mask.float()
            # event_list_pol_mask = data_func.create_polarity_mask(ps)

            # # hot pixel removal
            # if self.config["hot_filter"]["enabled"]:
            #     event_voxel, event_cnt, event_mask = self.hot_filter(batch, event_voxel, event_cnt, event_mask)

            idx = '0'
            # load frames when required
            if self.mode == "images":
                # frames = self.load_frames(self.file, self.maps, cur_ts)
                frames = None
                idx = self.maps.names[int(np.ceil(cur_ts + self.window))][-6:]

            # load GT optical flow when required
            dt_gt = 0.0
            if self.mode in ["flow_dt1", "flow_dt4"]:
                flow_map, dt_gt = self.load_flow(self.file, self.maps, cur_ts)
                idx = self.maps.names[int(np.ceil(cur_ts + self.window))][-6:]
                # frames = self.load_frames(file, maps, cur_ts)
            dt_gt = np.asarray(dt_gt)

            # prepare output
            output = {}

            output['cur_ts'] = cur_ts
            output['ts'] = seq_last_ts
            output['name'] = self.sequence_name
            output['event_list'] = event_list
            if self.mode == "images":
                output['frames'] = frames
            if self.mode in ["flow_dt1", "flow_dt4"]:
                output['gtflow'] = flow_map
            if self.predict_load:
                output['predict_flow'] = predict
                output['event_cnt'] = event_cnt
            output['idx'] = idx

            output['event_mask'] = event_mask
            # output["event_list_pol_mask"] = event_list_pol_mask

            if self.encoding == 'cnt':
                output['input'] = event_cnt
            elif self.encoding == 'timesurface':
                output['input'] = timesurface * event_mask
            elif self.encoding == 'mixture':
                output['input'] = torch.cat([event_cnt, timesurface])

            output['event_cnt'] = event_cnt
            output["dt_gt"] = torch.from_numpy(dt_gt)
            output["dt_input"] = torch.from_numpy(dt_input)
            output["file_name"] = self.fname
             # update window
            cur_ts += self.window
            yield output

class H5Dataloader(StreamDataLoader):
    def __init__(self, 
                path, mode, window, 
                resolution=[255, 255],
                orig_resolution=None,
                debug=False, 
                num_bins=2, 
                batch_size=1,
                num_workers=4,
                encoding='cnt',
                augmentation=[],
                augment_prob=[],
                predict_load=False,
                predict_dir=None,
                round_ts=False,
                shuffle=False,
                **kwargs):
        # input event sequences
        self.files = []
        self.events = {}
        self.idx = {}
        self.maps = {}
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".h5"):
                    fname = os.path.join(root, file)
                    vari = 1
                    if 'VariNum' in augmentation:
                        vari = (1 - augment_prob[augmentation.index('VariNum')] * np.random.random())
                    # if 'shapes_6dof.h5' in fname:
                    # if 'reflective_materials.h5' in fname:
                    # if 'outdoor_day1_data.h5' in fname:
                    self.get_event_info(fname, mode, window)
                    if debug and len(self.files) == batch_size:
                        break
        
        orig_resolution = resolution if orig_resolution is None else orig_resolution

        def iterator_func(file_name):
            events = self.events[file_name]
            idx = self.idx[file_name]
            maps = self.maps[file_name]
            return H5Stream(file_name, idx, events, maps, mode, window, resolution, orig_resolution, num_bins, encoding, augmentation, augment_prob, predict_load, predict_dir, round_ts)
            
        super().__init__(self.files, iterator_func, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

    def get_event_info(self, fname, mode, window):
        file = h5py.File(fname, "r")
        cur_ts, last_ts = 0, 0
        maps = None
        t0 = file.attrs.get("t0", 0)
        if mode in ["images", "flow_dt1", "flow_dt4"]:
            maps = Map()
            file[mode].visititems(maps)
            last_ts = len(maps.ts)
        elif mode == "time":
            last_ts = file["events/t"][-1] - t0
        else:
            last_ts = len(file["events/t"])

        events= {'y':file["events/y"],
                'x':file["events/x"],
                't':file["events/t"],
                'p':file["events/p"],}
        idx = []
        
        while self.check_seq(mode, cur_ts, last_ts, window):
            idx0, idx1 = self.get_event_index(mode, events['t'], t0, cur_ts, window=window, maps=maps, ms_to_idx=file['events/ms_to_idx'])
            idx.append((idx0, idx1))
            cur_ts += window

        self.idx[fname] = idx
        self.events[fname] = events
        self.maps[fname] = maps
        self.files.append(fname)

        # if 'indoor_flying3_data' in fname:
        #     st = 200
        #     self.idx[fname] = self.idx[fname][st:]
        #     self.maps[fname].names = self.maps[fname].names[st:]
        #     self.maps[fname].ts = self.maps[fname].ts[st:]

    def get_event_index(self, mode, ts, t0, cur_ts, window=0, maps=None, ms_to_idx=None):
        """
        Get all the event indices to be used for reading.
        :param batch: batch index
        :param window: input window
        :return event_idx: event index
        """

        event_idx0 = None
        event_idx1 = None
        if mode == "events":
            event_idx0 = cur_ts
            event_idx1 = cur_ts + int(window)
        elif mode == "time":
            event_idx0 = ms_to_idx[(cur_ts + t0) // 1000]
            event_idx1 = ms_to_idx[(cur_ts + t0 + window) // 1000]
            # event_idx0 = data_func.binary_search_array(ts, cur_ts + t0)
            # event_idx1 = data_func.binary_search_array(ts, cur_ts + t0 + window)
        elif mode in ["images", "flow_dt1", "flow_dt4"]:
            idx0 = int(np.floor(cur_ts))
            idx1 = int(np.ceil(cur_ts + window))
            if window < 1.0 and idx1 - idx0 > 1:
                idx0 += idx1 - idx0 - 1
            event_idx0 = data_func.binary_search_array(ts, maps.ts[idx0])
            event_idx1 = data_func.binary_search_array(ts, maps.ts[idx1])
            if window < 1.0:
                event_idx0, event_idx1 = data_func.delta_time(cur_ts, window, event_idx0, event_idx1)
        else:
            print("DataLoader error: Unknown mode.")
            raise AttributeError
        
        return event_idx0, event_idx1

    def check_seq(self, mode, cur_ts, last_ts, window):
        return (mode in ["images", "flow_dt1", "flow_dt4"]
                    and int(np.ceil(cur_ts + window)) < last_ts) \
                or (mode in ["time", "events"]
                    and (cur_ts + window) < last_ts)


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