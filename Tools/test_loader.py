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
import multiprocessing

import numpy as np

import encodings
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
                padding_mode='data',
                padding_value=None,
                ):

        self.stream_list = stream_list
        self.batch_size = batch_size
        self.streamer = streamer
        self.padding_mode = padding_mode
        self.padding_value = padding_value
        assert padding_mode in ['zeros', 'data']
        self.mutex = multiprocessing.Lock() 
        self.pos = multiprocessing.Value('i', 0)
        self.num_actives = multiprocessing.Value('i', 0)

    def shuffle(self):
        random.shuffle(self.stream_list)

    def _set_seed(self):
        """ so that data is different along threads and epochs"""
        worker = torch.utils.data.get_worker_info()
        worker_id = int(worker.id) if worker is not None else 0
        seed = int(time.time()) + worker_id
        np.random.seed(seed)
        random.seed(seed)

    def init_position(self):
        self.mutex.acquire()
        self.pos.value = 0
        self.num_actives.value = 0
        self.mutex.release()
    
    def __len__(self):
        return len(self.stream_list)

    def __iter__(self):
        """Iterates over stream files

        Note: Here we use a mutex (WIP, pytest not working!)

        Note: Here the scheduling of iterable is done at the beginning.
        Instead User can change this code to map lazily iterables.
        """
        self._set_seed()

        #initialization this should be done in worker_init_fnx
        worker = torch.utils.data.get_worker_info()
        worker_id = int(worker.id) if worker is not None else 0

        num_workers = 1 if worker is None else worker.num_workers
        split_sizes = split_batch_size(self.batch_size, num_workers)
        worker = torch.utils.data.get_worker_info()
        worker_id = int(worker.id) if worker is not None else 0
        split_size = split_sizes[worker_id]

        if len(self) < split_size:
            print('worker#', worker_id, ': Stopping... Number of streams < split_size')
            raise StopIteration

        """
        Just-in-time mapping
        The scheduling is done as we iterate.

        EDIT 9/7/2021: The position in the stream is shared accross workers
        This allows us to avoid the non ideal pre-iteration splitting of the dataset
        """
        def increment_pos():
            self.mutex.acquire()
            pos = self.pos.value
            stream = self.stream_list[pos%len(self.stream_list)]
            self.pos.value = pos + 1
            self.mutex.release()
            return stream

        iterators = []
        for i in range(split_size):
            stream = increment_pos()
            stream = iter(self.streamer(stream))
            iterators.append(stream)

        actives = [1 for i in range(len(iterators))]
        _num_actives = sum(actives)
        self.mutex.acquire()
        self.num_actives.value += _num_actives
        self.mutex.release()

        while self.num_actives.value:
            values = []
            for i, it in enumerate(iterators):
                try:
                    value = next(it)
                    assert value is not None
                except StopIteration:
                    if actives[i] and (self.pos.value >= len(self.stream_list)):
                        self.mutex.acquire()
                        self.num_actives.value -= 1
                        self.mutex.release()
                        if self.num_actives.value == 0:
                            yield None, worker_id
                    actives[i] = 1 * (self.pos.value < len(self.stream_list))
                        
                    if self.padding_mode == 'data' or actives[i]:
                        assert stream is not None, self.pos.value
                        stream = increment_pos()
                        iterators[i] = iter(self.streamer(stream))
                        value = next(iterators[i])
                    elif self.padding_mode == 'zeros':
                        value = 0
                values.append(value)
            yield tuple(values), worker_id

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


class H5Dataset(StreamDataset):
    def __init__(self, 
                path, mode, window, 
                resolution=[255, 255],
                debug=False, 
                num_bins=3, 
                batch_size=1, 
                augmentation=["Horizontal", "Vertical", "Polarity"],
                augment_prob=[0.5, 0.5, 0.5],
                round_ts=False,
                **kwargs):
        # input event sequences
        self.mode = mode
        self.window = window
        self.num_bins = num_bins
        self.resolution = resolution
        self.augmentation = augmentation
        self.augment_prob = augment_prob
        self.round_ts = round_ts
        self.files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".h5"):
                    self.files.append(os.path.join(root, file))
        
        if debug:
            self.files = sorted(self.files)[-batch_size:]

        super().__init__(self.files, self.seq_load, batch_size=batch_size)
    
    def get_event_index(self, file, cur_ts, window=0, maps=None):
        """
        Get all the event indices to be used for reading.
        :param batch: batch index
        :param window: input window
        :return event_idx: event index
        """

        event_idx0 = None
        event_idx1 = None
        if self.mode == "events":
            event_idx0 = cur_ts
            event_idx1 = cur_ts + window
        elif self.mode == "time":
            event_idx0 = data_func.binary_search_array(file["events/ts"], cur_ts + file.attrs["t0"])
            event_idx1 = data_func.binary_search_array(file["events/ts"], cur_ts + file.attrs["t0"] + window)
        elif self.mode in ["images", "flow_dt1", "flow_dt4"]:
            idx0 = int(np.floor(cur_ts))
            idx1 = int(np.ceil(cur_ts + window))
            if window < 1.0 and idx1 - idx0 > 1:
                idx0 += idx1 - idx0 - 1
            event_idx0 = data_func.binary_search_array(file["events/ts"], maps.ts[idx0])
            event_idx1 = data_func.binary_search_array(file["events/ts"], maps.ts[idx1])
            if self.window < 1.0:
                event_idx0, event_idx1 = data_func.delta_time(cur_ts, self.window)
        else:
            print("DataLoader error: Unknown mode.")
            raise AttributeError
        
        return event_idx0, event_idx1

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

        frames = np.zeros((2, self.resolution[0], self.resolution[1]))
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
        flowmap = torch.from_numpy(flowmap.copy())
        if idx > 0:
            dt_gt = maps.ts[idx] - maps.ts[idx - 1]
        return flowmap, dt_gt

    def load_events(self, file, cur_ts, maps = None):
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
        idx0, idx1 = self.get_event_index(file, cur_ts, window=self.window, maps=maps)
        xs = file["events/xs"][idx0:idx1]
        ys = file["events/ys"][idx0:idx1]
        ts = file["events/ts"][idx0:idx1]
        ps = file["events/ps"][idx0:idx1]
        ts -= file.attrs["t0"]  # sequence starting at t0 = 0

        # # reset sequence if not enough input events
        # if self.config["data"]["mode"] == "events" and xs.shape[0] < self.config["data"]["window"]:
        #     return [[]] * 5

        # handle case with very few events
        if xs.shape[0] <= 10:
            xs, ys, ts, ps = np.split(np.empty([4, 0]), 4)

        # event formatting and timestamp normalization
        dt_input = np.asarray(0.0)
        if ts.shape[0] > 0:
            dt_input = np.asarray(ts[-1] - ts[0])
        xs, ys, ts, ps = data_func.event_formatting(xs, ys, ts, ps)

        # data augmentation
        xs, ys, ps = data_func.augment_events(xs, ys, ps, self.augmentation, self.resolution)

        return xs, ys, ts, ps, dt_input

    def check_seq(self, cur_ts, last_ts):
        return (self.mode in ["images", "flow_dt1", "flow_dt4"]
                    and int(np.ceil(cur_ts + self.window)) < last_ts) \
                or (self.mode in ["time", "events"]
                    and cur_ts + self.window <= last_ts)
        
    def seq_load(self, file_name):
        file = h5py.File(file_name, "r")
        sequence_name = file_name.split("/")[-1].split(".")[0]
        maps = None
        cur_ts, last_ts = 0, 0

        if self.mode in ["images", "flow_dt1", "flow_dt4"]:
            maps = Map()
            file[self.mode].visititems(maps)
            last_ts = len(maps.ts)
        elif self.mode == "time":
            last_ts = file["events/ts"][-1] - file.attrs["t0"]
        else:
            last_ts = len(file["events/ts"])

        augmentation = {mechanism: False for mechanism in self.augmentation}
        for i, mechanism in enumerate(self.augmentation):
            if mechanism != "Pause":    
                augmentation[mechanism] = True if np.random.random() < self.augment_prob[i] else False

        while self.check_seq(cur_ts, last_ts):
            # load events
            xs, ys, ts, ps, dt_input = self.load_events(file, cur_ts, maps)

            # events to tensors
            event_cnt = encodings.events_to_channels(xs, ys, ps, self.resolution)
            event_mask = encodings.events_to_mask(xs, ys, ps, self.resolution)
            event_voxel = encodings.events_to_voxel(xs, ys, ts, ps, self.num_bins, self.resolution, self.round_ts)
            event_list = torch.stack([ts, ys, xs, ps])
            event_list_pol_mask = data_func.create_polarity_mask(ps)

            # # hot pixel removal
            # if self.config["hot_filter"]["enabled"]:
            #     event_voxel, event_cnt, event_mask = self.hot_filter(batch, event_voxel, event_cnt, event_mask)

            # load frames when required
            if self.mode == "frames":
                frames = self.load_frames(file, maps, cur_ts)

            # load GT optical flow when required
            dt_gt = 0.0
            if self.mode in ["flow_dt1", "flow_dt4"]:
                flow_map, dt_gt = self.load_flow(file, maps, cur_ts)
            dt_gt = np.asarray(dt_gt)

            # prepare output
            output = {}

            output['cur_ts'] = cur_ts
            output['ts'] = ts[-1]
            output['last_ts'] = last_ts
            output['name'] = sequence_name
            output["event_list"] = event_list
            output["event_list_pol_mask"] = event_list_pol_mask
            if self.mode == "images":
                output["frames"] = frames
            if self.mode in ["flow_dt1", "flow_dt4"]:
                output["gtflow"] = flow_map
            
            output['event_cnt'] = event_cnt
            output['event_mask'] = event_mask
            output['event_voxel'] = event_voxel
            output["dt_gt"] = torch.from_numpy(dt_gt)
            output["dt_input"] = torch.from_numpy(dt_input)
             # update window
            cur_ts += self.window
            yield output

if __name__ == '__main__':
    data = [[10, 11, 12, 13],
            [20, 21, 22],
            [42, 43],
            [90],
            [100]
            ]
    def temp(x):
        for t in x:
            yield t
    dataset = StreamDataset(data, temp, batch_size=4)
    # dataset = H5Loader()
    # dataset=None
    loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=4)
    for i in loader:
        print(list(i))
