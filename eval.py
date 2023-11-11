import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))
os.environ['CUDA_LAUNCH_BLOCKING']='1'
import h5py
import torch
import json
import importlib
import argparse
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import imageio

import numpy as np
import Tools.main_func as func

from torch.cuda.amp import autocast, GradScaler
from Tools.metrics import AEE, FWL, RSAT, BaseValidationLoss
from Tools.iwe import compute_pol_iwe, interpolate, get_flow
from Tools.visualize import Visualization
from Tools.visualize import VisdomPlotter

class Eval_Session(func.Session):
    def __init__(self, cfg, log_dir):
        self.cfg = cfg
        self.log_dir = log_dir
        self.cfg["Rec"]["Visdom"] = self.cfg["Rec"].get("Visdom", False)
        self.cfg["Rec"]["Desktop"] = self.cfg["Rec"].get("Desktop", False)
        self.cfg["Rec"]["vis_store"] = self.cfg["Rec"].get("vis_store", False)

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler() # Adding Terminal Logger
        log_file = os.path.join(self.log_dir, 'logger.txt')
        fh = logging.FileHandler(filename=log_file)
        fh.setFormatter(logging.Formatter("%(asctime)s  : %(message)s", "%b%d-%H:%M"))

        if not logger.handlers:
            logger.addHandler(ch)
            logger.addHandler(fh)

        self.logger = logger

    def load_model(self):
        super().build_env()
        
        net = importlib.import_module(f"Models.{self.cfg['Model']['name']}").Model(**self.cfg['Model'])
        net_log = torch.load(os.path.join(self.log_dir, 'checkpoint.pkl'))
        net.load_state_dict(net_log)
        self.net = net.to(self.device)

        params = np.sum([p.numel() for p in self.net.parameters()]).item()
        params = params * 8 / (1024 ** 2)
        print(f"Lodade {self.cfg['Model']['name']} parameters : {params:.3e} M")

    def generate(self):
        self.load_model()
        # self.cfg['Data']['mode'] = 'flow_dt1'
        # self.cfg['Data']['window'] = 1
        self.cfg['Data']['batch_size'] = 1
        self.cfg['Data']['predict_load'] = False
        self.cfg['Data']['predict_dir'] = os.path.join(self.log_dir, 'predict')
        loader = self.load_data(shuffle=False)

        metric = BaseValidationLoss(resolution=self.cfg['Data']['resolution'], flow_scaling=128, device=self.device)
        
        if self.cfg['Rec']['Visdom']:
            self.plotter = VisdomPlotter(env='test', port=7000)
        if self.cfg['Rec']['vis_store'] or self.cfg['Rec']['Desktop']:
            vis = Visualization(resolution=self.cfg['Data']['resolution'][0], path_results=self.log_dir)

        flow_map = None
        idx_AEE = 0

        for item in loader:

            if item['cur_ts'][0] == 0:
                print('\n', item['name'])
                self.net.reset_states()
                idx_AEE = 0

            with torch.no_grad():
                out = self.net(item['input'].to(self.device))
                flow = out['flow'][-1]

            metric.event_flow_association(
                flow,
                item['event_list'].to(self.device),
                item['event_mask'].to(self.device),
                item['dt_input'].to(self.device),
                item['dt_gt'].to(self.device),)

            if flow_map is None:
                flow_map = flow[:, None]
            else:
                flow_map = torch.cat([flow_map, flow[:, None]], dim=1)

            idx_AEE += 1
            if idx_AEE < self.cfg['Data']['test_per']:
                continue
            
            idx_AEE = 0

            events_window_vis = metric.compute_window_events()
            iwe_window_vis = metric.compute_window_iwe()
            masked_window_flow_vis = metric.compute_masked_window_flow()

            # accumulate results
            flow = flow_map.sum(axis=1)
            flow_map = None

            if self.cfg["Rec"]["enable"]:
                flow_idx = item["event_list"][:, :, 1:3].clone().to(self.device)
                flow_idx[:, :, 0] *= self.cfg['Data']['resolution'][1]  # torch.view is row-major
                flow_idx = torch.sum(flow_idx, dim=2) # B x N x 1
                event_flow = flow.view(flow.size(0), 2, -1).permute(0, 2, 1) # B x (HW) x 2
                event_flow = torch.gather(event_flow, 1, flow_idx[..., None].repeat(1, 1, 2).long())
              
                # image of warped events
                iwe = compute_pol_iwe(
                    events=item["event_list"].to(self.device),
                    flow=event_flow,
                    ts=1,
                    res=self.cfg["Data"]["resolution"],
                    flow_scaling=self.cfg["Loss"]["flow_scaling"],
                    round_idx=True,
                )
                flow *= item['event_mask'].to(self.device)

            # visualize
            if self.cfg["Rec"]["Visdom"]:
                self.plotter.vis_flow(flow.detach().cpu().numpy().transpose(0, 2, 3, 1), win='raw_flow', title=f"{item['idx']}")
                self.plotter.vis_event(item['event_cnt'].detach().cpu().numpy(), if_standard=True, win='raw')
                self.plotter.vis_event(iwe.detach().cpu().numpy(), if_standard=True, win='iwe')
            if self.cfg["Rec"]["Desktop"]:
                vis.update(
                    item, 
                    flow, 
                    iwe,
                )
            if self.cfg["Rec"]["vis_store"]:
                # sequence = item["name"][0].split("/")[-1].split(".")[0]
                # dir = f'Output/DSEC/{sequence}'
                # os.makedirs(dir, exist_ok=True)
                # img = np.zeros([flow.shape[-2], flow.shape[-1], 3], dtype=np.uint16)
                # out = flow.detach().cpu().numpy().transpose(0, 2, 3, 1)[-1] * 128
                # img[..., :2] = np.rint(out[..., (1, 0)] * 128  + 2 ** 15)
                # imageio.imwrite(os.path.join(dir, item['idx'][-1] + '.png'), img, 'PNG-FI')

                vis.store(
                    item,
                    flow,
                    iwe,
                    item['name'][-1],
                    events_window=events_window_vis, 
                    masked_window_flow=masked_window_flow_vis, 
                    iwe_window=iwe_window_vis,
                    ts=item['ts'][-1],
                    img_idx=int(item['idx'][-1])
                )

            os.makedirs('Output/RelativeTimeNet/bitahub/results/19_20/flow_results', exist_ok=True)
            np.save(f"Output/RelativeTimeNet/bitahub/results/19_20/flow_results/{int(item['idx'][-1])}", masked_window_flow_vis.detach().cpu().numpy())
            
            print(
                "[{:04f}/{:03d})]".format(
                    item['cur_ts'][-1] / self.cfg['Data']['window'],
                    len(loader.idx[item['file_name'][-1]]),
                    ),
                end="\r",
                )

            # reset criteria
            metric.reset()
        print('Done')

    def eval_FWL(self):
        self.load_model()
        metric = FWL(resolution=self.cfg['Data']['resolution'], flow_scaling=128, device=self.device)
        self.cfg['Data']['mode'] = 'events'
        self.cfg['Data']['window'] = 15000
        self.cfg['Data']['batch_size'] = 1
        self.cfg['Data']['predict_load'] = False
        loader = self.load_data(shuffle=False)
        self.net.eval()
        time = func.Time_Tracker()
        val_results = {}
        
        if self.cfg['Rec']['enable']:
            self.plotter = VisdomPlotter(env='test', port=7000)
            vis = Visualization(resolution=self.cfg['Data']['resolution'][0], path_results=self.log_dir)

        # cfg_FireNet = {
        #     'num_bins':2,
        #     'base_num_channels':32,
        #     'kernel_size':3,
        #     'encoding':'cnt',
        #     'mask_output':True,
        #     'activations':['relu', None],
        #     'norm_input':False
        # }
        # self.pre_net = FireNet(**cfg_FireNet).to(self.device)
        # self.pre_net.load_state_dict(torch.load('Output/FireNet/checkpoint.pkl'))
        # self.pre_net.eval()

        for item in loader:
            if item['cur_ts'][0] == 0:
                print('\n', item['name'])
                time.reset()
                self.net.reset_states()
            
            # pred_flow = torch.zeros((item['event_list'].size(0), 2, item['event_list'].size(1))).to(self.device)
            # flow = self.pre_net(item['event_voxel'].to(self.device), item['event_cnt'].to(self.device))['flow'][0]
            # flow = flow[:, (1, 0)]
            # B, _, W, H = flow.size()

            # pred_flow *= 128
            with torch.no_grad():
                out = self.net(item['input'].to(self.device))
                flow = out['flow']

            events_window_vis = None
            masked_window_flow_vis = None
            iwe_window_vis = None

            metric.event_flow_association(
                flow[-1], 
                item['event_list'].to(self.device),
                # item['event_mask'].to(self.device),
                # item['dt_input'].to(self.device),
                # item['dt_gt'].to(self.device),
                )

            # overwrite intermedia flow estimates with the final ones
            if self.cfg["Loss"]["overwrite_intermediate"]:
                metric.overwrite_intermediate_flow(
                    item["predict_flow"].to(self.device))

            # compute metric
            val_metric, IWE, IE = metric()

            # accumulate results
            for i in range(self.cfg["Data"]["batch_size"]):
                fn = item["name"][i].split("/")[-1]
                if fn not in val_results.keys():
                    val_results[fn] = {}
                    val_results[fn]["metric"] = 0
                    val_results[fn]["it"] = 0

                val_results[fn]["it"] += 1
                val_results[fn]["metric"] += val_metric[i].cpu().numpy()

            print(
                "[{:03d}/{:03d} ({:03d}%)] Metric: {:.6f}".format(
                    loader.dataloader.dataset.pos.value,
                    len(loader.dataloader.dataset),
                    int(100 * loader.dataloader.dataset.pos.value / len(loader.dataloader.dataset)),
                    val_metric.item(),
                    ),
                end="\r",
                )

            # visualize
            if self.cfg["Rec"]["visdom"]:
                # self.plotter.vis_flow(flow.detach().cpu().numpy().transpose(0, 3, 2, 1), win='raw_flow'
                # flow_map /= item['dt_input']
                # dh = slice(5, 40)
                # dw = slice(80, 115)
                # mask = item['event_cnt'].sum(dim = 1)
                # mask = mask > 1
                flow = flow[-1].detach().cpu() * item['event_mask'][:, None]
                # flow = flow[-1].detach().cpu()
                flow = flow[:, (1, 0)]

                # flow = flow[..., dh, dw]
                # IE = IE[..., dh, dw]
                # IWE = IWE[..., dh, dw]
                IE = item['input']

                self.plotter.vis_flow(flow.numpy().transpose(0, 2, 3, 1), win='pred_flow')
                self.plotter.vis_event(IE.detach().cpu().numpy(), if_standard=True, win='raw')
                self.plotter.vis_event(IWE.detach().cpu().numpy(), if_standard=True, win='IWE')

                # vis.update(
                #     item, 
                #     flow_map,
                #     iwe,
                # )
            
            # reset criteria
            metric.reset()
        
        v = 0
        it = 0
        for fn, val in val_results.items():
            v += val['metric']
            it += val['it']
            print(f"{fn} Metrics: {val['metric'] / val['it']:.3f}")     
        print(v / it)
    
    def eval_GT(self):
        self.load_model()
        metric = AEE(resolution=self.cfg['Data']['resolution'], device=self.device, flow_scaling=self.cfg['Loss']['flow_scaling'])
        # self.cfg['Data']['mode'] = 'flow_dt1'
        # self.cfg['Data']['window'] = 1
        self.cfg['Data']['batch_size'] = 1
        self.cfg['Data']['predict_load'] = False
        self.cfg['Data']['predict_dir'] = os.path.join(self.log_dir, 'predict')
        loader = self.load_data(shuffle=False)

        idx_AEE = 0
        val_results = {}
        
        # cfg_FireNet = {
        #     'num_bins':2,
        #     'base_num_channels':32,
        #     'kernel_size':3,
        #     'encoding':'voxel',
        #     'mask_output':True,
        #     'activations':['relu', None],
        #     'norm_input':False
        # }
        # self.pre_net = FireNet(**cfg_FireNet).to(self.device)
        # self.pre_net.load_state_dict(torch.load('Output/FireNet/checkpoint.pkl'))
        # self.pre_net.eval()

        if self.cfg['Rec']['Visdom']:
            self.plotter = VisdomPlotter(env='test', port=7000)
        if self.cfg['Rec']['vis_store'] or self.cfg['Rec']['Desktop']:
            vis = Visualization(resolution=self.cfg['Data']['resolution'][0], path_results=self.log_dir)

        for item in loader:

            if item['cur_ts'][0] == 0:
                idx_AEE = 0
                for fn, val in val_results.items():
                    print(f"{fn} Metrics: {val['metric'] / val['it']:.3f}, Percent: {val['percent'] / val['it'] * 100:.3f}")     
                print('\n', item['name'])
                self.net.reset_states()

            with torch.no_grad():
                out = self.net(item['input'].to(self.device))
                flow = out['flow'][-1]

            # events_window_vis = None
            # masked_window_flow_vis = None
            # iwe_window_vis = None

            metric.event_flow_association(
                flow,
                item['event_list'].to(self.device),
                item['event_mask'].to(self.device),
                item['dt_input'].to(self.device),
                item['dt_gt'].to(self.device),
                item['gtflow'].to(self.device))

            # overwrite intermedia flow estimates with the final ones
            if self.cfg["Loss"]["overwrite_intermediate"]:
                metric.overwrite_intermediate_flow(
                    flow,
                    item['event_list'].to(self.device),
                    )

            idx_AEE += 1
            if (item["dt_gt"] <= 0.0) or (idx_AEE != np.round(1.0 / self.cfg["Data"]["window"])):
                continue

            # compute metric
            val_metric = metric()
            idx_AEE = 0

            events_window_vis = metric.compute_window_events()
            iwe_window_vis = metric.compute_window_iwe()
            masked_window_flow_vis = metric.compute_masked_window_flow()

            # accumulate results
            for i in range(self.cfg["Data"]["batch_size"]):
                fn = item["name"][i].split("/")[-1]
                if fn not in val_results.keys():
                    val_results[fn] = {}
                    val_results[fn]["metric"] = 0
                    val_results[fn]["it"] = 0
                    val_results[fn]["percent"] = 0

                val_results[fn]["it"] += 1
                val_results[fn]["metric"] += val_metric["metric"][i].cpu().numpy()
                val_results[fn]["percent"] += val_metric["percent"][i].cpu().numpy()

            if self.cfg['Rec'].get('verbose',True):
                print(
                    "[{:03d}/{:03d} ({:03d}%)] Metric: {:.6f}".format(
                        loader.dataloader.dataset.pos.value,
                        len(loader.dataloader.dataset),
                        int(100 * loader.dataloader.dataset.pos.value / len(loader.dataloader.dataset)),
                        val_metric["metric"][0],
                    ),
                    end="\r",
                )

            if self.cfg["Rec"]["Visdom"] or self.cfg["Rec"]["Desktop"] or self.cfg["Rec"]["vis_store"]:
                flow_idx = item["event_list"][:, :, 1:3].clone().to(self.device)
                flow_idx[:, :, 0] *= self.cfg['Data']['resolution'][1]  # torch.view is row-major
                flow_idx = torch.sum(flow_idx, dim=2) # B x N x 1
                event_flow = flow.view(flow.size(0), 2, -1).permute(0, 2, 1) # B x (HW) x 2
                event_flow = torch.gather(event_flow, 1, flow_idx[..., None].repeat(1, 1, 2).long())
              
                # image of warped events
                iwe = compute_pol_iwe(
                    events=item["event_list"].to(self.device),
                    flow=event_flow,
                    ts=1,
                    res=self.cfg["Data"]["resolution"],
                    flow_scaling=self.cfg["Loss"]["flow_scaling"],
                    round_idx=True,
                )
                flow *= item['event_mask'].to(self.device)

            # visualize
            if self.cfg["Rec"]["Visdom"]:
                self.plotter.vis_flow(flow.detach().cpu().numpy().transpose(0, 2, 3, 1), win='raw_flow', title=f'{val_metric["metric"][0]:.3f}')
                self.plotter.vis_flow(item['gtflow'].numpy().transpose(0, 2, 3, 1), win='gt_flow')
                self.plotter.vis_event(item['event_cnt'].detach().cpu().numpy(), if_standard=True, win='raw')
                self.plotter.vis_event(iwe.detach().cpu().numpy(), if_standard=True, win='iwe')
            if self.cfg["Rec"]["Desktop"]:
                vis.update(
                    item, 
                    flow, 
                    iwe,
                )
            if self.cfg["Rec"]["vis_store"]:
                sequence = item["name"][0].split("/")[-1].split(".")[0]
                vis.store(
                    item,
                    flow,
                    iwe,
                    sequence,
                    events_window=events_window_vis, 
                    masked_window_flow=masked_window_flow_vis, 
                    iwe_window=iwe_window_vis,
                    ts=item['ts'],
                )
            
            # reset criteria
            metric.reset()
        
        for fn, val in val_results.items():
            self.logger.info(f"{fn} Metrics: {val['metric'] / val['it']:.3f}, Percent: {val['percent'] / val['it'] * 100:.3f}")


def dfs_dict(dic, str_list, v):
    if len(str_list) > 1:
        dfs_dict(dic[str_list[0].strip()], str_list[1:], v)
    else:
        dic[str_list[0].strip()] = v
    return True

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config',
                    type=str,
                    default='ECD_RelativeTimeNet')
    args.add_argument('--log_dir',
                    type=str,
                    default='Output/RelativeTimeNet/04182127',
                    help='Arguments for overriding config')
    args.add_argument('--override',
                    type=str)
    args = vars(args.parse_args())
    cfg = json.load(open(f"Config/{args['config']}.json", 'r'))

    if args['override']:
        override = args['override'].split(',')
        for term in override:
            k, v = term.split('=')
            dfs_dict(cfg, k.split('.'), eval(v.strip()))
    
    sess = Eval_Session(cfg, args['log_dir'])
    # sess.generate()
    # sess.eval_GT()
    sess.eval_FWL()

    # sess.visualize1()
    # sess.profile()