import argparse
import importlib
import json
import logging
import numpy as np
import os
import torch
from torch.cuda.amp import GradScaler, autocast

from Tools.iwe import compute_pol_iwe
from Tools.metrics import AEE, FWL, Depth_Benchmark

import Tools
# from Tools.stream_loader import H5Dataloader
# from Tools.dsec_loader import H5Dataloader

from Tools.visualize import VisdomPlotter, Visualization

def log_setting(cfg):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler() # Adding Terminal Logger

    # Adding File Logger

    fh = logging.FileHandler(filename=os.path.join(cfg['Rec']['dir'] , 'logger.txt'))
    fh.setFormatter(logging.Formatter("%(asctime)s  : %(message)s", "%b%d-%H:%M"))

    if not logger.handlers:
        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger

def eval_FWL(cfg, net, logger, dataset='MVSEC'):
    #-------- Start Testing --------
    cfg['Data'].update(cfg['Test_Dataset'][dataset])
    cfg['Data'].update({'batch_size':1,
                        'resolution': cfg['Data']['resolution'],
                        'path': cfg['Data']['path'],
                        'mode':'images',
                        'window':1,
                        'augment_prob':[],
                        'augmentation':[]})

    if cfg['Rec']['enable']:
        plotter = VisdomPlotter(env='test', port=7000)
        vis = Visualization(resolution=cfg['Data']['resolution'][0], path_results=cfg['Rec']['dir'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    
    loader = H5Dataloader(**cfg["Data"], shuffle=False)
    
    metric = FWL(resolution=cfg['Data']['resolution'], flow_scaling=cfg['Loss']['flow_scaling'], device=device)
    val_results = {}

    net.eval()
    with torch.no_grad():
        for item in loader:
            if item['cur_ts'][0] == 0:
                print('\n')
                # net.reset_states()
    
            # ---------- Predict ----------
            item = {k:v.to(device) if type(v) is torch.Tensor else v for k, v in item.items()}
            out = net(item['input'])
            flow = out['flow'][-1]

            # ---------- Metric Computation ----------
            metric.event_flow_association(flow, item['event_list'])
                
            if cfg['Loss']['overwrite_intermediate']:
                metric.overwrite_intermediate_flow(flow)

            if cfg['Data']['mode'] == 'events' and metric.num_events < cfg['Data']['window']:
                continue

            val_metric, iwe, ie = metric()
            fn = item["name"][0].split("/")[-1]
            if fn not in val_results.keys():
                val_results[fn] = Tools.Param_Tracker()

            val_results[fn](val_metric.item())

           # ---------- Log and Visualize ----------
            if cfg['Rec']['enable']:
                flow = flow * item['event_mask'][:, None]
                plotter.vis_flow(flow.detach().cpu().permute(0, 2, 3, 1), win='pred_flow')
                plotter.vis_event(item['event_cnt'].detach().cpu(), if_standard=True, win='raw')
                plotter.vis_event(iwe.detach().cpu(), if_standard=True, win='iwe')

            infor = f'{loader.dataset.pos.value:03d} / {len(loader.dataset):03d}' +\
                    f'{fn} FWL: {val_results[fn].avg:.3f}'
            print(infor, end="\r",)

            metric.reset()

    for fn, val in val_results.items():
        logger.info(f"{fn} FWL: {val.avg:.3f}")

def eval_GT(cfg, net, logger, dataset='MVSEC'):   
    #-------- Start Testing --------
    cfg['Data'].update(cfg['Test_Dataset'][dataset])
    cfg['Data'].update({'batch_size':1,
                        'augment_prob':[],
                        'augmentation':[]})
    
    if cfg['Rec']['visdom']:
        plotter = VisdomPlotter(env='test', port=7000)
        
        # if cfg['Rec']['store']:
        #     vis = Visualization(resolution=cfg['Data']['resolution'][0], path_results=cfg['Rec']['dir'])
        # else:
        #     vis = Visualization(resolution=cfg['Data']['resolution'][0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device) 
    if dataset in ['DSEC']:
        loader = Tools.dsec_loader.H5Dataloader(**cfg["Data"], shuffle=False)
    else:
        from Tools.MVSEC import H5Dataloader as Dataloader
        loader = Dataloader(**cfg["Data"], shuffle=True)

    metric = Depth_Benchmark(resolution=cfg["Data"]["resolution"], 
                             baseline=cfg["Data"]["baseline"],
                             focal=cfg["Data"]["focal"],
                             flow_scaling=cfg["Loss"]["flow_scaling"],
                             mask_output=True,
                             overwrite_intermediate=False,
                             device=device)
    val_results = {}
    timer = Tools.Time_Tracker()

    n_iter = 0
    net.eval()
    with torch.no_grad():
        for item in loader:
            if item['cur_ts'][0] == 0:
                print('\n')
                net.reset_states()

            # ---------- Predict --------
            # for k in item.keys():
            #     if type(item[k]) is torch.tensor:
            #         item[k].to(device)
            
            if cfg['Model']['name'] in ['ERaft']:
                out = net(item['input'])
            else:
                out = net(item['left']['event_vox'].to(device))
            
            disp = out['disps'][-1]
            timer()

            # ---------- Metric Computation ----------
            metric.event_association(disp, 
                                     item['left']['event_cnt'].to(device),
                                     item['gt']['depth'][..., 2:258, 45:301].to(device))
                
            if cfg['Loss']['overwrite_intermediate']:
                metric.overwrite_intermediate(disp, item['left']['event_cnt'].to(device))

            # if (item["dt_gt"] <= 0.0) or (metric.passes != np.round(1.0 / cfg["Data"]["window"])):
                # continue

            val_metric = metric()
            fn = item["name"][0].split("/")[-1]
            if fn not in val_results.keys():
                val_results[fn] = {}
                for k in val_metric.keys():
                    val_results[fn][k] = Tools.Param_Tracker()

            for k in val_metric.keys():
                val_results[fn][k](val_metric[k].item())

           # ---------- Log and Visualize ----------
            if cfg['Rec']['visdom']:
                depth = metric._depth_maps[-1]
                gt_depth = metric._gt_depth_maps[-1]

                mask = (metric._event_cnt[-1].sum(dim=1, keepdim=True) > 0)
                depth *= mask

                plotter.vis_disp(depth.detach().cpu(), win='pred_depth')
                plotter.vis_disp(gt_depth.detach().cpu(), win='gt_depth')

                for k in val_metric.keys():
                    plotter.vis_curve(X=np.array([val_metric[k].item()]), Y=np.array([n_iter]), win=k)

                n_iter += 1
            # if cfg["Rec"]["store"]:
                # flow = metric._flow_map[:, -1] * metric.flow_scaling
                
                # flow = flow * metric._event_mask.sum(1).bool()
                # vis.update(item, masked_window_flow=flow, iwe_window=iwe)
                
            # if cfg["Rec"]["store"]:
            #     sequence = item["name"][0].split("/")[-1].split(".")[0]
            #     vis.store(item, masked_window_flow=flow, iwe_window=iwe, sequence=sequence, ts=item['ts'])
            #     vis.store(item, flow, iwe, sequence, ts=item['ts'], other_info = f'AEE:{val_metric.item():.3f}')

            infor = f"{loader.dataset.pos.value:03d} / {len(loader.dataset):03d} \t"
            for k, v in val_metric.items():
                infor += f"{k}: {v.item():.3f}, "
            print(infor, end="\r",)
            metric.reset()
            timer.start()

    for fn, val in val_results.items():
        for k, v in val.items():
            logger.info(f"{fn} {k}: {v.avg:.3f}")

def save(cfg, net, logger, dataset='MVSEC'):
    #-------- Start Testing --------
    cfg['Data'].update(cfg['Test_Dataset'][dataset])
    cfg['Data'].update({'batch_size':1,
                        'resolution': cfg['Data']['resolution'],
                        'path': cfg['Data']['path'],
                        'mode':'images',
                        'window':2,
                        'augment_prob':[],
                        'augmentation':[]})

    if cfg['Rec']['enable']:
        plotter = VisdomPlotter(env='test', port=7000)
        vis = Visualization(resolution=cfg['Data']['resolution'][0], path_results=cfg['Rec']['dir'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    
    if dataset in ['DSEC']:
        cfg['Data']['mode'] ='test'
        loader = Tools.dsec_loader.H5Dataloader(**cfg["Data"], shuffle=True)
    else:
        loader = Tools.stream_loader.H5Dataloader(**cfg["Data"], shuffle=True)
    
    metric = FWL(resolution=cfg['Data']['resolution'], flow_scaling=cfg['Loss']['flow_scaling'], device=device)
    timer = Tools.Time_Tracker()

    net.eval()
    with torch.no_grad():
        for i, item in enumerate(loader):
            if item['cur_ts'][0] == 0:
                print('\n')
                if dataset == 'DSEC':
                    flow_init = None
                    valid_index = np.genfromtxt(
                            f"Datasets/DSEC/test/raw/{item['name'][0]}/test_forward_flow_timestamps.csv",
                            delimiter=',')[:,2].tolist()
                if cfg['Model']['name'] not in ['ERaft']:
                    net.reset_states()
    
            # ---------- Predict ----------
            item = {k:v.to(device) if type(v) is torch.Tensor else v for k, v in item.items()}
            if cfg['Model']['name'] in ['ERaft']:
                flow_low, out = net(item['input'], flow_init = flow_init)
                flow = out['flow'][-1]
                flow_init = forward_interpolate_tensor(flow_low)
            else:
                out = net(item['input'], cfg['Data']['resolution'])
                flow = out['flow'][-1] * 128 / 0.1
            timer()

            # ---------- Metric Computation ----------
            metric.event_flow_association(flow, item['event_list'])
                
            # if cfg['Loss']['overwrite_intermediate']:
            #     metric.overwrite_intermediate_flow(flow)

            if cfg['Data']['mode'] == 'events' and metric.num_events < cfg['Data']['window']:
                continue

            fn = item["name"][0].split("/")[-1]

            # ---------- Log and Visualize ----------
            sequence = item["name"][0].split("/")[-1].split(".")[0]
            # iwe = compute_pol_iwe(item["event_list"].to(device), metric._flow_list, 1, cfg["Data"]["resolution"], cfg["Loss"]["flow_scaling"], True,)
            iwe = compute_pol_iwe(item["event_list"].to(device), metric._flow_list, 1, cfg["Data"]["resolution"], 1, True,)

            # flow = flow * item['event_mask'][:, None]
            if dataset == 'DSEC':
                if len(valid_index) > 0 and int(item['idx'][0]) == int(valid_index[0]):
                    import imageio
                    dir = f"Output/{cfg['Model']['name']}/DSEC/Upload/{sequence}"
                    os.makedirs(dir, exist_ok=True)
                    img = np.zeros([flow.shape[-2], flow.shape[-1], 3], dtype=np.uint16)
                    out = flow.detach().cpu().numpy().transpose(0, 2, 3, 1)[-1]
                    img[..., :2] = np.rint(out * 128  + 2 ** 15)
                    imageio.imwrite(os.path.join(dir, f"{item['idx'][-1]:06d}.png"), img, 'PNG-FI')
                    valid_index.pop(0)

            if cfg['Rec']['enable']:
                vis.store(item, flow, iwe=iwe, sequence=sequence, ts=item['ts'])
                plotter.vis_flow(flow.detach().cpu().permute(0, 2, 3, 1), win='pred_flow')
                plotter.vis_event(item['event_cnt'].detach().cpu(), if_standard=True, win='raw')
                # plotter.vis_event(iwe.detach().cpu(), if_standard=True, win='iwe')
            
            # infor = f'{loader.dataset.pos.value:03d} / {len(loader.dataset):03d}' +\
            #         f"elpsed: {timer.avg:.4f} s"
            infor = f'{i:04d} / {len(loader.dataset):04d} ' +\
                    f"elpsed: {timer.avg:.4f} s"
            print(infor, end="\r",)

            metric.reset()
            timer.start()

if __name__ == '__main__':
    from main import init_seed, log_setting, override

    args = argparse.ArgumentParser()
    args.add_argument('--Model', type=str, default='EVUniNet')
    args.add_argument('--Test_Dataset', type=str, default='MVSEC')
    args.add_argument('--dir', type=str, default='Output/EVUniNet/2311092151', help='Arguments for overriding config')
    args.add_argument('--override', type=str, help='Arguments for overriding config')
    args = vars(args.parse_args())
    cfg = json.load(open(f"Config/{args['Model']}.json", 'r'))

    if args['override']:
        cfg = override(args['override'], cfg)

    logger = log_setting(cfg, log_dir = args['dir'], dump_config=False)
    cfg['Model'].update(cfg['Data'])
    
    # ---- load model -----
    net = importlib.import_module(f"Models.{cfg['Model']['name']}").Model(**cfg['Model'])
    if cfg['Model']['name'] in ['SpikeFlowNet', 'STEFlowNet']:
        checkpoint = torch.load(os.path.join(cfg['Rec']['dir'], 'steflow_dt1.pth.tar'))['state_dict']
    elif cfg['Model']['name'] in ['ERaft']:
        checkpoint = torch.load(os.path.join(cfg['Rec']['dir'], 'dsec.tar'))['model']
    else:
        checkpoint = torch.load(os.path.join(cfg['Rec']['dir'], 'best_checkpoint.pkl'))
    
    net.load_state_dict(checkpoint)

    params = np.sum([p.numel() for p in net.parameters()]).item() * 8 / (1024 ** 2)
    print(f"Lodade {cfg['Model']['name']} parameters : {params:.3e} M")

    #---------- Environment ---------
    eval_GT(cfg, net, logger, args['Test_Dataset'])
    # eval_FWL(cfg, net, logger, args['Test_Dataset'])
    # save(cfg, net, logger, args['Test_Dataset'])