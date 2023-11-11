import argparse
import importlib
import json
import logging
import numpy as np
import os
import torch
from torch.cuda.amp import GradScaler, autocast

from Tools.metrics import EventWarping
from Tools.metrics import AEE, FWL

import Tools
from Tools.visualize import VisdomPlotter

# from tester import eval_GT

def train(cfg, net, logger, dataset='UZHFPV'):
    #---------- Environment ---------
    if cfg['Rec']['visdom']:
        plotter = VisdomPlotter(env='train', port=7000)
    
    cfg['Data'].update(cfg['Train_Dataset'][dataset])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    param_list = [{'params':net.parameters(), 'lr':cfg['Data']['lr']}]
    optimizer = torch.optim.Adam(param_list)
    # loader = H5Dataloader(**cfg["Data"])
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['Data']['num_epochs'], eta_min=1e-6, verbose=True)

    from Tools.MVSEC import H5Dataloader as Dataloader
    loader = Dataloader(**cfg["Data"], shuffle=True)
    cfg['Loss'].update(cfg['Data'])
    loss_func = EventWarping(loader.left_intrinsics, loader.right_intrinsics, **cfg['Loss']).to(device)
    loss_func = loss_func.to(device)
    scaler = GradScaler()

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2, min_lr=1e-6, verbose=True)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, verbose=True)

    #-------- Start Trainning --------
    min_loss = float('inf')
    loss_tracker = Tools.Param_Tracker()
    timer = Tools.Time_Tracker()
    n_iter = 0
    for epoch in range(cfg['Data']['num_epochs']):
        net.train()
        for i, item in enumerate(loader, start=1):
            # ---------- Forward ----------
            if item['cur_ts'][0] == 0:
                optimizer.zero_grad()
                loss_func.reset()
                net.reset_states()
    
            with autocast(dtype=torch.float32):
                out = {'left':{}, 'right':{}}
                for loc in ['left', 'right']:
                    out[loc] = net(item[loc]['event_vox'].to(device))
                    ego_motion = torch.concat([item['gt']['ang_vel'], item['gt']['lin_vel']], axis=-1)
                    ego_motion = ego_motion * item['left']['dt_input'][..., None]

                    loss_func.event_association(item[loc]['event_list'].to(device), 
                                                item[loc]['event_cnt'].to(device),
                                                out[loc]['disps'],
                                                ego_motion.to(device, dtype=torch.float32),
                                                loc)

                    # ego_motion = torch.concat([item['gt']['ang_vel'], item['gt']['lin_vel']], axis=-1)
                    # ego_motion = ego_motion * item['left']['dt_input'][..., None]
                    # of = item['gt']['of'][..., 2:258, 45:301, :] * item['left']['dt_input'].reshape(-1, 1, 1, 1)

                    # loss_func.event_association(item[loc]['event_list'].to(device), 
                    #                             item[loc]['event_cnt'].to(device),
                    #                             [item['gt']['depth'].to(device)[..., 2:258, 45:301]],
                    #                             ego_motion.to(device, dtype=torch.float32),
                    #                             loc,
                    #                             of=of.to(device),)
                
                    if (i % cfg['Data']['seq_len']) > 0:
                        continue

                    if cfg['Loss']['overwrite_intermediate']:
                        loss_func.overwrite_intermediate(out[loc]['disps'],
                                                        out[loc]['ego_motion'],
                                                        loc=loc)
            
                # loss_output = loss_func()

                loss_output = loss_func()
                loss = loss_output['loss']
                left_iwe = loss_output['left_iwe']
                loss_left_terms = loss_output['loss_left_terms']

            # ---------- Backward ----------
            scaler.scale(loss).backward()

            if cfg['Loss']['clip_grad']:
                torch.nn.utils.clip_grad.clip_grad_norm_(net.parameters(), cfg['Loss']['clip_grad'])

            scaler.step(optimizer)
            scaler.update()

            loss_tracker(loss, n=cfg['Data']['batch_size'])
            timer(cfg['Data']['batch_size'])

            # ---------- Log and Visualize ----------
            if cfg['Rec']['visdom'] and i % (cfg['Data']['seq_len'] * 5) == 0:
            # if cfg['Rec']['visdom']:
                # mask = item['left']['event_cnt'].sum(dim=1, keepdim=True) > 0
                # plotter.vis_disp(out['left']['disps'][-1].detach().cpu(), win='pred_disp')
                # plotter.vis_disp(item['gt']['depth'][:, None][..., 2:258, 45:301].detach().cpu(), win='gt_disp')

                # plot the error
                gt_depth = item['gt']['depth'][:, None][..., 2:258, 45:301]
                pred_depth = loss_func.focal / (out['left']['disps'][-1] * loss_func.flow_scaling) * loss_func.baseline 
                pred_depth = pred_depth.detach().cpu()
                pred_depth[torch.isinf(pred_depth)] = 0
                mask = (gt_depth > 0) & (pred_depth > 0)
                rmse = torch.log(gt_depth) - torch.log(pred_depth)
                rmse = torch.sqrt(torch.mean((rmse ** 2)[mask]))
                plotter.vis_curve(X=np.array([rmse.item()]), Y=np.array([n_iter]), win='rmse curve')
                n_iter += 1

                plotter.vis_disp(pred_depth.detach().cpu(), win='pred_disp')
                plotter.vis_disp(gt_depth.detach().cpu(), win='gt_disp')

                # plotter.vis_census(census_left[:, :9].detach().cpu(), win='census_left')
                # plotter.vis_census(census_right[:, :9].detach().cpu(), win='census_right')
                # plotter.vis_census(census_r_warp_l[:, :9].detach().cpu(), win='census_r_warp_l')

                # flow = out['flow'][-1].detach().cpu()
                # # flow = flow * item['event_mask'][:, None]
                # plotter.vis_flow(flow.permute(0, 2, 3, 1), win='pred_flow')
                # ts_map = net.map.detach().cpu()[0]
                plotter.vis_event(left_iwe.detach().cpu(), if_standard=True, win='left_iwe')
                # plotter.vis_event(right_iwe.detach().cpu(), if_standard=True, win='right_iwe')

                plotter.vis_flow(loss_func._flow_maps['left'][-1][:, -1].detach().cpu(), win='pred_flow')
                plotter.vis_flow(item['gt']['of'][..., 2:258, 45:301, :].detach().cpu(), win='gt_flow')
                
                plotter.vis_event(item['left']['event_cnt'].detach().cpu(), if_standard=True, win='left_event_cnt')
                # plotter.vis_event(item['right']['event_cnt'].detach().cpu(), if_standard=True, win='right_event_cnt')

                # plotter.vis_event(item['left']['event_vox'].detach().cpu()[0], if_standard=True, win='raw')
                # plotter.vis_event(iwe.detach().cpu(), if_standard=True, win='iwe',  title=f':{loss_tracker.avg:.3f}')

            if dataset in ['DSEC']:
                infor = f'{i:04d} / {len(loader.dataset):04d} '
            else:
                infor = f'{loader.dataset.pos.value:03d} / {len(loader.dataset):03d}, '
            infor += f'loss track:{loss_tracker.avg:.3f}, '
            for k, v in loss_left_terms.items():
                infor += f"{k}: {v.item() / cfg['Data']['batch_size']:.3f}, "
            print(infor + f'{timer.avg:.4f} seconds/batch', end="\r",)

            optimizer.zero_grad(set_to_none=True)
            net.detach_states()
            loss_func.reset()
            timer.start()

        # scheduler.step(train_result['loss'])
        logger.info(f"Epoch: {epoch}, loss:{loss_tracker.avg:.3f}, {timer.avg:.6f} seconds/batch")

        if cfg['Rec']['visdom']:
            plotter.vis_curve(X=np.array([loss_tracker.avg.detach().cpu()]), Y=np.array([epoch]), win='loss curve')
        
        if cfg['Rec']['enable']:
            if loss_tracker.avg < min_loss:
                torch.save(net.state_dict(), os.path.join(cfg['Rec']['dir'], 'best_checkpoint.pkl'))
                min_loss = loss_tracker.avg
                best_epoch = epoch
        
        # TEST: For epoch recording
        # if epoch % 10 == 0:
        #     # torch.save(net.state_dict(), os.path.join(cfg['Rec']['dir'], f'checkpoint_{epoch}.pkl'))
        #     eval_GT(cfg, net, logger, dataset='MVSEC')
        #     cfg['Data'].update(cfg['Train_Dataset'][dataset])
        
        loss_tracker.reset()
        timer.reset()

        # scheduler.step()

    logger.info(f"Min loss {min_loss:.3f} @ {best_epoch} epoch")


if __name__ == '__main__':
    import time
    from main import init_seed, log_setting, override

    args = argparse.ArgumentParser()
    args.add_argument('--Model', type=str, default='EVUniNet')
    args.add_argument('--Train_Dataset', type=str, default='MVSEC')
    args.add_argument('--Test_Dataset', type=str, default='MVSEC')
    args.add_argument('--timestamp', type=str, default=None)
    args.add_argument('--refine_path', type=str, default=None)
    args.add_argument('--override', type=str, help='Arguments for overriding config')
    args = vars(args.parse_args())
    cfg = json.load(open(f"Config/{args['Model']}.json", 'r'))

    init_seed(2023)

    if args['override']:
        cfg = override(args['override'], cfg)

    if args['timestamp'] is None:
        cfg['timestamp'] = time.strftime('%y%m%d%H%M', time.localtime(time.time())) # Add timestamp with format yaer-mouth-day-hour-minute
    else:
        cfg['timestamp'] = args['timestamp']
    logger = log_setting(cfg)
    cfg['Model'].update(cfg['Data'])

    # ---- load model -----
    net = importlib.import_module(f"Models.{cfg['Model']['name']}").Model(**cfg['Model'])
    if args['refine_path'] is not None:
        net.load_state_dict(torch.load(args['refine_path']))

    params = np.sum([p.numel() for p in net.parameters()]).item() * 8 / (1024 ** 2)
    logger.info(f"Lodade {cfg['Model']['name']} parameters : {params:.3e} M")
    
    # Trainning
    train(cfg, net, logger, dataset=args['Train_Dataset'])