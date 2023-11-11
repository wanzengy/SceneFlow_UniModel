import argparse
import importlib
import json
import logging
import os
import time
import torch

import Tools
import numpy as np

# from tester import eval_GT
from trainer import train

os.environ['CUDA_LAUNCH_BLOCKING']='1'

def override(info, item):
    def dfs_dict(dic, str_list, v):
        if len(str_list) > 1:
            dfs_dict(dic[str_list[0].strip()], str_list[1:], v)
        else:
            dic[str_list[0].strip()] = v
        return True

    for term in info.split(','):
        k, v = term.split('=')
        dfs_dict(item, k.split('.'), eval(v.strip()))
    return item

def log_setting(cfg, log_dir=None, dump_config=True):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler() # Adding Terminal Logger

    # Adding File Logger
    if log_dir is None:
        log_dir = os.path.join(cfg['Rec']['dir'], cfg['Model']['name'], cfg['timestamp'])
        os.makedirs(log_dir, exist_ok=True)
    cfg['Rec']['dir'] = log_dir

    fh = logging.FileHandler(filename=os.path.join(log_dir, 'logger.txt'))
    fh.setFormatter(logging.Formatter("%(asctime)s  : %(message)s", "%b%d-%H:%M"))

    if not logger.handlers:
        logger.addHandler(ch)
        logger.addHandler(fh)

    if dump_config:
        logger.info(json.dumps(cfg, indent=4, separators=(',', ': ')))
    return logger

def init_seed(seed=1):
    import random

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enable = True

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
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
    
    # Testing
    # path = os.path.join(cfg['Rec']['dir'], 'best_checkpoint.pkl')
    # net.load_state_dict(torch.load(path))
    # eval_GT(cfg, net, logger, dataset=args['Test_Dataset'])