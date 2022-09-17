# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2021-9-24
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2021 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
from os.path import join
import time
from turtle import back
import yaml
import shutil
import random
import argparse
import logging
import numpy as np
from PIL import ImageDraw

import torch
import torch.nn as nn
from torch import optim

from .aslloss import AsymmetricLoss


class EarlyStopping(object):
    def __init__(self, patience):
        super(EarlyStopping, self).__init__()
        self.patience = patience
        self.counter = 0
        self.best_score = None
        
    def state_dict(self):
        return {
            'best_score': self.best_score,
            'counter': self.counter
        }
        
    def load_state_dict(self, state_dict):
        self.best_score = state_dict['best_score']
        self.counter = state_dict['counter']

    def __call__(self, score):
        is_save, is_terminate = True, False
        if self.best_score is None:
            self.best_score = score
        elif self.best_score >= score:
            self.counter += 1
            if self.counter >= self.patience:
                is_terminate = True
            is_save = False
        else:
            self.best_score = score
            self.counter = 0
        return is_save, is_terminate
    

class WarmUpLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
    
class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x
    
    
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
def get_trainable_params(model, cfg):
    if cfg.mode == 'full':
        group = model.parameters()
    elif cfg.mode == 'part':
        backbone, others = [], []
        for name, param in model.named_parameters():
            if 'img_encoder' in name:
                backbone.append(param)
            else:
                others.append(param)
        group = [
            {'params': backbone, 'lr': cfg.lr * 0.1},
            {'params': others, 'lr': cfg.lr}
        ]
        print(len(backbone), len(others))
    return group
    
def get_optimizer(params, cfg):
    if cfg.optimizer == 'SGD':
        optimizer = optim.SGD(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'AdamW':
        optimizer = optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    return optimizer

def get_lr_scheduler(optimizer, cfg):
    if cfg.lr_scheduler == 'ReduceLROnPlateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=1, verbose=True)
    elif cfg.lr_scheduler == 'StepLR':
        return optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=0.1)
    # elif cfg.lr_scheduler == 'OneCycleLR':
        # return optim.OneCycleLR(optimizer, max_lr=cfg.lr, steps_per_epoch=steps_per_epoch, epochs=Epochs,
                                        # pct_start=0.2)
    else:
        raise Exception('lr scheduler {} not found!'.format(cfg.lr_scheduler))

def get_loss_fn(cfg):
    if cfg.loss_fn == 'bce':
        return nn.BCEWithLogitsLoss()
    elif cfg.loss_fn == 'asl':
        return AsymmetricLoss(cfg.gamma_neg, cfg.gamma_pos, cfg.clip, disable_torch_grad_focal_loss=True)
    else:
        raise Exception('loss function {} not found!'.format(cfg.loss_fn))

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def get_experiment_id(exp_home):
    exp_names = [t for t in os.listdir(exp_home) if t[-1].isdigit()]
    if len(exp_names) == 0:
        new_exp_id = 1
    else:
        exp_ids = [int(en[3:]) for en in exp_names]
        new_exp_id = max(exp_ids) + 1
    return new_exp_id

def check_makedir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        
def check_exists(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError('file {} not found!'.format(filepath))

def prepare_env(args, argv):
    # prepare data config
    if args.restore_exp is not None: # restore config from the specified experiment
        cfg_path = os.path.join(args.restore_exp, 'config.yaml')
        cfg = yaml.load(open(cfg_path))
        cfg['restore_exp'] = args.restore_exp
    else:
        cfg = vars(args)
        cfg['train_path'] = join('data', args.data, 'train.txt')
        cfg['test_path'] = join('data', args.data, 'test.txt')
        cfg['label_path'] = join('data', args.data, 'label.txt')
        if cfg['embed_type'] == 'bert':
            cfg['embed_path'] = join('data', args.data, 'bert.npy')
        elif cfg['embed_type'] == 'glove':
            cfg['embed_path'] = join('data', args.data, 'glove.npy')
        cfg['ignore_path'] = join('data', args.data, 'ignore.npy')
        check_exists(cfg['train_path'])
        check_exists(cfg['test_path'])
        check_exists(cfg['label_path'])
        cfg['num_classes'] = len(open(cfg['label_path']).readlines())
    
    # prepare checkpoint and log config
    exp_home = join('experiments', '{}_{}'.format(args.model, args.data))
    check_makedir(exp_home)
    exp_name = 'exp{}'.format(get_experiment_id(exp_home))
    exp_dir = join(exp_home, exp_name)
    cfg['exp_dir'] = exp_dir
    cfg['log_path'] = join(exp_dir, 'train.log')
    cfg['ckpt_dir'] = join(exp_dir, 'checkpoints')
    cfg['ckpt_best_path'] = join(cfg['ckpt_dir'], 'best_model.pth')
    cfg['ckpt_latest_path'] = join(cfg['ckpt_dir'], 'latest_model.pth')
    check_makedir(cfg['exp_dir'])
    check_makedir(cfg['ckpt_dir'])
    
    # save experiment checkpoint
    exp_ckpt_path = os.path.join(exp_home, 'checkpoint.txt')
    temp = ' '.join(['python', *argv])
    with open(exp_ckpt_path, 'a') as fa:
        fa.writelines('{}\t{}\n'.format(exp_name, temp))
    
    # save config
    cfg_path = join(cfg['exp_dir'], 'config.yaml')
    with open(cfg_path, 'w') as fw:
        for k, v in cfg.items():
            fw.write('{}: {}\n'.format(k, v))
            
    cfg = argparse.Namespace(**cfg)
    log_path = join(exp_dir, 'train.log')
    prepare_log(log_path, cfg)
            
    return cfg

def prepare_log(log_path, cfg, level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(level)
    sh = logging.StreamHandler()
    th = logging.FileHandler(filename=log_path, encoding='utf-8')
    logger.addHandler(sh)
    logger.addHandler(th)
    
    logger.info('model training time: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    logger.info('model configuration: ')
    format_string = cfg.__class__.__name__ + '(\n'
    for k, v in vars(cfg).items():
        format_string += '    {}: {}\n'.format(k, v)
    format_string += ')'
    logger.info(format_string)
    
def clear_exp(exp_dir):
    logging.shutdown()
    shutil.rmtree(exp_dir)
    exp_home = os.path.dirname(exp_dir)
    exp_ckpt_path = os.path.join(exp_home, 'checkpoint.txt')
    with open(exp_ckpt_path, 'r') as fr:
        temp = fr.readlines()[:-1]
    with open(exp_ckpt_path, 'w') as fw:
        fw.writelines(temp)
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='TSFormer')
    parser.add_argument('--data', type=str, default='voc2007')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'part'])
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--lr_scheduler', type=str, default='ReduceLROnPlateau')
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--start_depth', type=int, default=0)
    
    parser.add_argument('--img_size', type=int, default=448)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--embed_type', type=str, default='bert', choices=['glove', 'bert', 'random', 'onehot'])

    parser.add_argument('--loss_fn', type=str, default='bce')
    parser.add_argument('--gamma_pos', type=float, default=0.0)
    parser.add_argument('--gamma_neg', type=float, default=1.0)
    parser.add_argument('--clip', type=float, default=0.05)
    
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--warmup_epoch', type=int, default=2)
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--pretrained', action='store_false')
    parser.add_argument('--restore_exp', type=str, default=None)
    args = parser.parse_args()
    return args