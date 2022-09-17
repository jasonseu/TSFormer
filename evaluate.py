# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2021-8-9
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2021 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import yaml
import argparse
from argparse import Namespace
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from lib.metrics import *
from lib.dataset import MLDataset
from models.factory import create_model

torch.backends.cudnn.benchmark = True


class Evaluator(object):
    def __init__(self, cfg):
        super(Evaluator, self).__init__()
        dataset = MLDataset(cfg.test_path, cfg.label_path, cfg.img_size, is_train=False)
        self.dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4)
        self.labels = dataset.labels
        
        self.model = create_model(cfg.model, pretrained=False, cfg=cfg)
        self.model.cuda()

        self.cfg = cfg
        self.voc07_mAP = VOCmAP(cfg.num_classes, year='2007', ignore_path=cfg.ignore_path)
        self.voc12_mAP = VOCmAP(cfg.num_classes, year='2007', ignore_path=cfg.ignore_path)

    @torch.no_grad()
    def run(self):
        model_dict = torch.load(self.cfg.ckpt_best_path)
        self.model.load_state_dict(model_dict, strict=True)
        print(f'loading best checkpoint success')
        
        self.model.eval()
        self.voc07_mAP.reset()
        self.voc12_mAP.reset()
        
        for batch in tqdm(self.dataloader):
            img = batch['img'].cuda()
            targets = batch['target'].numpy()
            logits, _ = self.model(img)

            scores = torch.sigmoid(logits).cpu().numpy()
            self.voc07_mAP.update(scores, targets)
            self.voc12_mAP.update(scores, targets)
            
        _, mAP_07 = self.voc07_mAP.compute()
        _, mAP_12 = self.voc12_mAP.compute()
        print('model {} data {} mAP_07: {:.4f} mAP_12: {:.4f}'.format(self.cfg.model, self.cfg.data, mAP_07, mAP_12))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='experiments/TSFormer_voc2007/exp6')
    args = parser.parse_args()
    cfg_path = os.path.join(args.exp, 'config.yaml')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError('config file not found in the {}!'.format(cfg_path))
    cfg = yaml.load(open(cfg_path, 'r'))
    cfg = Namespace(**cfg)
    print(cfg)
    
    evaluator = Evaluator(cfg)
    evaluator.run()