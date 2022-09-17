# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2021-8-9
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2021 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import yaml
import time
import argparse
from argparse import Namespace
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchstat import stat
from torchsummary import summary

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
        # for param in self.model.img_encoder.parameters():
        #     param.requires_grad = False
        # summary(self.model, input_size=(3, 448, 448), batch_size=-1)

        self.cfg = cfg
        self.voc07_mAP = VOCmAP(cfg.num_classes, year='2007', ignore_path=cfg.ignore_path)
        self.voc12_mAP = VOCmAP(cfg.num_classes, year='2007', ignore_path=cfg.ignore_path)
        self.topk_meter = TopkMeter(cfg.num_classes, cfg.ignore_path, topk=cfg.topk)
        self.threshold_meter = ThresholdMeter(cfg.num_classes, cfg.ignore_path, threshold=cfg.threshold)

    @torch.no_grad()
    def run(self):
        model_dict = torch.load(self.cfg.ckpt_best_path)
        self.model.load_state_dict(model_dict, strict=True)
        print(f'loading best checkpoint success')
        
        fw = open(os.path.join(self.cfg.exp_dir, 'prediction.txt'), 'w')
        self.model.eval()
        self.voc07_mAP.reset()
        self.voc12_mAP.reset()
        self.topk_meter.reset()
        self.threshold_meter.reset()
        
        start_time = time.time()
        cost = 0.0
        for batch in tqdm(self.dataloader):
            img = batch['img'].cuda()
            targets = batch['target'].numpy()
            img_path = batch['img_path']
            start = time.time()
            logits, _ = self.model(img)
            end = time.time()
            cost += end - start

            scores = torch.sigmoid(logits).cpu().numpy()
            self.voc07_mAP.update(scores, targets)
            self.voc12_mAP.update(scores, targets)
            self.topk_meter.update(scores, targets)
            self.threshold_meter.update(scores, targets)
            
            topk_inds = np.argsort(-scores)[:, :self.cfg.topk]
            for j in range(img.size(0)):
                img_name = os.path.basename(img_path[j])
                pred_labels = [self.labels[ind] for ind in topk_inds[j]]
                fw.write('{}\t{}\n'.format(img_name, ' '.join(pred_labels)))
                
        fw.close()
        end_time = time.time()
        fps = len(self.dataloader) / (end_time - start_time)
        print('fps: {}'.format(fps))
        print('fps2: {}'.format(len(self.dataloader)/cost))

        aps_07, mAP_07 = self.voc07_mAP.compute()
        aps_12, mAP_12 = self.voc12_mAP.compute()
        self.topk_meter.compute()
        self.threshold_meter.compute()

        ret = {
            'mAP': mAP_07,
            'topk_cp': self.topk_meter.cp,
            'topk_cr': self.topk_meter.cr,
            'topk_cf1': self.topk_meter.cf1,
            'topk_op': self.topk_meter.op,
            'topk_or': self.topk_meter.or_,
            'topk_of1': self.topk_meter.of1,
            'threshold_cp': self.threshold_meter.cp,
            'threshold_cr': self.threshold_meter.cr,
            'threshold_cf1': self.threshold_meter.cf1,
            'threshold_op': self.threshold_meter.op,
            'threshold_or': self.threshold_meter.or_,
            'threshold_of1': self.threshold_meter.of1,
        }
        ret = ['{:<15}\t{:.3f}\n'.format(k, v) for k, v in ret.items()]
        ret.append('\n')
        for label, ap in zip(self.labels, aps_07):
            ret.append('{:<15}\t{:.3f}\n'.format(label, ap))
        with open(os.path.join(self.cfg.exp_dir, 'result.txt'), 'w') as fw:
            fw.writelines(ret)
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