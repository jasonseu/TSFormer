# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2021-8-9
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2021 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import sys
import time
import logging
import traceback

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lib.util import *
from lib.metrics import *
from lib.dataset import MLDataset
from models.factory import create_model


logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, cfg):
        super(Trainer, self).__init__()
        train_dataset = MLDataset(cfg.train_path, cfg.label_path, cfg.img_size, is_train=True)
        self.train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
        val_dataset = MLDataset(cfg.test_path, cfg.label_path, cfg.img_size, is_train=False)
        self.val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4)

        self.model = create_model(cfg.model, pretrained=True, cfg=cfg)
        self.model.cuda()
        
        group = get_trainable_params(self.model, cfg)
        self.optimizer = get_optimizer(group, cfg)
        self.lr_scheduler = get_lr_scheduler(self.optimizer, cfg)
        self.warmup_scheduler = WarmUpLR(self.optimizer, len(self.train_loader)*cfg.warmup_epoch)
        self.criterion = get_loss_fn(cfg)
        
        self.early_stopping = EarlyStopping(patience=4)
        self.voc07_mAP = VOCmAP(cfg.num_classes, year='2007', ignore_path=cfg.ignore_path)

        self.cfg = cfg
        self.global_step = 0
        self.writer = SummaryWriter(log_dir=cfg.exp_dir)
        logger.info('total parameters: {}'.format(sum(p.numel() for p in self.model.parameters())))

    def run(self):
        for epoch in range(self.cfg.max_epoch):
            self.train(epoch)
            mAP = self.validation(epoch)
            self.lr_scheduler.step(mAP)
            is_save, is_terminate = self.early_stopping(mAP)
            if is_terminate:
                break
            if is_save:
                torch.save(self.model.state_dict(), self.cfg.ckpt_best_path)
                
        logger.info('\ntraining over, best validation score: {} mAP'.format(self.early_stopping.best_score))

    def train(self, epoch):
        self.model.train()
        for batch in self.train_loader:
            batch_begin = time.time()
            imgs = batch['img'].cuda()
            targets = batch['target'].cuda()
            logits = self.model(imgs)
            loss = self.criterion(logits, targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            dur = time.time() - batch_begin

            if self.global_step % (len(self.train_loader) // 6) == 0:
                lr = get_lr(self.optimizer)
                self.writer.add_scalar('Loss/train', loss, self.global_step)
                self.writer.add_scalar('lr', lr, self.global_step)
                logger.info('TRAIN [epoch {}] loss: {:4f} lr:{:.5f} time:{:.4f}'.format(epoch, loss, lr, dur))
            if epoch < self.cfg.warmup_epoch:
                self.warmup_scheduler.step()
            
            self.global_step += 1

    @torch.no_grad()
    def validation(self, epoch):
        self.model.eval()
        self.voc07_mAP.reset()
        for batch in self.val_loader:
            imgs = batch['img'].cuda()
            targets = batch['target'].cuda()
            logits, _ = self.model(imgs)
            targets = targets.cpu().numpy()
            scores = torch.sigmoid(logits).detach().cpu().numpy()
            self.voc07_mAP.update(scores, targets)
            
        _, mAP = self.voc07_mAP.compute()
        self.writer.add_scalar('mAP/val', mAP, self.global_step)

        logger.info("Validation [epoch {}] mAP: {:.4f}".format(epoch, mAP))
        return mAP


if __name__ == "__main__":
    args = get_args()
    cfg = prepare_env(args, sys.argv)
    setup_seed(cfg.seed)
    
    try:
        trainer = Trainer(cfg)
        trainer.run()
    except (Exception, KeyboardInterrupt):
        print(traceback.format_exc())
        if not os.path.exists(cfg.ckpt_latest_path):
            clear_exp(cfg.exp_dir)