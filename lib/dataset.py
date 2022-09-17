# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2021-9-24
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2021 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import logging
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from randaugment import RandAugment
from lib.util import CutoutPIL


class MLDataset(Dataset):
    def __init__(self, data_path, label_path, img_size, is_train=True, transform=None):
        super(MLDataset, self).__init__()

        self.labels = [line.strip() for line in open(label_path)]
        self.num_classes = len(self.labels)
        self.label2id = {label:i for i, label in enumerate(self.labels)}

        self.data = []
        with open(data_path, 'r') as fr:
            for line in fr.readlines():
                image_path, image_label = line.strip().split('\t')
                image_label = [self.label2id[l] for l in image_label.split(',')]
                self.data.append([image_path, image_label])
        
        if transform is None:
            self.transform = self.get_transform(img_size, is_train)
        else:
            self.transform = transform
            
        logging.info(self.transform)
    
    def get_transform(self, img_size, is_train):
        t = []
        if is_train:
            t.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                # CutoutPIL(cutout_factor=0.5),
                RandAugment()
            ])
        
        t.extend([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
        ])

        return transforms.Compose(t)
    
    def __getitem__(self, index):
        img_path, img_label = self.data[index]
        img_data = Image.open(img_path).convert('RGB')
        img_data = self.transform(img_data)

        # one-hot encoding for label
        target = np.zeros(self.num_classes).astype(np.float32)
        target[img_label] = 1.0
        
        pos_target = np.arange(196)
        
        item = {
            'img': img_data,
            'target': target,
            'img_path': img_path,
            'pos_target': pos_target
        }
        
        return item
        
    def __len__(self):
        return len(self.data)
    