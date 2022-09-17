# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2021-9-24
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2021 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
from pycocotools.coco import COCO

coco_dir = 'datasets/MSCOCO'
save_dir = 'data/mscoco'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
def make_data(tag):
    anno_name = 'annotations/instances_{}2014.json'.format(tag)
    coco = COCO(os.path.join(coco_dir, anno_name))

    cat_id2name = {}
    for k, v in coco.cats.items():
        cat_name = v['name']
        cat_id2name[k] = cat_name
    
    data = []
    img_ids = coco.getImgIds()
    for img_id in sorted(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        if len(anns) == 0:
            continue
        categories = set()
        for ann in anns:
            cat_id = ann['category_id']
            cat_name = cat_id2name[cat_id]
            categories.add(cat_name)
        img_name = img_info['file_name']
        img_path = os.path.join(coco_dir, '{}2014'.format(tag), img_name)
        data.append('{}\t{}\n'.format(img_path, ','.join(list(categories))))
    
    if tag == 'train':
        labels = ['{}\n'.format(v) for v in cat_id2name.values()]
        return data, labels
    
    return data

train_data, labels = make_data(tag='train')
with open(os.path.join(save_dir, 'train.txt'), 'w') as fw:
    fw.writelines(train_data)
with open(os.path.join(save_dir, 'label.txt'), 'w') as fw:
    fw.writelines(labels)

val_data = make_data(tag='val')
with open(os.path.join(save_dir, 'test.txt'), 'w') as fw:
    fw.writelines(val_data)
