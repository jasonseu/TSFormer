# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2021-8-17
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2021 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
from collections import defaultdict
from tqdm import tqdm


data_dir = 'datasets/NUS-WIDE'
save_dir = 'data/nuswide'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

img_path = os.path.join(data_dir, 'ImageList/Imagelist.txt')
train_img_path = os.path.join(data_dir, 'ImageList/TrainImagelist.txt')
test_img_path = os.path.join(data_dir, 'ImageList/TestImagelist.txt')
groundtruth_dir = os.path.join(data_dir, 'Groundtruth/AllLabels')
label_path = os.path.join(data_dir, 'Concepts81.txt')

img_list = [t.strip().replace('\\', '/') for t in open(img_path)]
train_img_list = [t.strip().replace('\\', '/') for t in open(train_img_path)]
test_img_list = [t.strip().replace('\\', '/') for t in open(test_img_path)]
label_list = [t.strip() for t in open(label_path)]

img_split_flag = {img_name:0 for img_name in train_img_list}
img_split_flag.update({img_name:1 for img_name in test_img_list})

print('all images number: ', len(img_list))
print('train images number: ', len(train_img_list))
print('test images number: ', len(test_img_list))

label_mat = []
for label in label_list:
    filename = 'Labels_{}.txt'.format(label)
    filepath = os.path.join(groundtruth_dir, filename)
    labels = [t.strip() for t in open(filepath)]
    label_mat.append(labels)
    
label_mat = list(map(list, zip(*label_mat))) # 269648 x 81

# split data according to official method
train_data, test_data = [], []
for img_name, img_labels in tqdm(zip(img_list, label_mat)):
    img_path = os.path.join(data_dir, 'Flickr', img_name)
    if not os.path.exists(img_path):
        raise FileNotFoundError('file {} not found!'.format(img_path))
    labels = [label_list[i] for i, t in enumerate(img_labels) if t == '1']
    if len(labels) == 0:
        continue
    flag = img_split_flag.get(img_name)
    if  flag == 0:
        train_data.append('{}\t{}\n'.format(img_path, ','.join(labels)))
    elif flag == 1:
        test_data.append('{}\t{}\n'.format(img_path, ','.join(labels)))

with open(os.path.join(save_dir, 'train.txt'), 'w') as fw:
    fw.writelines(train_data)
with open(os.path.join(save_dir, 'test.txt'), 'w') as fw:
    fw.writelines(test_data)
with open(os.path.join(save_dir, 'label.txt'), 'w') as fw:
    fw.writelines(['{}\n'.format(t) for t in label_list])

print('all images number: ', len(train_data) + len(test_data))
print('train images number: ', len(train_data))
print('test images number: ', len(test_data))