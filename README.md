## Two-Stream Transformer for Multi-Label Image Classification

### Introduction
This is an official PyTorch implementation of Two-Stream Transformer for Multi-Label Image Classification [paper](https://dl.acm.org/doi/abs/10.1145/3503161.3548343).
![alt tsformer](src/tsformer.png)

### Data Preparation
1. Download dataset and organize them as follow:
```
|datasets
|---- MSCOCO
|---- NUS-WIDE
|---- VOC2007
```
2. Preprocess using following commands:
```bash
python scripts/mscoco.py
python scripts/nuswide.py
python scripts/voc2007.py
python embedding.py --data [mscoco, nuswide, voc2007]
```

### Requirements
```
torch >= 1.9.0
torchvision >= 0.10.0
```

### Training
One can use following commands to train model.
```bash
python train.py --data mscoco --batch_size 16 --optimizer AdamW --lr 0.00001 --mode part --start_depth 9
python train.py --data nuswide --batch_size 16 --optimizer AdamW --lr 0.00001 --mode part --start_depth 1
python train.py --data voc2007 --batch_size 16 --optimizer AdamW --lr 0.00001 --mode part --start_depth 4
```

### Evaluation
Pre-trained weights can be found in [google drive](https://drive.google.com/drive/folders/1XOiLTpWHYRGR8itp4aqQZsbXWHV_TT0j?usp=sharing). Download and put them in the `experiments` folder, then one can use follow commands to reproduce results reported in paper.

```bash
python evaluate.py --exp experiments/TSFormer_mscoco/exp1    # Microsoft COCO
python evaluate.py --exp experiments/TSFormer_nuswide/exp1   # NUS-WIDE
python evaluate.py --exp experiments/TSFormer_voc2007/exp1   # Pascal VOC 2007
```

### Main Results
|  dataaset   | mAP  |
|  ---------  | ---- |
| VOC 2007    | 97.0 |
| MS-COCO     | 88.9 |
| NUS-WIDE    | 69.3 |