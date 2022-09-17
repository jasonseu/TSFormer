# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2021-9-24
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2021 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import abc
import numpy as np


class AverageLoss(object):
    def __init__(self):
        super(AverageLoss, self).__init__()

    def reset(self):
        self._sum = 0
        self._counter = 0

    def update(self, loss, n=0):
        self._sum += loss * n
        self._counter += n

    def compute(self):
        return self._sum / self._counter


class VOCmAP(object):
    def __init__(self, num_classes, year='2007', ignore_path=None):
        """Calculate the mean average precision over all labels.
        
        Args:
            num_classes (int): the number of the classes.
            year (str): Pascal VOC provide two kinds of algorithm for computing average 
                        precision in 2007 and 2012, respectively.
            ignore_path (str): a numpy array path for loading the matrix in which each entity
                        x_ij = 1 denotes that the sample i should be ignored when computing the AP
                        for the class j.
        """
        
        super().__init__()
        self.num_classes = num_classes
        if year == '2007':
            self._voc_ap = self._voc07_ap
        elif year == '2012':
            self._voc_ap = self._voc12_ap
        else:
            raise Exception('year should be 2007 or 2012!')
        self.ignore = np.load(ignore_path) if os.path.exists(ignore_path) else None

    def reset(self):
        self.scores = np.array([], dtype=np.float64).reshape(0, self.num_classes)
        self.targets = np.array([], dtype=np.float64).reshape(0, self.num_classes)

    def update(self, scores, targets):
        """
        Args:
            scores (numpy.array, float64): the predicted confidences for all classes of a batch of samples.
            targets (numpy.array, int64): the ground truth of a batch of samples.
        """
        self.scores = np.vstack((self.scores, scores))
        self.targets = np.vstack((self.targets, targets))

    def compute(self, scores=None, targets=None):
        if scores is not None and targets is not None:
            self.scores = scores
            self.targets = targets
        return self._voc_mAP()
    
    def _voc_mAP(self):
        aps = np.zeros(self.num_classes, dtype=np.float64)
        for i in range(self.num_classes):
            target = self.targets[:, i]
            pred = self.scores[:, i]
            if self.ignore is not None:
                ignore = self.ignore[:, i]
                t = np.where(ignore != 1)[0]
                target = target[t]
                pred = pred[t]
            t = np.argsort(pred)[::-1]
            target = target[t]
            aps[i] = self._voc_ap(target)
        mAP = np.mean(aps)
        return aps, mAP
    
    def _voc07_ap(self, target):
        """average precision with Psacal VOC 2007 algorithm"""
        num_samples = len(target)
        pre, obj = 0, 0
        for j in range(num_samples):
            if target[j] == 1:
                obj += 1.0
                pre += obj / (j + 1)
        ap = pre / obj
        return ap
        
    def _voc12_ap(self, target):
        """average precision with Pascal VOC 2012 algorithm"""
        sample_num = len(target)
        tp = (target == 1).astype(np.int64)   # true positive
        fp = (target == 0).astype(np.int64)   # false positive
        tp_num = max(sum(tp), np.finfo(np.float64).eps)
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        recall = tp / float(tp_num)
        precision = tp / np.arange(1, sample_num+1, dtype=np.float64)

        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap


class Meter(object):
    def __init__(self, num_classes, ignore_path):
        super().__init__()
        self.num_classes = num_classes
        self.ignore = np.load(ignore_path) if os.path.exists(ignore_path) else None

    def reset(self):
        self._tp_num = np.zeros(self.num_classes)  # true positive predicted image per-class counter
        self._pd_num = np.zeros(self.num_classes)    # predicted image per-class counter
        self._gt_num = np.zeros(self.num_classes)  # ground-truth image per-class counter
        self.scores = np.array([], dtype=np.float64).reshape(0, self.num_classes)
        self.targets = np.array([], dtype=np.float64).reshape(0, self.num_classes)

    def update(self, scores, targets):
        self.scores = np.vstack((self.scores, scores))
        self.targets = np.vstack((self.targets, targets))

    def compute(self, eps=np.finfo(np.float64).eps):
        # self.targets[self.targets==-1] = 0
        self._count()
        self._pd_num[self._pd_num == 0] = 1
        self._op = np.sum(self._tp_num) / np.sum(self._pd_num)
        self._or = np.sum(self._tp_num) / np.sum(self._gt_num)
        self._of1 = 2 * self._op * self._or / (self._op + self._or)
        # self._tp_num = np.maximum(self._tp_num, eps)
        # self._pd_num = np.maximum(self._pd_num, eps)
        # self._gt_num = np.maximum(self._gt_num, eps)
        self._cp = np.mean(self._tp_num / self._pd_num)
        self._cr = np.mean(self._tp_num / self._gt_num)
        self._cf1 = 2 * self._cp * self._cr / (self._cp + self._cr)

    @abc.abstractmethod
    def _count(self):
        pass

    @property
    def op(self):   # overall precision
        return self._op

    @property   # overall recall
    def or_(self):
        return self._or

    @property   # overall F1
    def of1(self):
        return self._of1

    @property   # per-class precision
    def cp(self):
        return self._cp

    @property   # per-class recall
    def cr(self):
        return self._cr

    @property   # per-class F1
    def cf1(self):
        return self._cf1


class TopkMeter(Meter):
    def __init__(self, num_classes, ignore_path=None, topk=3):
        super().__init__(num_classes, ignore_path)
        self.topk = topk
    
    def _count(self):
        num_samples = self.scores.shape[0]
        for i in range(num_samples):
            score = self.scores[i]
            ind = np.argsort(-score)[self.topk:]
            score[ind] = 0.0
            self.scores[i] = score
        
        for i in range(self.num_classes):
            score = self.scores[:, i]
            target = self.targets[:, i]
            if self.ignore is not None:
                ignore = self.ignore[:, i]
                t = np.where(ignore != 1)[0]
                score = score[t]
                target = target[t]
            self._gt_num[i] = np.sum(target)
            self._pd_num[i] = np.sum(score >= 0.5)
            self._tp_num[i] = np.sum(target * (score >= 0.5))
            

class ThresholdMeter(Meter):
    def __init__(self, num_classes, ignore_path=None, threshold=0.5):
        super().__init__(num_classes, ignore_path)
        self.threshold = threshold
            
    def _count(self):
        for i in range(self.num_classes):
            score = self.scores[:, i]
            target = self.targets[:, i]
            if self.ignore is not None:
                ignore = self.ignore[:, i]
                t = np.where(ignore != 1)[0]
                score = score[t]
                target = target[t]
            self._gt_num[i] = np.sum(target == 1)
            self._pd_num[i] = np.sum(score >= self.threshold)
            self._tp_num[i] = np.sum(target * (score >= self.threshold))