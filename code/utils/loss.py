import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score
from IPython import embed


def binary_crossentropy(y_pred, y_true, k=4):
    bcy = F.binary_cross_entropy(y_pred, y_true)
    loss = bcy * (1 + y_true * (k - 1)) * (k + 1) / (2 * k)
    losses = torch.mean(loss)
    return losses


class weighted_binary_crossentropy(nn.Module):
    def __init__(self, weight):
        super(weighted_binary_crossentropy, self).__init__()
        self.weight = weight

    def forward(self, y_pred, y_true, k=4):
        bcy = F.binary_cross_entropy(y_pred, y_true, reduction='none')
        bcy *= self.weight
        bcy = torch.mean(bcy)
        loss = bcy * (1 + y_true * (k - 1)) * (k + 1) / (2 * k)
        losses = torch.mean(loss)
        return losses


def calc_f1(y_true, y_pre, threshold=0.5):
    y_true = y_true.view(-1).cpu().detach().numpy().astype(np.int)
    y_pre = y_pre.view(-1).cpu().detach().numpy() > threshold
    return f1_score(y_true, y_pre)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
