import torch
import cv2
import numpy as np

from torchmetrics import Metric
from typing import List, Optional


class BaseIoUMetric(torch.nn.Module):
    """
    Computes intersection over union at given thresholds
    """
    def __init__(self, thresholds=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]):
        super().__init__()

        self.thresholds = torch.from_numpy(np.array(thresholds))
        self.tp = torch.zeros_like(self.thresholds)
        self.fp = torch.zeros_like(self.thresholds)
        self.fn = torch.zeros_like(self.thresholds)

    def update(self, pred, label, isLogit=True):

        if (isLogit): pred = pred.detach().to('cpu').sigmoid().reshape(-1)
        else: pred = pred.detach().to('cpu').reshape(-1)
        label = label.detach().to('cpu').bool().reshape(-1)

        pred = pred[:, None] >= self.thresholds[None]
        label = label[:, None]

        self.tp += (pred & label).sum(0)
        self.fp += (pred & ~label).sum(0)
        self.fn += (~pred & label).sum(0)

    def compute(self):
        thresholds = self.thresholds.squeeze(0)
        ious = self.tp / (self.tp + self.fp + self.fn + 1e-7)

        output = {}
        for t, i in zip(thresholds, ious):
            output.update({f'@{t.item():.2f}': i.item()})

        return {f'@{t.item():.2f}': i.item() for t, i in zip(thresholds, ious)}


class IoUMetric(BaseIoUMetric):
    def __init__(self, label_indices: List[List[int]],
                 min_visibility: Optional[int] = None,
                 target_class: Optional[str] = None): # update 231006
        """
        label_indices:
            transforms labels (c, h, w) to (len(labels), h, w)
            see config/experiment/* for examples

        min_visibility:
            passing "None" will ignore the visibility mask
            otherwise uses visibility values to ignore certain labels
            visibility mask is in order of "increasingly visible" {1, 2, 3, 4, 255 (default)}
            see https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md#visibility
        """
        super().__init__()

        self.label_indices = label_indices
        self.min_visibility = min_visibility
        self.target_class = target_class

    def update(self, pred, batch):

        label = batch['bev']                                                                # b n h w
        label = [label[:, idx].max(1, keepdim=True).values for idx in self.label_indices]
        label = torch.cat(label, 1)                                                         # b c h w

        visibility = None
        if (self.target_class == 'vehicle'): visibility = batch['visibility'][:, [0]]
        elif (self.target_class == 'pedestrian'): visibility = batch['visibility'][:, [1]]

        if self.min_visibility is not None:
            mask = visibility >= self.min_visibility
            mask = mask.expand_as(pred)                                            # b c h w

            pred = pred[mask]                                                               # m
            label = label[mask]                                                             # m

        return super().update(pred, label)
