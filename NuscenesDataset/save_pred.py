import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import get_cmap


class BaseSave:

    def __init__(self, label_indices, SEMANTICS, threshold=0.5, visibility=2):
        self.label_indices = label_indices
        self.SEMANTICS = SEMANTICS
        self.threshold = threshold
        self.visibility = visibility

    def return_gt(self, batch, target):
        '''
        gt_bev : b h w
        '''

        label_indices = self.label_indices[target][0]

        bev = batch['bev'][-1] # b ch h w

        if (target == 'vehicle'):
            visibility = batch['visibility'][-1, :, 0]  # b h w
        elif (target == 'pedestrian'):
            visibility = batch['visibility'][-1, :, 1]  # b h w
        else:
            visibility = None

        # b ch h w -> b h w
        gt_bev = bev[:, label_indices].max(1)[0].numpy()

        if (visibility is not None):
            disable = visibility < self.visibility
            gt_bev[disable] = 0

        return gt_bev

    def return_pred(self, pred, target):
        '''
        pred_bev : b h w
        '''

        pred_logit = pred[target][0][:, 0].sigmoid() # b h w
        pred_logit = pred_logit.detach().to('cpu').numpy()
        chk1 = pred_logit >= self.threshold

        pred_bev = np.zeros_like(pred_logit)
        pred_bev[chk1] = 1

        return pred_bev


    def return_cams(self, batch):

        image = batch['image'][0].permute(0, 1, 3, 4, 2).numpy() # b n_cams h w ch

        return (255.0 * image).astype('uint8')
