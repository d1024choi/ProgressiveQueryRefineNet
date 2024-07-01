import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from fvcore.nn import sigmoid_focal_loss
import sys

class SigmoidFocalLoss(torch.nn.Module):
    def __init__(self, alpha=-1.0, gamma=2.0, reduction='mean'):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, label):
        return sigmoid_focal_loss(pred, label, self.alpha, self.gamma, self.reduction)

class LossScratch(torch.nn.Module):
    def __init__(self, cfg, min_visibility=2):
        super().__init__()

        self.cfg = cfg
        self.target = cfg['target']
        self.label_indices = cfg['label_indices']
        self.min_visibility = min_visibility

        self.bce = SigmoidFocalLoss(alpha=cfg['bce']['alpha'], gamma=cfg['bce']['gamma'], reduction='none')
        self.focal = SigmoidFocalLoss(alpha=cfg['focal']['alpha'], gamma=cfg['focal']['gamma'], reduction='none')
        self.l1 = nn.L1Loss(size_average=None, reduce=None, reduction='none')

    def bce_loss(self, pred, label, label_indices=None, visibility=None, ignore_label_indices=False):
        '''
        pred : b x 1 x h x w
        label : b x 1 x h x w
        visibility : b x 1 x h x w
        '''

        if ignore_label_indices is False:
            if label_indices is not None:
                label = [label[:, idx].max(dim=1, keepdim=True).values for idx in label_indices]
                label = torch.cat(label, dim=1)

        _, _, hp, wp = pred.size()
        _, _, hl, wl = label.size()
        if (hp < hl):
            scale = hp / hl
            label = F.interpolate(label, scale_factor=scale, mode='nearest')
            if (visibility is not None):
                visibility = F.interpolate(visibility.to(pred), scale_factor=scale, mode='nearest')

        loss = self.bce(pred, label)

        if self.min_visibility is not None and visibility is not None:
            mask = visibility >= self.min_visibility
            loss = loss[mask]

        return loss

    def focal_loss(self, pred, label, label_indices=None, visibility=None, ignore_label_indices=False):
        '''
        pred : b x 1 x h x w
        label : b x 1 x h x w
        visibility : b x 1 x h x w
        '''

        if ignore_label_indices is False:
            if label_indices is not None:
                label = [label[:, idx].max(dim=1, keepdim=True).values for idx in label_indices]
                label = torch.cat(label, dim=1)

        _, _, hp, wp = pred.size()
        _, _, hl, wl = label.size()
        if (hp < hl):
            scale = hp / hl
            label = F.interpolate(label, scale_factor=scale, mode='nearest')
            if (visibility is not None):
                visibility = F.interpolate(visibility.to(pred), scale_factor=scale, mode='nearest')

        loss = self.focal(pred, label)

        if self.min_visibility is not None and visibility is not None:
            mask = visibility >= self.min_visibility
            loss = loss[mask]

        return loss

    def l1_loss(self, pred, label, visibility=None, instance=None):
        '''
        pred : b x 2 x h x w
        label : b x 2 x h x w
        visibility : b x 1 x h x w
        instance : b x 1 x h x w
        '''

        if (pred is None):
            return torch.zeros(1).to(label)

        loss = self.l1(pred, label)
        if (visibility is not None and instance is not None):
            mask_visibility = visibility >= self.min_visibility
            mask_instance = instance > 0
            mask = torch.logical_and(mask_visibility, mask_instance)
            loss = loss[mask.repeat(1, pred.size(1), 1, 1)]

        return loss

    def main(self, pred, batch):
        '''
        ** pred
            pred['vehicle'][0] : b 1 h w
            pred['vehicle'][1] : b 1 h w
            pred['road'][0] : b 1 h w
            pred['lane'][0] : b 1 h w
            pred['intp'] :  a list of tensors of shape 'b 1 h w'

        ** batch
            batch['bev'] : b 12 h w
            batch['center'] : b 1 h w
            batch['visibility'] : b 1 h w

        ** batch['bev']
            load : 0, 1
            lane : 2, 3
            vehicle : 4, 5, 6, 7, 8, 10, 11
            pedestrian : 9
        '''

        # labels
        bev_gt = batch['bev'].to(pred[self.target[0]][0]) # b 12 h w

        center_gt = {'vehicle': batch['center'][:, [0]].to(pred[self.target[0]][0]),
                     'pedestrian': batch['center'][:, [1]].to(pred[self.target[0]][0])}

        visibility = {'vehicle': batch['visibility'][:, [0]],
                      'pedestrian': batch['visibility'][:, [1]]}

        # losses
        losses = {}
        for _, target in enumerate(self.target):
            if (target == 'road'):
                bce = self.bce_loss(pred[target][0], bev_gt, self.label_indices[target])
                focal = self.focal_loss(pred[target][0], bev_gt, self.label_indices[target])
                losses.update({target: {'loss': bce.mean() + focal.mean(), 'weight': 1.0}})

            elif (target == 'lane'):
                focal = self.focal_loss(pred[target][0], bev_gt, self.label_indices[target])
                losses.update({target: {'loss': focal.mean(), 'weight': 1.0}})

            elif (target == 'vehicle' or target == 'pedestrian'):
                bev = self.focal_loss(pred[target][0], bev_gt, self.label_indices[target], visibility[target])
                center = self.focal_loss(pred[target][1], center_gt[target], label_indices=None, visibility=visibility[target],
                                         ignore_label_indices=True)
                losses.update({target: {'loss': bev.mean() + center.mean(), 'weight': 1.0}})

            else:
                sys.exit(f'>> {target} is not supported for loss calculation!!')

        return losses

    def intermediate(self, pred, batch):
        '''
        ** pred
            pred['vehicle'][0] : b 1 h w
            pred['vehicle'][1] : b 1 h w
            pred['road'][0] : b 1 h w
            pred['lane'][0] : b 1 h w
            pred['intp'] :  a list of tensors of shape 'b 1 h w'

        ** batch
            batch['bev'] : b 12 h w
            batch['center'] : b 1 h w
            batch['visibility'] : b 1 h w

        ** batch['bev']
            load : 0, 1
            lane : 2, 3
            vehicle : 4, 5, 6, 7, 8, 10, 11
            pedestrian : 9
        '''

        tdix = {}
        for i, key in enumerate(self.target):
            tdix[key] = [i, i+1]

        # labels
        bev_gt = batch['bev'].to(pred[self.target[0]][0]) # b 12 h w
        visibility = {'vehicle': batch['visibility'][:, [0]],
                      'pedestrian': batch['visibility'][:, [1]]}

        # losses
        losses = {}
        intm_logits = pred['intm']
        for _, target in enumerate(self.target):
            if (target == 'road'):
                loss = torch.zeros(1).to(bev_gt)
                for intp_logit in intm_logits:
                    loss += self.focal_loss(intp_logit[:, tdix[target][0]:tdix[target][1]],
                                           bev_gt, self.label_indices[target]).mean()
                losses.update({target: {'loss': loss, 'weight': 1.0}})

            elif (target == 'lane'):
                loss = torch.zeros(1).to(bev_gt)
                for intp_logit in intm_logits:
                    loss += self.focal_loss(intp_logit[:, tdix[target][0]:tdix[target][1]],
                                           bev_gt, self.label_indices[target]).mean()
                losses.update({target: {'loss': loss, 'weight': 1.0}})

            elif (target == 'vehicle' or target == 'pedestrian'):
                loss = torch.zeros(1).to(bev_gt)
                for intp_logit in intm_logits:
                    loss += self.focal_loss(intp_logit[:, tdix[target][0]:tdix[target][1]],
                                           bev_gt, self.label_indices[target], visibility[target]).mean()
                losses.update({target: {'loss': loss, 'weight': 1.0}})
            else:
                sys.exit(f'>> {target} is not supported for loss calculation!!')

        return losses

    def offset(self, pred, batch):
        '''
        ** pred
        pred['vehicle'][0] : b 1 h w
        pred['vehicle'][1] : b 1 h w
        pred['road'][0] : b 1 h w
        pred['lane'][0] : b 1 h w
        pred['intp'] :  a list of tensors of shape 'b 1 h w'

        ** batch
        batch['bev'] : b 12 h w
        batch['center'] : b 1 h w
        batch['visibility'] : b 1 h w
        batch['offsets'] : b 2 h w
        '''

        # labels
        bev_gt = {}
        bev_gt['vehicle'] = batch['offsets'][:, :2].to(pred[self.target[0]][0]) # b 2 h w
        bev_gt['pedestrian'] = batch['offsets'][:, 2:].to(pred[self.target[0]][0]) # b 2 h w

        visibility = {'vehicle': None, 'pedestrian': None}
        instance = {'vehicle': None, 'pedestrian': None}

        if (self.cfg['bool_use_vis_offset']):
            visibility['vehicle'] = batch['visibility'][:, [0]]
            visibility['pedestrian'] = batch['visibility'][:, [1]]
            instance['vehicle'] = batch['instance'][:, [0]]
            instance['pedestrian'] = batch['instance'][:, [1]]

        # losses
        losses = {}
        for _, target in enumerate(self.target):
            if (target == 'vehicle'):
                loss = self.l1_loss(pred['offsets'][target], bev_gt[target], visibility[target], instance[target]).mean()
                losses.update({target: {'loss': loss, 'weight': 1.0}})
            elif (target == 'pedestrian'):
                loss = self.l1_loss(pred['offsets'][target], bev_gt[target], visibility[target], instance[target]).mean()
                losses.update({target: {'loss': loss, 'weight': 1.0}})

        return losses