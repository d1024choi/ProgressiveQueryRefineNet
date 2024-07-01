import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
import math

class CrossTaskHead(torch.nn.Module):
    '''
    Xu et al. "PAD-Net: Multi-Tasks Guided Prediction-and-Distillation Network for
    Simultaneous Depth Estimation and Scene Parsing," CVPR 2018.
    '''

    def __init__(self, dim=256, targets=['vehicle']):
        super().__init__()


        self.dim = dim
        self.targets = targets
        self.net = nn.ModuleDict()
        for key in targets:
            self.net[key] = nn.ModuleDict()
            self.net[key]['scale'] = nn.Sequential(*[nn.Conv2d(dim, 1, 3, padding=1, bias=False), nn.Sigmoid()])
            self.net[key]['proj'] = nn.Conv2d(dim, dim, 3, padding=1, bias=False)


    def forward(self, projections):

        if (len(self.targets) == 1):
            return projections

        h = w = int(math.sqrt(projections[self.targets[0]].size(1)))

        output = dict()
        for target in self.targets:
            tar_bev = projections[target]
            tar_bev = rearrange(tar_bev, 'b (h w) c -> b c h w', h=h, w=w)

            for key, ngh_bev in projections.items():
                ngh_bev = rearrange(ngh_bev, 'b (h w) c -> b c h w', h=h, w=w)
                if (key is not target):
                    scale = self.net[key]['scale'](tar_bev).repeat(1, self.dim, 1, 1)
                    tar_bev = tar_bev + scale * self.net[key]['proj'](ngh_bev)
            output[target] = rearrange(tar_bev, 'b c h w -> b (h w) c')

        return output


class SimpleUpsampler(torch.nn.Module):
    def __init__(self, mode='bilinear'):
        super().__init__()

        if (str(mode) not in ['nearest']): self.up = nn.Upsample(scale_factor=2, mode=mode, align_corners=True)
        else: self.up = nn.Upsample(scale_factor=2, mode=mode)

    def forward(self, x):
        return self.up(x)

class DecoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, residual=True, factor=2):
        super().__init__()

        dim = out_channels // factor

        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, out_channels, 1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels))

        if residual:
            self.up = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.up = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, identity):
        x = self.conv(x)

        if self.up is not None:
            up = self.up(identity)
            up = F.interpolate(up, x.shape[-2:])
            x = x + up

        return self.relu(x)

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_iter, out_logit_dim, residual=True, factor=2):
        super().__init__()

        layers = list()
        for _ in range(num_iter):
            layer = DecoderBlock(in_channels, out_channels, residual, factor)
            layers.append(layer)
        self.layers = nn.Sequential(*layers)
        self.conv = nn.Conv2d(in_channels, out_logit_dim, kernel_size=1)


    def forward(self, x):
        identity = x
        for layer in self.layers:
            x = layer(x, identity)
            identity = x
        return self.conv(x)

class MaskDecoderBlock(torch.nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1, qkv_bias=False):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.dropout0 = nn.Dropout(dropout)
        self.norm0 = nn.LayerNorm(dim)

        # ffn
        self.linear1 = nn.Linear(dim, dim)
        self.activation = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim, dim)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, query, feats):
        """
        query : b num_classes dim
        feats (=BEVfeat) : b (h w) dim
        """

        b, q_len, dim = query.size()
        h = w = int(np.sqrt(feats.size(1)))
        q = self.q(query.view(b*q_len, dim)).view(b, q_len, self.num_heads, self.head_dim)
        k = self.k(feats.view(-1, dim)).view(b, -1, self.num_heads, self.head_dim)
        v = self.v(feats.view(-1, dim)).view(b, -1, self.num_heads, self.head_dim)

        q = rearrange(q, 'b l h d -> (b h) l d') * self.scale
        k = rearrange(k, 'b l h d -> (b h) d l')
        v = rearrange(v, 'b l h d -> (b h) l d')

        # (b h) lq lk
        attn_weight = (q @ k)
        attn_weight_norm = attn_weight.softmax(dim=-1)

        # (b h) lq lk x (b h) lk d = (b h) lq d
        out = attn_weight_norm @ v
        out = rearrange(out, '(b h) l d -> b l (h d)', b=b, h=self.num_heads)

        # (b lq) d
        query2 = self.proj(out.view(-1, dim))
        query = query.view(-1, dim) + self.dropout0(query2)
        query = self.norm0(query)   # b q_len d

        # b q_len k_len
        mask = attn_weight.view(b, self.num_heads, -1, h*w).sum(1)
        mask = rearrange(mask, 'b l (h w) -> b l h w', h=h, w=w)

        return self.forward_ffn(query).view(b, -1, dim), mask

class MaskDecoder(nn.Module):
    def __init__(self, dim, num_heads=8, num_blocks=6, sum_masks=False, dim_info={'vehicle': [[0, 1], [1, 2]]}):
        super().__init__()

        self.num_blocks = num_blocks
        self.keys = []

        # create multiple decoders, each of which corresponds to a specific class
        self.decoders = nn.ModuleDict()
        for key, _ in dim_info.items():
            self.keys.append(key)
            layers = list()
            for i in range(num_blocks):
                layers.append(MaskDecoderBlock(dim=dim, num_heads=num_heads))
            self.decoders[key] = nn.Sequential(*layers)

        self.sum_masks = sum_masks
        self.dim_info = dim_info

    def forward(self, query, BEVfeat):
        """
        query : b n_class dim
        BEVfeat : {key : b (h w) dim}

        x : b n_class dim
        mask : b n_class h w
        """

        # split query into different classes
        mask_dict, query_dict = {}, {}
        for key, item in self.dim_info.items():
            query_dict[key], mask_dict[key] = [], []
            for idx in item: query_dict[key].append(query[:, idx[0]:idx[1]])
            query_dict[key] = torch.cat(query_dict[key], dim=1)

        # apply decoder
        for key, decoder in self.decoders.items():
            x, feat = query_dict[key], BEVfeat[key]
            for layer in decoder:
                x, mask = layer(x, feat)
                mask_dict[key].append(mask) # b class_dim h w

        # merge into one tensor
        masks = []
        for n in range(self.num_blocks):
            cur_layer_mask = []
            for key in self.keys: cur_layer_mask.append(mask_dict[key][n])
            masks.append(torch.cat(cur_layer_mask, dim=1))

        if (self.sum_masks):
            return [torch.stack(masks, dim=0).sum(dim=0)]
        else:
            return masks