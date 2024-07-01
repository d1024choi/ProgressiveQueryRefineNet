from utils.functions import *
import torch
from einops import rearrange, repeat
import torch.nn as nn
import torch.nn.functional as F
from models.scratch.backbone import Backbone
from models.scratch.decoder import Decoder, MaskDecoderBlock, MaskDecoder, DecoderBlock
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from models.ops.modules import MSDeformAttn
from models.scratch.encoder import BEVFormerEncoder
import copy
import os
import math

class UpsampleBlock(torch.nn.Module):
    def __init__(self, dim, scale=1.0):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class Upsampler(torch.nn.Module):
    def __init__(self, dim, scale, num_repeat):
        super().__init__()

        self.layers = nn.ModuleDict()
        for i in range(num_repeat):
            self.layers[str(i)] = UpsampleBlock(dim=dim, scale=scale)

    def forward(self, x):
        for key, layer in self.layers.items():
            x = layer(x)
        return x


class Scratch(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # parameter settings ----
        self.cfg = cfg
        self.h_dim = cfg['encoder']['dim']
        self.num_res = cfg['hierarchy_depth']

        self.batch = cfg['training_params']['batch_size']
        self.n_cam = cfg['image']['n_cam']
        self.n_lvl = len(cfg['encoder']['feat_levels'])
        self.z_candi = cfg['encoder']['z_candi']

        # from high-res (0) to low-res (num_res-1)
        self.h = [cfg['bev']['h'] // int(math.pow(2, _-1)) for _ in range(1, self.num_res+1)]
        self.w = [cfg['bev']['w'] // int(math.pow(2, _-1)) for _ in range(1, self.num_res+1)]

        self.output, num_dec_Q = {}, 0
        for _, key in enumerate(cfg['target']):
            if (key == 'vehicle' or key == 'pedestrian'):
                self.output[key] = [[num_dec_Q, num_dec_Q+1], [num_dec_Q+1, num_dec_Q+2]]
                num_dec_Q += 2
            else:
                self.output[key] = [[num_dec_Q, num_dec_Q+1]]
                num_dec_Q += 1

        # Image backbone ----
        self.BackBone = Backbone(cfg=cfg)

        # Decoding Queries ----
        self.dec_queries = nn.Embedding(num_dec_Q, self.h_dim)

        # Hierarchical Encoder ----
        self.BEVEncoder, self.UpSampler = nn.ModuleDict(), nn.ModuleDict()
        for _ in range(1, self.num_res+1):
            scale = int(math.pow(2, _-1))
            self.BEVEncoder[str(_-1)] = BEVFormerEncoder(cfg=cfg, scale=scale)
            if (_-1 == 0): self.UpSampler[str(_-1)] = None
            else:
                from models.scratch.decoder import SimpleUpsampler
                self.UpSampler[str(_-1)] = SimpleUpsampler(mode='nearest')

        # Headers ----
        self.Header = nn.ModuleDict()
        for _, key in enumerate(cfg['target']):
            self.Header[key] = None
            if (cfg['bool_dec_head']):
                self.Header[key] = nn.Sequential(nn.Conv2d(self.h_dim, self.h_dim, 3, padding=1, bias=False),
                                                 nn.BatchNorm2d(self.h_dim),
                                                 nn.ReLU(inplace=True),
                                                 nn.Conv2d(self.h_dim, self.h_dim, 1))


        # Decoder ----
        self.Decoder = nn.ModuleDict()
        self.Decoder['main'] = MaskDecoder(dim=self.h_dim, num_heads=8, dim_info=self.output)

        num_repeat = int(math.log(self.h[0] / self.h[-1], 2))
        self.Decoder['shortcut'] = Upsampler(dim=self.h_dim, scale=2.0, num_repeat=num_repeat)

        out_logit_dim = len(cfg['target'])
        from models.scratch.decoder import Decoder
        self.Decoder['intm'] = nn.ModuleDict()
        for bev_size in sorted(self.h):
            if (bev_size < self.h[0]):
                self.Decoder['intm'][str(bev_size)] = Decoder(in_channels=self.h_dim,
                                                              out_channels=self.h_dim,
                                                              num_iter=int(math.log(self.h[0] / bev_size, 2)),
                                                              out_logit_dim=out_logit_dim)


        self.Decoder['offset'] = nn.ModuleDict()
        self.Decoder['offset']['vehicle'], self.Decoder['offset']['pedestrian'] = None, None
        if ('vehicle' in cfg['target'] and cfg['bool_learn_offset']):
            self.Decoder['offset']['vehicle'] = nn.Sequential(nn.Conv2d(self.h_dim, self.h_dim, 3, padding=1, bias=False),
                                                   nn.BatchNorm2d(self.h_dim),
                                                   nn.ReLU(inplace=True),
                                                   nn.Conv2d(self.h_dim, 2, 1))

        if ('pedestrian' in cfg['target'] and cfg['bool_learn_offset']):
            self.Decoder['offset']['pedestrian'] = nn.Sequential(nn.Conv2d(self.h_dim, self.h_dim, 3, padding=1, bias=False),
                                                   nn.BatchNorm2d(self.h_dim),
                                                   nn.ReLU(inplace=True),
                                                   nn.Conv2d(self.h_dim, 2, 1))

        self.crosstaskhead = None
        if (cfg['bool_apply_crosshead']):
            from models.scratch.decoder import CrossTaskHead
            self.crosstaskhead = CrossTaskHead(dim=self.h_dim, targets=cfg['target'])

        # Initialize
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.dec_queries.weight)

    def forward(self, batch, dtype, rank, isTrain=True):

        # input keys
        inputs = ['image', 'intrinsics', 'extrinsics']


        # conversion to sequential batch
        b, s, n, c, h, w = batch['image'].size()
        seq_batch = [{} for _ in range(s)]
        for t in range(s):
            for key in inputs:
                if (self.cfg['ddp']): seq_batch[t][key] = batch[key][:, t].type(dtype).to(rank)
                else: seq_batch[t][key] = batch[key][:, t].type(dtype).cuda()

        # intrinsic/extrinsic parameters
        intrinsics = seq_batch[-1]['intrinsics']                                        # b n 3 3
        extrinsics = seq_batch[-1]['extrinsics']                                        # b n 4 4


        # extract image feature maps ---------------------------
        features = self.BackBone(seq_batch[-1]['image'].view(-1, c, h, w))


        # encoding part ----------------------------------------
        intm_bev_queries = {}
        bev_queries = None
        for (_, enc), (_, up) in sorted(zip(self.BEVEncoder.items(), self.UpSampler.items()), reverse=True):
            bev_queries = enc(bev_queries, features, intrinsics, extrinsics) # b (h w) d

            if (up is not None):
                bev_queries = rearrange(bev_queries, 'b (h w) d -> b d h w', h=self.h[int(_)], w=self.w[int(_)])
                intm_bev_queries[str(bev_queries.size(3))] = bev_queries
                bev_queries = rearrange(up(bev_queries), 'b d h w -> b (h w) d')


        # skip connection : final = lowest res + highest res
        bev_LR = intm_bev_queries[str(self.h[-1])]
        bev_queries = rearrange(bev_queries, 'b (h w) d -> b d h w', h=self.h[0], w=self.w[0])
        bev_queries = self.Decoder['shortcut'](bev_LR) + bev_queries


        # projection headers
        projected_queries = {}
        for key, header in self.Header.items():
            if (header is not None):
                projected_queries[key] = rearrange(header(bev_queries), 'b d h w -> b (h w) d')
            else:
                projected_queries[key] = rearrange(bev_queries, 'b d h w -> b (h w) d')

        # cross task headers
        if (self.crosstaskhead is not None):
            projected_queries = self.crosstaskhead(projected_queries)


        # decoding part ----------------------------------------
        dec_queries = self.dec_queries.weight[None].repeat(b, 1, 1)
        z = self.Decoder['main'](dec_queries, projected_queries)[-1]

        # intermediate resol. queries to logits
        intm_logits = []
        for key, Q in intm_bev_queries.items():
            if (int(key) == self.h[-1]):
                intm_logits.append(self.Decoder['intm'][key](Q))

        # offsets from vehicle/pedestrian centers
        offsets = None
        if (self.cfg['bool_learn_offset']):
            offsets = {'vehicle': None, 'pedestrian': None}
            if (self.Decoder['offset']['vehicle'] is not None):
                _input = rearrange(projected_queries['vehicle'], 'b (h w) d -> b d h w', h=self.h[0], w=self.w[0])
                offsets['vehicle'] = self.Decoder['offset']['vehicle'](_input)

            if (self.Decoder['offset']['pedestrian'] is not None):
                _input = rearrange(projected_queries['pedestrian'], 'b (h w) d -> b d h w', h=self.h[0], w=self.w[0])
                offsets['pedestrian'] = self.Decoder['offset']['pedestrian'](_input)


        # outputs ----------------------------------------------
        output = {}
        for key, item in self.output.items():
            output[key] = []
            for idx in item: output[key].append(z[:, idx[0]:idx[1]])
        output.update({'intm': intm_logits})
        output.update({'offsets': offsets})

        return output
