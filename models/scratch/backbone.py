from collections import OrderedDict
import sys
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
from torch import nn
from typing import Dict
from einops import rearrange
from models.ops.modules import MSDeformAttn, MSDeformAttnv2
from models.scratch.encoder import PositionEmbeddingSine

class Projection(torch.nn.Module):
    def __init__(self, in_channels, out_channels, residual=True, factor=2):
        super().__init__()

        dim = out_channels // factor

        self.conv = nn.Sequential(
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

    def forward(self, x):
        skip = x.clone()

        x = self.conv(x)
        if self.up is not None:
            up = self.up(skip)
            x = x + up

        return self.relu(x)


class Upsampler(torch.nn.Module):
    def __init__(self, in_channels, out_channels, residual, scale=1.0, factor=2):
        super().__init__()

        dim = out_channels // factor

        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True),
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

    def forward(self, x):
        skip = x.clone()

        x = self.conv(x)
        if self.up is not None:
            up = self.up(skip)
            up = F.interpolate(up, x.shape[-2:])
            x = x + up

        return self.relu(x)

class FeatCrossAttnLayer2(torch.nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        '''
        Feature interaction module via Key-Aware Deformable Attention.
        Each of n camera image features becomes a query and all the camera image features become values.
        '''

        self.n = 6
        self.conv0 = nn.Conv2d(dim, 256, 1)
        self.self_attn = MSDeformAttnv2(d_model=256, n_levels=self.n, n_heads=num_heads, n_points=4)
        self.conv1 = nn.Conv2d(256, dim, 1)

        self.dropout0 = nn.Dropout(dropout)
        self.norm0 = nn.LayerNorm(dim)

        # ffn
        self.linear1 = nn.Linear(dim, dim)
        self.activation = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim, dim)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)

        self.position_embedding = PositionEmbeddingSine(256 // 2, normalize=True)
        self.cam_emb = nn.Parameter(torch.rand(self.n, 256))

        # global feature
        self.conv2 = nn.Conv2d(256, 256, 1)

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def deformable_attention(self, featmap):

        bn, d, h, w = featmap.size()
        b = bn // self.n

        # camera dim. becomes level dim.
        src = rearrange(featmap, '(b n) d h w -> b n d h w', b=b, n=self.n)
        srcs, masks = [], []
        for _ in range(self.n):
            srcs.append(src[:, _]) # b d h w
            masks.append(torch.zeros(size=(b, h, w)).bool().to(src.device)) # b h w

        pos = self.position_embedding(h=h, w=w, device=featmap.device)  # b d h w
        pos = rearrange(pos, 'b c h w -> b (h w) c')   # b (h w) d

        # prepare input for encoder
        src_flatten, mask_flatten, cam_flatten, pos_flatten, global_flatten, spatial_shapes = [], [], [], [], [], []
        for cam_id, (src, mask) in enumerate(zip(srcs, masks)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2) # b c h w -> b (h w) c
            mask = mask.flatten(1) # b h w -> b (h w)
            cam = self.cam_emb[cam_id].view(1, 1, -1).repeat(b, h*w, 1)

            src_flatten.append(src)
            mask_flatten.append(mask)
            cam_flatten.append(cam)
            pos_flatten.append(pos)

            # TODO : what about src_pos_cam
            src_img = rearrange(src, 'b (h w) d -> b d h w', h=h, w=w)
            pos_img = rearrange(pos, 'b (h w) d -> b d h w', h=h, w=w)
            cam_img = rearrange(cam, 'b (h w) d -> b d h w', h=h, w=w)
            src_pos = rearrange(self.conv2(src_img+pos_img+cam_img), 'b d h w -> b (h w) d')
            global_flatten.append(torch.mean(src_pos, dim=1, keepdim=True)) # b 1 d

        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        cam_flatten = torch.cat(cam_flatten, 1)
        pos_flatten = torch.cat(pos_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device) # n 2
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1])) # n

        output = self.self_attn(src_flatten+cam_flatten+pos_flatten,
                                src_flatten, global_flatten, spatial_shapes, level_start_index, mask_flatten)

        return rearrange(output, 'b (n h w) d -> (b n) d h w', n=self.n, h=h, w=w)

    def forward(self, x_in):
        """
        x (feat. map) : (b n) c h w
        """

        if (isinstance(x_in, tuple)):
            x = x_in[0]
        else:
            x = x_in

        bn, d, h, w = x.size()
        b = bn // self.n

        input = x.clone()
        input = rearrange(input, '(b n) d h w -> b (n h w) d', b=b, n=self.n)

        x = self.conv0(x) # diminish ch size of x
        x = self.deformable_attention(x) # run deform. attn.
        x = self.conv1(x) # recover ch size
        output = rearrange(x, '(b n) d h w -> b (n h w) d', b=b, n=self.n)

        output = input + self.dropout0(output) # b (n h w) d
        output = self.norm0(output) # b (n h w) d
        output = self.forward_ffn(output) # b (n h w) d

        return rearrange(output, 'b (n h w) d -> (b n) d h w', n=self.n, h=h, w=w)


class _IntermediateLayerGetter(nn.ModuleDict):

    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__(layers)

        self.return_layers = orig_return_layers

        # debug,230622
        # _num_channels = [256, 512, 1024, 2048]
        # self.attn1 = nn.Sequential(*[FeatCrossAttnLayer(dim=_num_channels[0], num_heads=8) for _ in range(1)])
        # self.attn2 = nn.Sequential(*[FeatCrossAttnLayer(dim=_num_channels[1], num_heads=8) for _ in range(1)])
        # self.attn3 = nn.Sequential(*[FeatCrossAttnLayer(dim=_num_channels[2], num_heads=8) for _ in range(1)])
        # self.attn4 = FeatCrossAttnLayer(dim=_num_channels[3], num_heads=8)

    def forward(self, x):

        out = OrderedDict()
        for name, module in self.items():
            if ('attn' in name): continue
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x

        # out['0'] = self.attn1(out['0'])
        # out['1'] = self.attn2(out['1'])
        # out['2'] = self.attn3(out['2'])

        return out


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, cfg, backbone, train_backbone, return_interm_layers=True):
        super().__init__()

        # set to train mode
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

        # load training settings
        self.cfg = cfg
        self.target_lvls = [lvl for _, lvl in enumerate(cfg['target_feat_levels'])]

        h_dim = cfg['encoder']['dim']
        _return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        _strides, _num_channels = [4, 8, 16, 32], [256, 512, 1024, 2048]

        # main body
        self.return_layers = {}
        for _, (key, lvl) in enumerate(_return_layers.items()):
            if (lvl in self.target_lvls):
                self.return_layers[key] = lvl
        self.body = _IntermediateLayerGetter(backbone, return_layers=self.return_layers)

        # feat-wise attention & projection
        self.keys = []
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.attn, self.input_proj, self.merge = nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict()
        for key, lvl in self.return_layers.items():

            in_channels = _num_channels[int(lvl)]

            # attention (Inter Camera Interaction Module)
            self.attn[lvl] = nn.Sequential(*[FeatCrossAttnLayer2(dim=in_channels, num_heads=8)
                                      for _ in range(cfg['feat_int_repeat'])])


            # projection
            self.input_proj[lvl] = nn.Conv2d(in_channels, h_dim, kernel_size=1)

            # merge  (Intra Camera Interaction Module - FPN)
            self.merge[lvl] = nn.Conv2d(h_dim, h_dim, kernel_size=3, padding=1)

            # others
            self.keys.append(lvl)



    def forward(self, x):

        # extract multi-scale feat. maps
        x_intermediate = self.body(x)

        # attention and then projection ----
        x_proj = {}
        for key in x_intermediate.keys():

            # attention (inter-camera interaction module)
            if (self.attn[key] is not None):
                x_attn = self.attn[key](x_intermediate[key])
            else:
                x_attn = x_intermediate[key]

            # projection
            if (self.input_proj[key] is not None):
                x_attn = self.input_proj[key](x_attn)
            x_proj[key] = x_attn


        # FPN (intra camera interaction module)------
        x_fpn = {self.keys[-1] : x_proj[self.keys[-1]]}
        for idx, key in enumerate(sorted(self.keys, reverse=True)): # from low-res to high-res
            key_m1 = str(int(key)-1)
            if (key_m1 in self.keys):
                high = x_proj[key_m1]
                high = high + self.up(x_fpn[key])
                x_fpn[key_m1] = self.merge[key_m1](high)


        # Reformatting -----
        x_out = []
        for key in self.keys:
            x_out.append(x_fpn[key])

        return x_out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, cfg, train_backbone=True, return_interm_layers=True, dilation=False):

        name = cfg['backbone']['model_name']
        print(f'>> {name} is selected for backbone network!!')
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=True, norm_layer=FrozenBatchNorm2d)
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(cfg, backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2
