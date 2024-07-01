import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import copy
from einops import rearrange, repeat
from torchvision.models.resnet import Bottleneck
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from models.ops.modules import MSDeformAttn, MSDeformAttn3D
from models.scratch.deformable_transformer import DeformableTransformerEncoderLayer

def generate_grid(height: int, width: int):
    '''
    F.pad : to pad the last 3 dimensions, use (left, right, top, bottom, front, back)
    For example,
       x = torch.zeros(size=(2, 3, 4))
       x = F.pad(x, (a, b, c, d, e, f), value=1)
       x.size() # 2+e+f x 3+c+d x 4+a+b
    '''

    if (False):
        xs = torch.linspace(0, 1, width)
        ys = torch.linspace(0, 1, height)
    else:
        # Original paper implementation
        xs = (torch.linspace(0.5, width - 0.5, width) / width)
        ys = (torch.linspace(0.5, height - 0.5, height) / height)

    indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)       # 2 h w
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)                   # 3 h w
    indices = indices[None]                                                 # 1 3 h w

    return indices


def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    """
    copied from ..data.common but want to keep models standalone
    """
    sh = h / h_meters
    sw = w / w_meters

    return [
        [ 0., -sw,          w/2.],
        [-sh,  0., h*offset+h/2.],
        [ 0.,  0.,            1.]
    ]


class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()

        self.register_buffer('mean', torch.tensor(mean)[None, :, None, None], persistent=False)
        self.register_buffer('std', torch.tensor(std)[None, :, None, None], persistent=False)

    def forward(self, x):
        return (x - self.mean) / self.std


class RandomCos(nn.Module):
    def __init__(self, *args, stride=1, padding=0, **kwargs):
        super().__init__()

        linear = nn.Conv2d(*args, **kwargs)

        self.register_buffer('weight', linear.weight)
        self.register_buffer('bias', linear.bias)
        self.kwargs = {
            'stride': stride,
            'padding': padding,
        }

    def forward(self, x):
        return torch.cos(F.conv2d(x, self.weight, self.bias, **self.kwargs))


class GenerateGrid(nn.Module):
    def __init__(
        self,
        bev_height,
        bev_width,
        h_meters,
        w_meters,
        offset,
        z_candi=[0.0, 1.0, 2.0, 3.0, 4.0]
    ):
        super().__init__()

        # each decoder block upsamples the bev embedding by a factor of 2
        h = bev_height
        w = bev_width

        # bev coordinates
        grid_2D = generate_grid(h, w).squeeze(0)
        grid_2D[0] = bev_width * grid_2D[0]
        grid_2D[1] = bev_height * grid_2D[1]

        # map from bev coordinates to ego frame
        V = get_view_matrix(bev_height, bev_width, h_meters, w_meters, offset)  # 3 3
        V_inv = torch.FloatTensor(V).inverse()                                  # 3 3
        grid_2D = V_inv @ rearrange(grid_2D, 'd h w -> d (h w)')                # 3 (h w)
        grid_2D = rearrange(grid_2D, 'd (h w) -> d h w', h=h, w=w)              # 3 h w

        # create 3D grid
        grid_3D = []
        for _, z in enumerate(z_candi):
            z_axis = z * torch.ones(size=(1, h, w)).to(grid_2D)                 # 1 h w
            grid_wz = torch.cat((grid_2D[:2], z_axis), dim=0)                   # 3 h w
            grid_3D.append(grid_wz.unsqueeze(0))                                # 1 3 h w
        grid_3D = torch.cat(grid_3D, dim=0)                                     # D 3 h w

        # egocentric frame
        self.register_buffer('grid_2D', grid_2D, persistent=False)              # 3 h w
        self.register_buffer('grid_3D', grid_3D, persistent=False)              # D 3 h w


class ReferencePoints(nn.Module):
    def __init__(self, image_h, image_w, depth_thr=1.0):
        super().__init__()

        self.h, self.w = image_h, image_w
        self.depth_thr = depth_thr

    def get_3d(self, grid_3D, I, E, normalize=True):
        '''
        grid_3D : D 3 h w
        I : b n 3 3
        E : b n 4 4

        image_2D (mask) : b n D 2 (h w)
        '''

        b, n, _, _ = I.size()

        grid_homo = F.pad(grid_3D, (0, 0, 0, 0, 0, 1, 0, 0), value=1)       # D 4 h w
        grid_homo_flat = rearrange(grid_homo, 'd c h w -> d c (h w)')       # D 4 (h w)
        camera_3D = E[:, :, None] @ grid_homo_flat[None, None]              # b n D 4 (h w)
        image_homo = I[:, :, None] @ camera_3D[..., :3, :]                  # b n D 3 (h w)

        eps = 1e-5
        bev_mask = (image_homo[:, :, :, [-1], :] > eps).repeat(1, 1, 1, 2, 1)           # b n D 2 (h w)
        denorm = torch.maximum(image_homo[:, :, :, [-1], :], torch.ones_like(image_homo[:, :, :, [-1], :]) * eps)
        ref_pts_3D = image_homo[..., :2, :] / denorm                                    # b n D 2 (h w)

        mask = torch.ones_like(ref_pts_3D).bool()                                       # b n D 2 (h w)
        mask = torch.logical_and(mask, bev_mask)
        mask = torch.logical_and(mask, ref_pts_3D[..., [0], :] > 0)
        mask = torch.logical_and(mask, ref_pts_3D[..., [0], :] < self.w)
        mask = torch.logical_and(mask, ref_pts_3D[..., [1], :] > 0)
        mask = torch.logical_and(mask, ref_pts_3D[..., [1], :] < self.h)

        if (normalize):
            ref_pts_3D[..., 0, :] = ref_pts_3D[..., 0, :] / self.w
            ref_pts_3D[..., 1, :] = ref_pts_3D[..., 1, :] / self.h

        ref_pts_3D = rearrange(ref_pts_3D, 'b n D p l -> b n l D p')
        mask = rearrange(mask, 'b n D p l -> b n l D p')

        return ref_pts_3D, mask

    def get_2d(self, h, w):

        # TODO : shift by 0.5?
        x = torch.linspace(0, 1, w)
        y = torch.linspace(0, 1, h)
        # # Original paper implementation
        # x = (torch.linspace(0.5, w - 0.5, w) / w)
        # y = (torch.linspace(0.5, h - 0.5, h) / h)

        indices = torch.stack(torch.meshgrid((x, y), indexing='xy'), 0)  # 2 h w
        ref_pts_2D = rearrange(indices, 'c h w -> 1 (h w) c')   # 1 (h w) 2

        return ref_pts_2D


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, h, w, device=None):

        mask = torch.ones(size=(1, h, w)).to(device)
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale # 0 to 2*pi
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t # sin(2 * pi / T)
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos


class BEVFormerEncoder(nn.Module):
    def __init__(self, cfg, scale=1):
        super().__init__()

        self.cfg = cfg
        self.h_dim = cfg['encoder']['dim']
        self.bs = cfg['training_params']['batch_size']
        self.n_cam = cfg['image']['n_cam']
        self.n_lvl = len(cfg['encoder']['feat_levels'])
        self.z_candi = cfg['encoder']['z_candi']
        self.h, self.w = cfg['bev']['h'] // int(scale), cfg['bev']['w'] // int(scale)

        # Ego-centric grids
        self.grids = GenerateGrid(bev_height=self.h,
                                  bev_width=self.w,
                                  h_meters=cfg['bev']['h_meters'],
                                  w_meters=cfg['bev']['w_meters'],
                                  offset=cfg['bev']['offset'],
                                  z_candi=self.z_candi)

        # Ego-centric grid points to reference points
        self.GetReferencePoints = ReferencePoints(cfg['image']['h'], cfg['image']['w'])



        # Embeddings
        self.position_embedding = PositionEmbeddingSine(self.h_dim // 2, normalize=True)
        self.level_embed = nn.Parameter(torch.rand(self.n_lvl, self.h_dim))
        self.cams_embed = nn.Parameter(torch.rand(self.n_cam, self.h_dim))

        encoder = DeformableTransformerEncoderLayer(cfg=cfg, scale=scale, d_model=self.h_dim, d_ffn=1024, dropout=0.1, activation="relu",
                                                    n_levels=self.n_lvl, n_heads=8, n_points=4,
                                                    mixed_precision=cfg['bool_mixed_precision'])

        # Transformer Encoder
        self.DeformAttnEnc = nn.ModuleList([copy.deepcopy(encoder) for _ in range(cfg['encoder']['repeat'])])

        # Initialize
        self.bev_queries_init = nn.Embedding(self.h * self.w, self.h_dim)
        self._reset_parameters()


    def _reset_parameters(self):
        xavier_uniform_(self.level_embed)
        xavier_uniform_(self.cams_embed)
        xavier_uniform_(self.bev_queries_init.weight)

    def preprocess_queries(self, queries_init, queries_prev):
        '''
        queries_init (_prev) : b (h w) d
        '''
        if (queries_prev is None):
            queries = queries_init
        else:
            queries = queries_init + queries_prev

        return queries

    def forward(self, queries_prev, features, intrinsics, extrinsic):
        '''
        queries (from previous level) : b (h w) d
        features : [((b n) c h' w')]
        intrinsics : b n 3 3
        extrinsics : b n 4 4
        '''

        b = intrinsics.size(0)
        queries_init = self.bev_queries_init.weight[None].repeat(b, 1, 1)  # b (h w) d
        queries = self.preprocess_queries(queries_init, queries_prev)

        # reference points
        reference_points_3d, bev_mask = self.GetReferencePoints.get_3d(self.grids.grid_3D, intrinsics, extrinsic)  # b n (h w) D 2
        reference_points_2d = self.GetReferencePoints.get_2d(self.h, self.w)
        reference_points_2d = reference_points_2d.repeat(b, 1, 1).to(queries) # b (h w) 2

        # positional embeddings
        pos_emb = self.position_embedding(h=self.h, w=self.w, device=queries.device)    # b d h w
        pos_emb = rearrange(pos_emb, 'b c h w -> b (h w) c')   # b (h w) d

        embeds = [pos_emb, self.level_embed, self.cams_embed]
        for _, layer in enumerate(self.DeformAttnEnc):
            queries = layer(queries, embeds, reference_points_3d, reference_points_2d, features, bev_mask)

        return queries  # b 1 (h w) d