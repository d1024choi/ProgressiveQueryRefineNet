import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from einops import rearrange, repeat
from models.ops.functions import MSDeformAttnFunction
from models.ops.modules import MSDeformAttn, MSDeformAttn3D

import warnings
warnings.filterwarnings(action='ignore')

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class TemporalSelfAttention(nn.Module):
    def __init__(self, cfg, scale=1, d_model=256, n_levels=1, n_heads=8, n_points=4, mixed_precision=False):
        super().__init__()

        self.cfg = cfg
        self.h, self.w = cfg['bev']['h'] // int(scale), cfg['bev']['w'] // int(scale)

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.mixed_precision = mixed_precision

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self,
                query,
                query_pos,
                reference_points,
                input_flatten,
                input_padding_mask=None):

        """
        query : b num_queries dim
        query_pos : b num_queries dim
        reference_points : b num_queries 1 2
        input_flatten : b num_inputs dim
        input_padding_mask : b num_inputs
        """

        if (query_pos is not None):
            query = query + query_pos

        input_spatial_shapes = torch.as_tensor([(self.h, self.w)], dtype=torch.long, device=query.device) # lvl 2
        input_level_start_index = torch.cat((input_spatial_shapes.new_zeros((1, )),
                                             input_spatial_shapes.prod(1).cumsum(0)[:-1])) # lvl

        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)

        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        assert torch.count_nonzero(torch.isnan(attention_weights)) == 0

        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        else:
            raise ValueError(
                'Last dim of reference_points must be 2, but get {} instead.'.format(reference_points.shape[-1]))

        if (self.mixed_precision):
            value = value.float()
            sampling_locations = sampling_locations.float()
            attention_weights = attention_weights.float()

        output = MSDeformAttnFunction.apply(value, input_spatial_shapes, input_level_start_index, sampling_locations,
                                            attention_weights, self.im2col_step)

        output = self.output_proj(output)
        return output

class SpatialCrossAttention(nn.Module):
    def __init__(self, cfg, d_model=256, n_levels=1, n_heads=8, n_points=4, mixed_precision=False, scale=1):
        super().__init__()

        self.cfg = cfg
        self.h, self.w = cfg['bev']['h'] // int(scale), cfg['bev']['w'] // int(scale)
        self.z_candi = cfg['encoder']['z_candi']

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.mixed_precision = mixed_precision

        self.cross_attn = MSDeformAttn3D(d_model, n_levels, n_heads, 2 * n_points, mixed_precision)
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, queries, embeds, features, reference_points_3D, bev_mask):
        '''
        queries : b n (h w) d
        features : [(b*n c h' w')]
        embeds : [pos_emb, lvl_emb, cam_emb]
           pos_emb : b 1 (h w) d
           lvl_emb : lvl d
           cam_emb : n_cam d
        reference_points_3d : b n (h w) D 2
        bev_mask : b n (h w) D
        '''

        b, n, q_len, dim = queries.size()
        pos_emb, lvl_emb, cam_emb = embeds[0], embeds[1], embeds[2],

        # reshape input features
        input_flatten, mask_flatten, spatial_shapes = [], [], []
        for l in range(len(features)):

            h, w = features[l].size(-2), features[l].size(-1)
            spatial_shapes.append((h, w))

            # (b n) c h w -> b n c (h w)
            input = rearrange(features[l], '(b n) c h w -> b n c (h w)', b=b, n=n)

            if (lvl_emb is not None):
                input = input + lvl_emb[l][None, None, :, None]

            if (cam_emb is not None):
                input = input + cam_emb[None, :, :, None]

            # b n c (h w) -> (b n) (h w) c
            input = rearrange(input, 'b n c l -> (b n) l c')
            input_flatten.append(input)

            # (b n) (h w) 1
            mask = torch.zeros(size=(b*n, h*w)).bool().to(input.device)
            mask_flatten.append(mask)

        input_flatten = torch.cat(input_flatten, dim=1)     # (b n) (h' w') c
        mask_flatten = torch.cat(mask_flatten, 1)           # (b n) (h' w')
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=queries.device)      # lvl 2
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1])) # lvl


        # assume that batch_size = 1
        indexes = []
        for i in range(bev_mask.size(1)):
            # mask for i-th camera image
            mask_per_img = bev_mask[:, i, ..., 0]  # b (h w) D

            # if at least one depth falls into the image, the corresponding query is used.
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)  # b (h w) D -> (h w) D -> (h w)
            indexes.append(index_query_per_img)
        max_q_len = max([len(each) for each in indexes])

        # create new queries and reference_points of size max_q_len
        queries_pos = queries
        if (pos_emb is not None): queries_pos = queries_pos + pos_emb
        queries_rebatch = queries.new_zeros([b, n, max_q_len, self.d_model])  # b n l dim
        reference_points_rebatch = reference_points_3D.new_zeros([b, n, max_q_len, len(self.z_candi), 2])  # b n l D 2
        for j in range(b):
            for i in range(n):
                index_query_per_img = indexes[i]
                queries_rebatch[j, i, :len(index_query_per_img)] = queries_pos[j, i, index_query_per_img]

                reference_points_per_img = reference_points_3D[j, i]  # (h w) D 2
                reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[
                    index_query_per_img]

        queries_rebatch = rearrange(queries_rebatch, 'b n l d -> (b n) l d')
        reference_points_rebatch = rearrange(reference_points_rebatch, 'b n l d c -> (b n) l d c')
        queries_attn = self.cross_attn(queries_rebatch, reference_points_rebatch, input_flatten, spatial_shapes,
                                 level_start_index, mask_flatten)
        queries_attn = rearrange(queries_attn, '(b n) l d -> b n l d', b=b, n=n)

        # re-store
        slots = torch.zeros_like(queries)  # b n (h w) d
        count = torch.zeros(size=(n, self.h * self.w)).to(slots)
        for j in range(b):
            for i, index_query_per_img in enumerate(indexes):
                slots[j, i, index_query_per_img] = queries_attn[j, i, :len(index_query_per_img)] # BUG??
                count[i, index_query_per_img] += 1

        slots = slots.sum(dim=1)    # b (h w) d
        count = count.view(n, self.h, self.w).sum(dim=0).squeeze()  # (h w)
        count = torch.clamp(count, min=1.0)
        slots = slots / count.view(1, -1, 1) # b (h w) d

        return self.output_proj(slots)

class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self, cfg, scale=1, d_model=256, d_ffn=1024, dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, mixed_precision=False):
        super().__init__()

        self.cfg = cfg
        self.n_cam = cfg['image']['n_cam']
        self.h, self.w = cfg['bev']['h'] // int(scale), cfg['bev']['w'] // int(scale)
        self.h_dim = cfg['encoder']['dim']
        self.n_lvl = len(cfg['encoder']['feat_levels'])
        self.z_candi = cfg['encoder']['z_candi']

        # self attention
        self.self_attn = TemporalSelfAttention(cfg, scale, d_model, 1, n_heads, n_points, mixed_precision)
        self.dropout0 = nn.Dropout(dropout)
        self.norm0 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = SpatialCrossAttention(cfg, d_model, n_levels, n_heads, n_points, mixed_precision, scale=scale)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def _reset_parameters(self):
        xavier_uniform_(self.linear1.weight.data)
        constant_(self.linear1.bias.data, 0.)
        xavier_uniform_(self.linear2.weight.data)
        constant_(self.linear2.bias.data, 0.)


    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, queries, embeds, reference_points_3d, reference_points_2d, features, bev_mask):
        '''
        queries : b (h w) d
        features : [(b*n c h' w')]
        embeds : [pos_emb, level_emb, cam_emb]
           pos_emb : b (h w) d
           level_emb : lvl d
           cam_emb : n_cam d
        reference_points_3d : b n (h w) D 2
        bev_mask : b n (h w) D 2
        reference_points_2d : b (h w) 2
        '''

        b, q_len, d = queries.size()
        pos_emb, lvl_emb, cam_emb = embeds[0], embeds[1], embeds[2]

        # self-attention + (add, norm)
        reference_points_2d = rearrange(reference_points_2d, 'b l d -> b l 1 d')
        queries_out = self.self_attn(query=queries,
                                     query_pos=pos_emb,
                                     reference_points=reference_points_2d,
                                     input_flatten=queries)
        queries = queries + self.dropout0(queries_out)
        queries = self.norm0(queries) # b q_len d


        # cross-attention + (proj, add, norm)
        queries_repeat = queries[:, None].repeat(1, self.n_cam, 1, 1) # b n (h w) d
        pos_emb_repeat = pos_emb[:, None].repeat(1, self.n_cam, 1, 1) # b n (h w) d
        queries_out = self.cross_attn(queries_repeat, [pos_emb_repeat, lvl_emb, cam_emb],
                                      features, reference_points_3d, bev_mask)
        queries = queries + self.dropout1(queries_out)
        queries = self.norm1(queries) # b q_len d

        # ffn (linear, add, norm)
        return self.forward_ffn(queries.view(-1, d)).view(b, q_len, d)


def main():
    N, M, D = 1, 2, 2
    Lq, L, P = 2, 2, 2
    shapes = torch.as_tensor([(6, 4), (3, 2)], dtype=torch.long).cuda()
    level_start_index = torch.cat((shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1]))
    S = sum([(H * W).item() for H, W in shapes])

    value = torch.rand(N, S, M, D).cuda() * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2).cuda()
    attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
    im2col_step = 2


if __name__ == '__main__':
    main()