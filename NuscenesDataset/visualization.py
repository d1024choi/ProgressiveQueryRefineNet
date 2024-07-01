import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import get_cmap


# many colors from
# https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/color_map.py
COLORS = {
    # static
    'lane':                 (110, 110, 110),
    'road_segment':         (90, 90, 90),

    # dividers
    'road_divider':         (255, 200, 0),
    'lane_divider':         (130, 130, 130),

    # dynamic
    'car':                  (255, 158, 0),
    'truck':                (255, 99, 71),
    'bus':                  (255, 127, 80),
    'trailer':              (255, 140, 0),
    'construction':         (233, 150, 70),
    'pedestrian':           (0, 0, 230),
    'motorcycle':           (255, 61, 99),
    'bicycle':              (220, 20, 60),

    'nothing':              (200, 200, 200)
}


def colorize(x, colormap=None):
    """
    x: (h w) np.uint8 0-255
    colormap
    """
    try:
        return (255 * get_cmap(colormap)(x)[..., :3]).astype(np.uint8)
    except:
        pass

    if x.dtype == np.float32:
        x = (255 * x).astype(np.uint8)

    if colormap is None:
        return x[..., None].repeat(3, 2)

    return cv2.applyColorMap(x, getattr(cv2, f'COLORMAP_{colormap.upper()}'))


def get_colors(semantics):
    return np.array([COLORS[s] for s in semantics], dtype=np.uint8)


def to_image(x):
    return (255 * x).byte().cpu().numpy().transpose(1, 2, 0)


def greyscale(x):
    return (255 * x.repeat(3, 2)).astype(np.uint8)


def resize(src, dst=None, shape=None, idx=0):
    if dst is not None:
        ratio = dst.shape[idx] / src.shape[idx]
    elif shape is not None:
        ratio = shape[idx] / src.shape[idx]

    width = int(ratio * src.shape[1])
    height = int(ratio * src.shape[0])

    return cv2.resize(src, (width, height), interpolation=cv2.INTER_CUBIC)


class BaseViz:

    def __init__(self, label_indices, SEMANTICS, Threshold, colormap='inferno'):
        self.label_indices = label_indices
        self.SEMANTICS = SEMANTICS
        self.Threshold = Threshold
        self.colors = get_colors(self.SEMANTICS)
        self.colormap = colormap

    def visualize_pred(self, bev, pred, bev_img, threshold=None):
        """
        (c, h, w) torch float {0, 1}
        (c, h, w) torch float [0-1]
        """
        if isinstance(bev, torch.Tensor):
            bev = bev.cpu().numpy().transpose(1, 2, 0)

        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy().transpose(1, 2, 0)

        if self.label_indices is not None:
            bev = [bev[..., idx].max(-1) for idx in self.label_indices]
            bev = np.stack(bev, -1)

        # if threshold is not None:
        #     pred = (pred > threshold).astype(np.float32)
        # result = colorize((255 * pred.squeeze(2)).astype(np.uint8), self.colormap)

        chk = pred[:, :, 0] > threshold
        bev_img[chk, 0] = 255
        bev_img[chk, 1] = 0
        bev_img[chk, 2] = 0

        return bev_img

    def visualize_bev(self, bev):
        """
        (c, h, w) torch [0, 1] float

        returns (h, w, 3) np.uint8
        """
        if isinstance(bev, torch.Tensor):
            bev = bev.cpu().numpy().transpose(1, 2, 0)

        h, w, c = bev.shape

        assert c == len(self.SEMANTICS)

        # Prioritize higher class labels
        eps = (1e-5 * np.arange(c))[None, None] # 1 1 c
        idx = (bev + eps).argmax(axis=-1)   # h w
        val = np.take_along_axis(bev, idx[..., None], -1)

        # Spots with no labels are light grey
        empty = np.uint8(COLORS['nothing'])[None, None] # 1 1 3

        result = (val * self.colors[idx]) + ((1 - val) * empty)
        result = np.uint8(result)

        return result

    def visualize_custom(self, batch, pred, b):
        return []

    @torch.no_grad()
    def visualize(self, batch, pred=None, b_max=8, **kwargs):
        bev = batch['bev'][:, -1] # b c targets h w -> b targets h w
        batch_size = bev.shape[0]

        for b in range(min(batch_size, b_max)):
            right = self.visualize_bev(bev[b])
            if pred is not None:
                pred_img = self.visualize_pred(bev[b], pred[0][b].sigmoid(), np.copy(right), threshold=self.Threshold)
                right_most = [pred_img]
            else:
                right_most = []


            right = [right] + right_most
            right = [x for x in right if x is not None]
            right = np.hstack(right)

            image = None if not hasattr(batch.get('image'), 'shape') else batch['image'][:, -1]


            if image is not None:
                imgs = [to_image(image[b][i]) for i in range(image.shape[1])]

                if len(imgs) == 6:
                    a = np.hstack(imgs[:3])
                    b = np.hstack(imgs[3:])
                    left = resize(np.vstack((a, b)), right)
                else:
                    left = np.hstack([resize(x, right) for x in imgs])

                yield np.hstack((left, right))
            else:
                yield right

    def __call__(self, batch=None, pred=None, **kwargs):
        return list(self.visualize(batch=batch, pred=pred, **kwargs))
