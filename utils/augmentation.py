from typing import Tuple
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import numpy.random as random
from PIL import Image
import matplotlib.pyplot as plt
import skimage
import random

class AffineTransform:

    def __init__(self, max_rotation_degree=0, max_translation_pixel=0, max_scale_severity=0):
        '''
        max_rotation_degree: maximum allowable rotation degree
        max_translation_pixel: maximum allowable shifting amount
        max_scale_severity: max allowable scale amount (from 0 to 1)
        '''

        self.max_rotation_degree = max_rotation_degree
        self.max_translation_pixel = max_translation_pixel
        self.max_scale_severity = max_scale_severity
        self.max_scale_factor = 1 + max_scale_severity
        self.min_scale_factor = 1 - max_scale_severity

        assert self.max_scale_factor >= 1 and self.min_scale_factor <= 1

    def __call__(self, img, intrinsic):
        """
        img : image of type PIL.Image
        intrinsic : 3 x 3 numpy array
        """

        # conversion to np.array of type np.float32
        img = np.array(img).astype('float')

        # read camera images
        jitter_x, jitter_y, scale_x, scale_y = 0, 0, 1.0, 1.0

        if (self.max_translation_pixel > 0):
            sign_x = -1 if np.random.rand(1) < 0.5 else 1
            sign_y = -1 if np.random.rand(1) < 0.5 else 1

            jitter_x = int(sign_x * self.max_translation_pixel * np.random.rand(1))
            jitter_y = int(sign_y * self.max_translation_pixel * np.random.rand(1))

        if (self.max_scale_severity > 0):
            scale_x = random.uniform(self.min_scale_factor, self.max_scale_factor)
            scale_y = random.uniform(self.min_scale_factor, self.max_scale_factor)

        intrinsic[0, -1] += jitter_x
        intrinsic[1, -1] += jitter_y

        intrinsic[0, 0] *= scale_x
        intrinsic[1, 1] *= scale_y

        tform = skimage.transform.AffineTransform(translation=(jitter_x, jitter_y), scale=(scale_x, scale_y))
        img_trans = skimage.transform.warp(img, tform._inv_matrix, mode='reflect', preserve_range=True)

        return Image.fromarray(img_trans.astype('uint8'))


class PhotoMetricDistortion:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=9):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        self.dtype = np.float32

    def __call__(self, img):
        """
        img : image of type PIL.Image
        """

        # conversion to np.array of type np.float32
        img = np.array(img).astype(self.dtype)


        # random brightness
        if random.randint(2):
            delta = random.uniform(-self.brightness_delta, self.brightness_delta)
            img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2YCrCb).astype(self.dtype)
            img[..., 0] += delta
            img[img[..., 0] > 255, 0] = 255.0
            img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_YCrCb2RGB).astype(self.dtype)
            # print(f'random brightness: {delta}')


        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                img *= alpha
                img[img > 255.0] = 255.0
                # print(f'random contrast: {alpha}')


        # convert color from BGR to HSV
        img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2HSV).astype(self.dtype)


        # random saturation
        if random.randint(2):
            alpha = random.uniform(self.saturation_lower, self.saturation_upper)
            img[..., 1] *= alpha
            img[img[..., 1] > 255.0] = 255.0
            # print(f'random saturation: {alpha}')


        # random hue
        if random.randint(2):
            delta = random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0] += delta
            img[..., 0][img[..., 0] > 179] = 179
            # print(f'random hue: {delta}')

        # convert color from HSV to BGR
        img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_HSV2RGB).astype(self.dtype)

        # random contrast
        if mode == 0:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                img *= alpha
                # print(f'random contrast: {alpha}')

        # randomly swap channels
        if random.randint(2):
            img = img[..., random.permutation(3)]
            # print('random permutation')

        return Image.fromarray(img.astype('uint8'))



if __name__ == '__main__':

    img_dist = PhotoMetricDistortion()
    img = Image.open('/home/dooseop/Pytorch/TopViewSeg/TVSS_v2p3a/test.png')
    output = img_dist(img)