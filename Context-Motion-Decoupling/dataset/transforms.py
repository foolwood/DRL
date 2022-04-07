# Copyright (C) Alibaba Group Holding Limited. 

import random
import numpy as np
import cv2
import torch
import math
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter

__all__ = ['Compose', 'RandomCrop', 'Rescale', 'CenterCrop', 'RandomHFlip',
            'GaussianBlur', 'ColorJitter', 'RandomGray', 'ToTensor', 
            'Normalize', 'MCompose', 'MRandomCrop', 'MRescale',
            'MCenterCrop', 'MRandomHFlip', 'MToTensor', 'MNormalize',
            'VCompose', 'VRandomRotation', 'VRandomCrop', 'VRescale',
            'VCenterCrop', 'VRandomHFlip', 'VGaussianBlur',
            'VColorJitter', 'VRandomGray', 'VToTensor', 'VNormalize']

class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, rgb, m, i):
        if m[0].dtype != np.float32:
            m = [u.astype(np.float32) for u in m]
        for t in self.transforms:
            rgb, m, i = t(rgb, m, i)
        return rgb, m, i


class RandomCrop(object):
    
    def __init__(self, size=112, min_area=0.2):
        self.size = size
        self.min_area = min_area
    
    def __call__(self, rgb, m, i):
        assert rgb[0].size == m[0].shape[1::-1] == i.size

        # consistent crop between rgb and m
        w, h = rgb[0].size
        area = w * h
        out_w, out_h = float('inf'), float('inf')
        while out_w > w or out_h > h:
            target_area = random.uniform(self.min_area, 1.0) * area
            aspect_ratio = random.uniform(3. / 4., 4. / 3.)
            out_w = int(round(math.sqrt(target_area * aspect_ratio)))
            out_h = int(round(math.sqrt(target_area / aspect_ratio)))
        x1 = random.randint(0, w - out_w)
        y1 = random.randint(0, h - out_h)

        rgb = [u.crop((x1, y1, x1 + out_w, y1 + out_h)) for u in rgb]
        rgb = [u.resize((self.size, self.size), Image.BILINEAR) for u in rgb]
        m = [u[y1:y1 + out_h, x1:x1 + out_w, :] for u in m]
        m = [cv2.resize(u, (self.size, self.size), interpolation=cv2.INTER_LINEAR) for u in m]

        # random crop on I
        w, h = i.size
        area = w * h
        out_w, out_h = float('inf'), float('inf')
        while out_w > w or out_h > h:
            target_area = random.uniform(self.min_area, 1.0) * area
            aspect_ratio = random.uniform(3. / 4., 4. / 3.)
            out_w = int(round(math.sqrt(target_area * aspect_ratio)))
            out_h = int(round(math.sqrt(target_area / aspect_ratio)))
        x1 = random.randint(0, w - out_w)
        y1 = random.randint(0, h - out_h)

        i = i.crop((x1, y1, x1 + out_w, y1 + out_h))
        i = i.resize((self.size, self.size), Image.BILINEAR)

        return rgb, m, i


class Rescale(object):

    def __init__(self, size=128):
        self.size = size
    
    def __call__(self, rgb, m, i):
        assert rgb[0].size == m[0].shape[1::-1] == i.size
        w, h = rgb[0].size
        scale = self.size / min(w, h)
        out_w, out_h = int(round(w * scale)), int(round(h * scale))
        rgb = [u.resize((out_w, out_h), Image.BILINEAR) for u in rgb]
        m = [cv2.resize(u, (out_w, out_h), interpolation=cv2.INTER_LINEAR) for u in m]
        i = i.resize((out_w, out_h), Image.BILINEAR)
        return rgb, m, i


class CenterCrop(object):

    def __init__(self, size=112):
        self.size = size
    
    def __call__(self, rgb, m, i):
        assert rgb[0].size == m[0].shape[1::-1] == i.size
        w, h = rgb[0].size
        assert min(w, h) >= self.size
        x1 = (w - self.size) // 2
        y1 = (h - self.size) // 2
        rgb = [u.crop((x1, y1, x1 + self.size, y1 + self.size)) for u in rgb]
        m = [u[y1:y1 + self.size, x1:x1 + self.size, :] for u in m]
        i = i.crop((x1, y1, x1 + self.size, y1 + self.size))
        return rgb, m, i


class RandomHFlip(object):
    
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, rgb, m, i):
        if random.random() < self.p:
            # consistent between m and i
            rgb = [u.transpose(Image.FLIP_LEFT_RIGHT) for u in rgb]
            m = [np.ascontiguousarray(u[:, ::-1]) for u in m]
        if random.random() < self.p:
            # random flip for I
            i = i.transpose(Image.FLIP_LEFT_RIGHT)
        return rgb, m, i


class GaussianBlur(object):

    def __init__(self, sigmas=[0.1, 2.0], p=0.5):
        self.sigmas = sigmas
        self.p = p
    
    def __call__(self, rgb, m, i):
        if random.random() < self.p:
            sigma = random.uniform(*self.sigmas)
            rgb = [u.filter(ImageFilter.GaussianBlur(radius=sigma)) for u in rgb]
        if random.random() < self.p:
            sigma = random.uniform(*self.sigmas)
            i = i.filter(ImageFilter.GaussianBlur(radius=sigma))
        return rgb, m, i


class ColorJitter(object):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p
    
    def __call__(self, rgb, m, i):
        if random.random() < self.p:
            brightness, contrast, saturation, hue = self._random_params()
            transforms = [
                lambda f: TF.adjust_brightness(f, brightness),
                lambda f: TF.adjust_contrast(f, contrast),
                lambda f: TF.adjust_saturation(f, saturation),
                lambda f: TF.adjust_hue(f, hue)]
            random.shuffle(transforms)
            for t in transforms:
                rgb = [t(u) for u in rgb]
        if random.random() < self.p:
            brightness, contrast, saturation, hue = self._random_params()
            transforms = [
                lambda f: TF.adjust_brightness(f, brightness),
                lambda f: TF.adjust_contrast(f, contrast),
                lambda f: TF.adjust_saturation(f, saturation),
                lambda f: TF.adjust_hue(f, hue)]
            random.shuffle(transforms)
            for t in transforms:
                i = t(i)
        return rgb, m, i

    def _random_params(self):
        brightness = random.uniform(
            max(0, 1 - self.brightness), 1 + self.brightness)
        contrast = random.uniform(
            max(0, 1 - self.contrast), 1 + self.contrast)
        saturation = random.uniform(
            max(0, 1 - self.saturation), 1 + self.saturation)
        hue = random.uniform(-self.hue, self.hue)
        return brightness, contrast, saturation, hue


class RandomGray(object):

    def __init__(self, p=0.2):
        self.p = p
    
    def __call__(self, rgb, m, i):
        if random.random() < self.p:
            rgb = [u.convert('L').convert('RGB') for u in rgb]
        if random.random() < self.p:
            i = i.convert('L').convert('RGB')
        return rgb, m, i


class ToTensor(object):

    def __call__(self, rgb, m, i):
        rgb = torch.stack([TF.to_tensor(u) for u in rgb], dim=1)
        m = torch.stack([torch.from_numpy(u) for u in m]).permute(3, 0, 1, 2).float()
        i = TF.to_tensor(i)
        return rgb, m, i


class Normalize(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
    
    def __call__(self, rgb, m, i):
        rgb, m, i = rgb.clone(), m.clone(), i.clone()
        rgb.clamp_(0, 1); m.clamp_(-10, 10); i.clamp_(0, 1)
        if not isinstance(self.mean, torch.Tensor):
            self.mean = rgb.new_tensor(self.mean).view(-1)
        if not isinstance(self.std, torch.Tensor):
            self.std = rgb.new_tensor(self.std).view(-1)
        rgb.sub_(self.mean.view(-1, 1, 1, 1)).div_(self.std.view(-1, 1, 1, 1))
        m.div_(10.)
        i.sub_(self.mean.view(-1, 1, 1)).div_(self.std.view(-1, 1, 1))
        return rgb, m, i


class MCompose(object):

    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, m):
        if m[0].dtype != np.float32:
            m = [u.astype(np.float32) for u in m]
        for t in self.transforms:
            m = t(m)
        return m


class MRandomCrop(object):
    
    def __init__(self, size=112, min_area=0.2):
        self.size = size
        self.min_area = min_area
    
    def __call__(self, m):
        h, w = m[0].shape[:2]
        area = w * h
        out_w, out_h = float('inf'), float('inf')
        while out_w > w or out_h > h:
            target_area = random.uniform(self.min_area, 1.0) * area
            aspect_ratio = random.uniform(3. / 4., 4. / 3.)
            out_w = int(round(math.sqrt(target_area * aspect_ratio)))
            out_h = int(round(math.sqrt(target_area / aspect_ratio)))
        x1 = random.randint(0, w - out_w)
        y1 = random.randint(0, h - out_h)
        m = [u[y1:y1 + out_h, x1:x1 + out_w, :] for u in m]
        m = [cv2.resize(u, (self.size, self.size), interpolation=cv2.INTER_LINEAR) for u in m]
        return m


class MRescale(object):
    
    def __init__(self, size=128):
        self.size = size
    
    def __call__(self, m):
        h, w = m[0].shape[:2]
        scale = self.size / min(w, h)
        out_w, out_h = int(round(w * scale)), int(round(h * scale))
        m = [cv2.resize(u, (out_w, out_h), interpolation=cv2.INTER_LINEAR) for u in m]
        return m


class MCenterCrop(object):
    
    def __init__(self, size=112):
        self.size = size
    
    def __call__(self, m):
        h, w = m[0].shape[:2]
        assert min(w, h) >= self.size
        x1 = (w - self.size) // 2
        y1 = (h - self.size) // 2
        m = [u[y1:y1 + self.size, x1:x1 + self.size, :] for u in m]
        return m


class MRandomHFlip(object):
    
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, m):
        if random.random() < self.p:
            m = [np.ascontiguousarray(u[:, ::-1]) for u in m]
        return m


class MToTensor(object):
    
    def __call__(self, m):
        m = torch.stack([torch.from_numpy(u) for u in m]).permute(3, 0, 1, 2).float()
        return m


class MNormalize(object):
    
    def __call__(self, m):
        m = torch.clamp(m, -10, 10)
        return m.div_(10.)

class VCompose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, item):
        for t in self.transforms:
            item = t(item)
        return item

class VRandomRotation(object):

    def __init__(self, angle=10):
        self.angle = angle

    def __call__(self, vclip):
        angle = random.uniform(-self.angle, self.angle)
        vclip = [u.rotate(angle) for u in vclip]
        return vclip

class VRandomCrop(object):

    def __init__(self, size=112, min_area=0.2):
        self.size = size
        self.min_area = min_area

    def __call__(self, vclip):
        w, h = vclip[0].size
        area = w * h
        for _ in range(100):
            target_area = random.uniform(self.min_area, 1.0) * area
            aspect_ratio = random.uniform(3. / 4., 4. / 3.)
            out_w = int(round(math.sqrt(target_area * aspect_ratio)))
            out_h = int(round(math.sqrt(target_area / aspect_ratio)))
            if out_w <= w and out_h <= h:
                break
        else:
            out_w = w
            out_h = h
        x1 = random.randint(0, w - out_w)
        y1 = random.randint(0, h - out_h)
        vclip = [u.crop((x1, y1, x1 + out_w, y1 + out_h)) for u in vclip]
        vclip = [u.resize((self.size, self.size), Image.BILINEAR) for u in vclip]
        return vclip

class VRescale(object):

    def __init__(self, size=256):
        self.size = size

    def __call__(self, vclip):
        w, h = vclip[0].size
        scale = self.size / min(w, h)
        out_w, out_h = int(round(w * scale)), int(round(h * scale))
        vclip = [u.resize((out_w, out_h), Image.BILINEAR) for u in vclip]
        return vclip

class VCenterCrop(object):

    def __init__(self, size=112):
        self.size = size

    def __call__(self, vclip):
        w, h = vclip[0].size
        assert min(w, h) >= self.size
        x1 = (w - self.size) // 2
        y1 = (h - self.size) // 2
        vclip = [u.crop((x1, y1, x1 + self.size, y1 + self.size)) for u in vclip]
        return vclip

class VRandomHFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, vclip):
        if random.random() < self.p:
            vclip = [u.transpose(Image.FLIP_LEFT_RIGHT) for u in vclip]
        return vclip

class VGaussianBlur(object):

    def __init__(self, sigmas=[0.1, 2.0], p=0.5):
        self.sigmas = sigmas
        self.p = p

    def __call__(self, vclip):
        if random.random() < self.p:
            sigma = random.uniform(*self.sigmas)
            vclip = [u.filter(ImageFilter.GaussianBlur(radius=sigma)) for u in vclip]
        return vclip

class VColorJitter(object):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p

    def __call__(self, vclip):
        if random.random() < self.p:
            brightness, contrast, saturation, hue = self._random_params()
            transforms = [
                lambda f: TF.adjust_brightness(f, brightness),
                lambda f: TF.adjust_contrast(f, contrast),
                lambda f: TF.adjust_saturation(f, saturation),
                lambda f: TF.adjust_hue(f, hue)]
            random.shuffle(transforms)
            for t in transforms:
                vclip = [t(u) for u in vclip]
        return vclip

    def _random_params(self):
        brightness = random.uniform(
            max(0, 1 - self.brightness), 1 + self.brightness)
        contrast = random.uniform(
            max(0, 1 - self.contrast), 1 + self.contrast)
        saturation = random.uniform(
            max(0, 1 - self.saturation), 1 + self.saturation)
        hue = random.uniform(-self.hue, self.hue)
        return brightness, contrast, saturation, hue

class VRandomGray(object):

    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, vclip):
        if random.random() < self.p:
            vclip = [u.convert('L').convert('RGB') for u in vclip]
        return vclip


class VToTensor(object):

    def __call__(self, vclip):
        vclip = torch.stack([TF.to_tensor(u) for u in vclip], dim=1)
        return vclip

class VNormalize(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, vclip):
        assert vclip.min() > -0.1 and vclip.max() < 1.1, \
            'vclip values should be in [0, 1]'
        vclip = vclip.clone()
        if not isinstance(self.mean, torch.Tensor):
            self.mean = vclip.new_tensor(self.mean).view(-1, 1, 1, 1)
        if not isinstance(self.std, torch.Tensor):
            self.std = vclip.new_tensor(self.std).view(-1, 1, 1, 1)
        vclip.sub_(self.mean).div_(self.std)
        return vclip
