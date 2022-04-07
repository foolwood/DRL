# Copyright (C) Alibaba Group Holding Limited. 

import os
import torch
import logging
import os.path as osp
from datetime import datetime

from ops import Config

__all__ = ['cfg']

cfg = Config(__name__='SSL-Context and Motion Decoupling')

# data
cfg.dataset = 'UCF-101'
cfg.train_root = 'data/ucf101/UCF-101_rawvideo'
cfg.train_list = 'data/ucf101/ucfTrainTestlist/trainlist01.txt'
cfg.val_root = cfg.train_root
cfg.val_list = 'data/ucf101/ucfTrainTestlist/testlist01.txt'

cfg.clip_frames = 16
cfg.clip_stride = 4
cfg.future_frames = 8
cfg.future_interval = 2
cfg.neg_num = 3
cfg.neg_interval = 0

cfg.scale_size = 128
cfg.crop_size = 112
cfg.min_area = 0.2
cfg.mean = [0.485, 0.456, 0.406]
cfg.std = [0.229, 0.224, 0.225]
cfg.sampler_seed = 6666

if torch.cuda.get_device_properties(0).total_memory > 3.2e10:
    cfg.train_batch_size = 16
    cfg.val_batch_size = 16
else:
    cfg.train_batch_size = 8
    cfg.val_batch_size = 8
cfg.num_workers = 8

# model
cfg.rgb_backbone = 'resnet26_2p1d' #['resnet26_2p1d', 'resnet26_3d', 'C3D']
cfg.m_backbone = 'resnet10_3d'
cfg.i_backbone = 'resnet10_2d'
cfg.hidden_dim = 512
cfg.num_encoder_layers = 2
cfg.num_decoder_layers = 4
cfg.temperature = 0.1

# optimizer
cfg.num_epochs = 120
cfg.warmup_epochs = 5
cfg.base_lr = 5e-4
cfg.warmup_lr = 5e-6
cfg.momentum = 0.9
cfg.weight_decay = 1.e-3
cfg.dampening = 0.0
cfg.nesterov = True
cfg.clip_gradient = 15

# training
cfg.log_frequency = 20
cfg.val_frequency = 1
cfg.save_frequency = 1

# checkpoint
cfg.resume_from = ''
cfg.strict = True

cfg.log_dir = 'exps/cmd_{0}_{1}/cmd_{0}_{1}_{2}'.format(
    cfg.dataset, cfg.rgb_backbone, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
cfg.log_level = logging.INFO
