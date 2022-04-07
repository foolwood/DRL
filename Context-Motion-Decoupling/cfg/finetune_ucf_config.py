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
cfg.num_classes = 101
cfg.train_root = 'data/ucf101/UCF-101'
cfg.train_list = 'data/ucf101/ucfTrainTestlist/trainlist01.txt'
cfg.val_root = cfg.train_root
cfg.val_list = 'data/ucf101/ucfTrainTestlist/testlist01.txt'
cfg.clsid_file = 'data/ucf101/ucfTrainTestlist/classInd.txt'
cfg.clip_frames = 16
cfg.clip_stride = 4

cfg.scale_size = 128
cfg.crop_size = 112
cfg.min_area = 0.2
cfg.rotation = 10
cfg.multicrop = 10
cfg.mean = [0.485, 0.456, 0.406]
cfg.std = [0.229, 0.224, 0.225]
cfg.sampler_seed = 6666

cfg.train_batch_size = 8
cfg.val_batch_size = 8
cfg.num_workers = 2

# model
cfg.backbone = 'resnet26_2p1d' #['resnet26_2p1d', 'resnet26_3d', 'C3D']
cfg.dropout = 0.3
cfg.pretrain = 'exps/resnet26_2p1d_epoch120.pth' # pretrained model path

# optimizer
cfg.num_epochs = 120
cfg.warmup_epochs = 10
cfg.base_lr = 1.e-4
cfg.warmup_lr = 4.e-6
cfg.final_lr = 1.e-6
cfg.backbone_lr_mult = 1.0
cfg.momentum = 0.9
cfg.weight_decay = 3.e-3
cfg.dampening = 0.0
cfg.nesterov = True
cfg.clip_gradient = 15

# training
cfg.log_frequency = 200
cfg.save_frequency = 1

# checkpoint
cfg.resume_from = ''
cfg.strict = True

cfg.log_dir = 'exps/cmd_{0}_{1}/cmd_{0}_{1}_{2}'.format(
    cfg.dataset, cfg.backbone, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
cfg.log_level = logging.INFO
