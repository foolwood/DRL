# Copyright (C) Alibaba Group Holding Limited. 

import os
import os.path as osp
import sys
import time
import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.autograd as autograd
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from threading import Thread

import dataset
import models
import dataset.transforms as T
from cfg.finetune_ucf_config import cfg

def train():

    if not os.path.exists(cfg.log_dir):
        os.makedirs(cfg.log_dir)
    log_file = os.path.join(cfg.log_dir, 'log.txt')
    cfg.log_file = log_file
    logging.basicConfig(
        level=cfg.log_level,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(filename=log_file),
            logging.StreamHandler(stream=sys.stdout)])
    logging.info(cfg)
    
    cudnn.benchmark = True

    train_transforms = T.VCompose([
        T.VRandomRotation(angle=cfg.rotation),
        T.VRandomCrop(size=cfg.crop_size, min_area=cfg.min_area),
        T.VRandomHFlip(p=0.5),
        T.VGaussianBlur(sigmas=[0.1, 2.0], p=0.5),
        T.VColorJitter(p=0.5),
        T.VToTensor(),
        T.VNormalize(mean=cfg.mean, std=cfg.std)])
    train_data = dataset.VideoDataset(
        cfg.train_root,
        cfg.train_list,
        cfg.clsid_file,
        video_frames=cfg.clip_frames,
        video_stride=cfg.clip_stride,
        video_size=cfg.crop_size,
        training=True,
        transforms=train_transforms)
    train_loader = DataLoader(
        train_data,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True)
    train_steps = len(train_loader)

    val_transforms = T.VCompose([
        T.VRescale(size=cfg.scale_size),
        T.VCenterCrop(size=cfg.crop_size),
        T.VToTensor(),
        T.VNormalize(mean=cfg.mean, std=cfg.std)])
    val_data = dataset.VideoDataset(
        cfg.val_root,
        cfg.val_list,
        cfg.clsid_file,
        video_frames=cfg.clip_frames,
        video_stride=cfg.clip_stride,
        video_size=cfg.crop_size,
        multicrop=cfg.multicrop,
        training=False,
        transforms=val_transforms)
    val_loader = DataLoader(
        val_data,
        batch_size=cfg.val_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False)
    val_steps = len(val_loader)
 
    # model and criterion
    net = getattr(models, cfg.backbone)(
            num_classes=cfg.num_classes, dropout=cfg.dropout, last_pool=True).cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=-1).cuda()

    if cfg.pretrain:
        state_dict = torch.load(cfg.pretrain, map_location='cpu')
        new_state = {k.replace('rgb_backbone.', ''): v for k, v in state_dict.items() if 'rgb_backbone.' in k}
        status = net.load_state_dict(new_state, strict=False)
        logging.info('Load pretrained model from {}'.format(cfg.pretrain))
        if len(status.missing_keys) > 0:
            logging.info('Missing keys: {}'.format(status.missing_keys))
        if len(status.unexpected_keys) > 0:
            logging.info('Unexpected keys: {}'.format(status.unexpected_keys))

    start_epoch = 0
    if cfg.resume_from:
        if 'epoch' in os.path.basename(cfg.resume_from):
            start_epoch = int(os.path.basename(cfg.resume_from)[:-4].split('_')[-1])
        state_dict = torch.load(cfg.resume_from, map_location='cpu')
        status = net.load_state_dict(state_dict, strict=cfg.strict)
        logging.info('Resumed from {}'.format(cfg.resume_from))
        if len(status.missing_keys) > 0:
            logging.info('Missing keys: {}'.format(status.missing_keys))
        if len(status.unexpected_keys) > 0:
            logging.info('Unexpected keys: {}'.format(status.unexpected_keys))
     
    # optimizer
    cfg.base_lr *=  cfg.train_batch_size
    cfg.warmup_lr *= cfg.train_batch_size
    backbone_lr = cfg.base_lr * cfg.backbone_lr_mult
    backbone_params = filter(
        lambda p: p.requires_grad,
        [p for n, p in net.named_parameters() if not n.startswith('fc.')])
    fc_params = filter(
        lambda p: p.requires_grad,
        [p for n, p in net.named_parameters() if n.startswith('fc.')])
    param_groups = [
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': fc_params}]

    optimizer = optim.SGD(
        param_groups,
        lr=cfg.base_lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        dampening=cfg.dampening,
        nesterov=cfg.nesterov)

    for epoch in range(start_epoch + 1, cfg.num_epochs + 1):
        # train on one epoch
        net.train()
        train_metrics = torch.zeros(3).cuda()
        train_counter = 0.0
 
        for step, batch in enumerate(train_loader):
            # compute learning rate for this epoch/step
            if epoch <= cfg.warmup_epochs:
                ratio = (step + (epoch - 1) * train_steps) / \
                    (cfg.warmup_epochs * train_steps)
                lr = cfg.warmup_lr + (cfg.base_lr - cfg.warmup_lr) * ratio
            else:
                ratio = (epoch - cfg.warmup_epochs) / \
                    (cfg.num_epochs - cfg.warmup_epochs)
                lr = (cfg.base_lr - cfg.final_lr) * (math.cos(math.pi * ratio) + 1.0) * 0.5 + \
                    cfg.final_lr
            
            # update learning rate
            optimizer.param_groups[0]['lr'] = lr * cfg.backbone_lr_mult
            optimizer.param_groups[1]['lr'] = lr
 
            # read batch
            batch = [u.cuda(non_blocking=True) for u in batch]
            clips, labels, _ = batch

            # run forward pass
            logits = net(clips)
            loss = criterion(logits, labels)

            # optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred = logits.topk(5, dim=1)[1]
                top1 = (pred[:, :1] == labels.unsqueeze(1)).any(dim=1).float().mean()
                top5 = (pred[:, :5] == labels.unsqueeze(1)).any(dim=1).float().mean()

            metrics = torch.stack([loss.data, top1, top5])
            train_metrics += metrics.data
            train_counter += 1

            if step < 1 or step == train_steps - 1 or step % cfg.log_frequency == 0:
                metrics[1:] *= 100
                logging.info(
                    'Epoch: {}/{} Step: {}/{} Loss: {:.5f} Top1: {:.3f} Top5: {:.3f} ' \
                    'backbone lr: {:.6f} head lr: {:.6f}'.format(
                    epoch, cfg.num_epochs, step, train_steps,
                    *metrics.tolist(), optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']))
        
        # logging
        train_metrics /= train_counter
        train_metrics[1:] *= 100
        logging.info('[{}-train] Training on Epoch {}/{} completed!'.format(
            cfg.dataset, epoch, cfg.num_epochs))
        logging.info('**** Loss: {:.5f} Top1: {:.3f} Top5: {:.3f}'.format(
            *train_metrics.tolist()))

        # save checkpoint 
        save_key = osp.join(cfg.log_dir, 'latest.pth')
        Thread(target=torch.save, args=(
                net.state_dict(), save_key)).start()
        if epoch % cfg.save_frequency == 0 or epoch == cfg.num_epochs:
            save_key = osp.join(cfg.log_dir, 'epoch_{}.pth'.format(epoch))
            Thread(target=torch.save, args=(
                    net.state_dict(), save_key)).start()

        # validate on one epoch
        net.eval()
        val_logits, val_labels, val_indices = [], [], []
        for step, batch in enumerate(val_loader):
            batch = [u.cuda(non_blocking=True) for u in batch]
            clips, labels, indices = batch

            with torch.no_grad():
                b, n = clips.shape[:2]
                logits = net(clips.flatten(0, 1))
                logits = F.softmax(logits, dim=-1).view(b, n, -1)
                logits = logits.mean(dim=1)

                val_logits.append(logits.data)
                val_labels.append(labels)
                val_indices.append(indices)

             # logging
            if step == 0 or step == val_steps or \
                step % cfg.log_frequency == 0:
                logging.info('[{}-val] Step: {}/{}'.format(cfg.dataset, step, val_steps))

        val_logits = torch.cat(val_logits, dim=0)
        val_labels = torch.cat(val_labels, dim=0)
        val_indices = torch.cat(val_indices, dim=0)

        mask = (val_indices >= 0)
        val_logits = val_logits[mask]
        val_labels = val_labels[mask]

        pred = val_logits.topk(5, dim=1)[1]
        top1 = (pred[:, :1] == val_labels.unsqueeze(1)).any(dim=1).float().mean()
        top5 = (pred[:, :5] == val_labels.unsqueeze(1)).any(dim=1).float().mean()
 
        logging.info('Val Epoch: {}/{}: Top1: {:.3f} Top5: {:.3f}'.format(
                        epoch, cfg.num_epochs, top1 * 100., top5 * 100.))
        
    logging.info('Congratulations! The training is completed!')
    
    # synchronize to finish some processes
    torch.cuda.synchronize()

if __name__ == '__main__':
    train()
