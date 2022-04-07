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
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from threading import Thread

import dataset
import models
import dataset.transforms as T
from cfg.ucf_config import cfg

class Net(nn.Module):

    def __init__(self,
                 rgb_backbone='resnet26_2p1d',
                 m_backbone='resnet10_3d',
                 i_backbone='resnet10_2d',
                 hidden_dim=512,
                 src_numel=2 * 7 * 7,
                 tgt_numel=1 * 7 * 7,
                 num_encoder_layers=2,
                 num_decoder_layers=4):
        super(Net, self).__init__()
        self.src_numel = src_numel
        self.tgt_numel = tgt_numel
        self.rgb_backbone = getattr(models, rgb_backbone)(
            num_classes=None, last_pool=False)
        if 'resnet' in m_backbone:
            self.m_backbone = getattr(models, m_backbone)(
                num_classes=None, last_pool=False, inplanes=2, first_stride=1)
        else:
            self.m_backbone = getattr(models, m_backbone)(
                num_classes=None, last_pool=False, inplanes=2)
        self.i_backbone = getattr(models, i_backbone)(
            num_classes=None, last_pool=True)

        self.fc = nn.Linear(self.rgb_backbone.out_planes, hidden_dim, bias=False)
        self.k_embedding = nn.Embedding(src_numel, hidden_dim)
        self.q_embedding = nn.Embedding(tgt_numel, hidden_dim)
        self.transformer = models.Transformer(
            hidden_dim=hidden_dim,
            num_heads=8,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=0.1)
        
        self.mq_head = nn.Sequential(
            nn.Conv1d(hidden_dim, 2048, 1),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Conv1d(2048, 512, 1, bias=False),
            nn.BatchNorm1d(512))
        self.mk_head = nn.Sequential(
            nn.Conv1d(self.m_backbone.out_planes, 2048, 1),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Conv1d(2048, 512, 1, bias=False),
            nn.BatchNorm1d(512))
        self.iq_head = nn.Sequential(
            nn.Conv1d(self.rgb_backbone.out_planes, 2048, 1),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Conv1d(2048, 512, 1, bias=False),
            nn.BatchNorm1d(512))
        self.ik_head = nn.Sequential(
            nn.Conv1d(self.i_backbone.out_planes, 2048, 1),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Conv1d(2048, 512, 1, bias=False),
            nn.BatchNorm1d(512))
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, rgb, m=None, i=None):
        if m is None or i is None:
            return self.rgb_backbone(rgb)
        rgb = self._modified_backbone(rgb)

        # predicted motion features
        b, c, t, h, w = rgb.size()
        memory = rgb.permute(2, 3, 4, 0, 1).reshape(-1, b, c)  # [L, B, C]
        memory = self.fc(memory)
        k_pos = self.k_embedding(torch.arange(self.src_numel, device=rgb.device))
        k_pos = k_pos.unsqueeze(1).expand(k_pos.size(0), b, k_pos.size(1))
        q_pos = self.q_embedding(torch.arange(self.tgt_numel, device=rgb.device))
        q_pos = q_pos.unsqueeze(1).expand(q_pos.size(0), b, q_pos.size(1))
        mq = self.transformer(memory, q_pos, k_pos)
        mq = mq.permute(1, 0, 2).reshape(-1, mq.size(-1))
        mq = self.mq_head(mq.unsqueeze(-1)).squeeze(-1)
        mq = F.normalize(mq, p=2, dim=1)

        # target motion features
        mk = self.m_backbone(m)
        mk = mk.permute(0, 2, 3, 4, 1).reshape(-1, mk.size(1))
        mk = self.mk_head(mk.unsqueeze(-1)).squeeze(-1)
        mk = F.normalize(mk, p=2, dim=1)

        # predicted i-frame features
        iq = rgb.mean(dim=(2, 3, 4))
        iq = self.iq_head(iq.unsqueeze(-1)).squeeze(-1)
        iq = F.normalize(iq, p=2, dim=1)

        # target i-frame features
        ik = self.i_backbone(i)
        ik = self.ik_head(ik.unsqueeze(-1)).squeeze(-1)
        ik = F.normalize(ik, p=2, dim=1)

        return mq, mk, iq, ik
    
    def _modified_backbone(self, x):
        net = self.rgb_backbone
        if 'ResNet' in net.__class__.__name__:
            if isinstance(net, models.ResNet2p1d):
                x = net.conv1_s(x)
                x = net.bn1_s(x)
                x = net.relu(x)
                x = net.conv1_t(x)
                x = net.bn1_t(x)
                x = net.relu(x)
                x = net.maxpool(x)
            else:
                x = net.conv1(x)
                x = net.bn1(x)
                x = net.relu(x)
                x = net.maxpool(x)
            x = net.layer1(x)
            x = net.layer2(x)
            x = net.layer3(x)
            x3 = torch.cat([x[:, :, 0::2], x[:, :, 1::2]], dim=1)
            x = net.layer4(x)
            x = F.interpolate(x, x3.shape[2:], mode='nearest') + x3
            return x
        else:
            return net(x)


def collate_fn(batch):
    clips, m_clips, i_frames = zip(*batch)
    clips = torch.stack(clips, dim=0)
    m_clips = torch.stack(m_clips, dim=1).flatten(0, 1)
    i_frames = torch.stack(i_frames, dim=0).unsqueeze(2)
    return clips, m_clips, i_frames

def train(**kwargs):
    cfg.update(**kwargs)
    cfg.parse()
    cfg.gpus_per_machine = torch.cuda.device_count()
    cfg.world_size = cfg.gpus_per_machine
    mp.spawn(worker, nprocs=cfg.gpus_per_machine, args=(cfg, ))
    return cfg

def worker(gpu, cfg):
    cfg.gpu = gpu
    cfg.rank = gpu

    if cfg.rank == 0:
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
    
    torch.cuda.set_device(gpu)
    cudnn.benchmark = True
    dist.init_process_group(backend='nccl', world_size=cfg.world_size, rank=cfg.rank)

    train_transforms = T.Compose([
        T.RandomCrop(size=cfg.crop_size, min_area=cfg.min_area),
        T.RandomHFlip(p=0.5),
        T.GaussianBlur(sigmas=[0.1, 2.0], p=0.5),
        T.ColorJitter(p=0.5),
        T.RandomGray(p=0.2),
        T.ToTensor(),
        T.Normalize(mean=cfg.mean, std=cfg.std)])
    train_neg_transforms = T.MCompose([
        T.MRandomCrop(size=cfg.crop_size, min_area=cfg.min_area),
        T.MRandomHFlip(p=0.5),
        T.MToTensor(),
        T.MNormalize()])
    train_data = dataset.MotionPredictionDataset(
        root_dir=cfg.train_root,
        list_file=cfg.train_list,
        clip_frames=cfg.clip_frames,
        clip_stride=cfg.clip_stride,
        clip_size=cfg.crop_size,
        future_frames=cfg.future_frames,
        future_interval=cfg.future_interval,
        neg_num=cfg.neg_num,
        neg_interval=cfg.neg_interval,
        transforms=train_transforms,
        neg_transforms=train_neg_transforms)
    train_sampler = dataset.InfiniteSampler(
        train_data,
        num_replicas=cfg.world_size,
        rank=cfg.rank,
        shuffle=True,
        seed=cfg.sampler_seed,
        batch_size=cfg.train_batch_size)
    train_loader = DataLoader(
        train_data,
        batch_size=cfg.train_batch_size,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False)
    
    train_iter = iter(train_loader)
    train_steps = train_sampler.length // cfg.train_batch_size

    val_transforms = T.Compose([
        T.Rescale(size=cfg.scale_size),
        T.CenterCrop(size=cfg.crop_size),
        T.ToTensor(),
        T.Normalize(mean=cfg.mean, std=cfg.std)])
    val_neg_transforms = T.MCompose([
        T.MRescale(size=cfg.scale_size),
        T.MCenterCrop(size=cfg.crop_size),
        T.MToTensor(),
        T.MNormalize()])
    val_data = dataset.MotionPredictionDataset(
        root_dir=cfg.val_root,
        list_file=cfg.val_list,
        clip_frames=cfg.clip_frames,
        clip_stride=cfg.clip_stride,
        clip_size=cfg.crop_size,
        future_frames=cfg.future_frames,
        future_interval=cfg.future_interval,
        neg_num=cfg.neg_num,
        neg_interval=cfg.neg_interval,
        transforms=val_transforms,
        neg_transforms=val_neg_transforms)
    val_sampler = dataset.InfiniteSampler(
        val_data,
        num_replicas=cfg.world_size,
        rank=cfg.rank,
        shuffle=False,
        seed=cfg.sampler_seed,
        batch_size=cfg.val_batch_size)
    val_loader = DataLoader(
        val_data,
        batch_size=cfg.val_batch_size,
        sampler=val_sampler,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False)
    
    val_iter = iter(val_loader)
    val_steps = val_sampler.length // cfg.val_batch_size

    # deduce encoder and decoder sequence lengths
    if cfg.rgb_backbone.endswith('_2d3d'):
        src_t = int(math.ceil(cfg.clip_frames / 4))
    else:
        src_t = int(math.ceil(cfg.clip_frames / 8))
    tgt_t = int(math.ceil(cfg.future_frames / 8))
    src_s = int(math.ceil(cfg.crop_size / 16))
    tgt_s = int(math.ceil(cfg.crop_size / 16))

    net = Net(
        rgb_backbone=cfg.rgb_backbone,
        m_backbone=cfg.m_backbone,
        i_backbone=cfg.i_backbone,
        hidden_dim=cfg.hidden_dim,
        src_numel=src_t * src_s * src_s,
        tgt_numel=tgt_t * tgt_s * tgt_s,
        num_encoder_layers=cfg.num_encoder_layers,
        num_decoder_layers=cfg.num_decoder_layers)

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

    net = nn.SyncBatchNorm.convert_sync_batchnorm(net).to(gpu)
    net = DistributedDataParallel(net, device_ids=[gpu], output_device=gpu)
    criterion = nn.CrossEntropyLoss().to(gpu)

    cfg.base_lr *= cfg.world_size * cfg.train_batch_size
    cfg.warmup_lr *= cfg.world_size * cfg.train_batch_size
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=cfg.base_lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        dampening=cfg.dampening,
        nesterov=cfg.nesterov)
    
    # initialize buffers for communication
    train_mfeats, train_mlabels = None, None
    train_ifeats, train_ilabels = None, None
    val_mfeats, val_mlabels = None, None
    val_ifeats, val_ilabels = None, None
    
    total_steps = (cfg.num_epochs - start_epoch) * (train_steps + val_steps)
    start_time = time.time()

    for epoch in range(start_epoch + 1, cfg.num_epochs + 1):
        # train on one epoch
        net.train()
        if cfg.rank == 0:
            train_metrics = torch.zeros(7, device=gpu)
        
        for step in range(1, train_steps + 1):
            # compute learning rate for this epoch/step
            if epoch <= cfg.warmup_epochs:
                ratio = (step + (epoch - 1) * train_steps) / \
                    (cfg.warmup_epochs * train_steps)
                lr = cfg.warmup_lr + (cfg.base_lr - cfg.warmup_lr) * ratio
            else:
                ratio = (epoch - cfg.warmup_epochs) / \
                    (cfg.num_epochs - cfg.warmup_epochs)
                lr = cfg.base_lr * (math.cos(math.pi * ratio) + 1.0) * 0.5
            
            # update learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            batch = next(train_iter)
            batch = [u.to(gpu, non_blocking=True) for u in batch]
            clips, m_clips, i_frames = batch

            # run forward pass
            mq, mk, iq, ik = net(clips, m_clips, i_frames)

            # gather features from all other ranks
            if train_mfeats is None:
                train_mfeats = [torch.zeros_like(mk) for _ in range(cfg.world_size)]
                train_mlabels = torch.arange(
                    cfg.rank * len(mk),
                    cfg.rank * len(mk) + len(mq)).to(gpu)
                train_ifeats = [torch.zeros_like(ik) for _ in range(cfg.world_size)]
                train_ilabels = torch.arange(
                    cfg.rank * len(ik),
                    cfg.rank * len(ik) + len(iq)).to(gpu)
            
            dist.all_gather(train_mfeats, mk.data)
            dist.all_gather(train_ifeats, ik.data)
            all_mk = torch.cat(train_mfeats, dim=0)
            all_ik = torch.cat(train_ifeats, dim=0)
            all_mk.requires_grad_(True)
            all_ik.requires_grad_(True)

            # compute similarity matrices and loss
            logits_m = torch.mm(mq, all_mk.T)
            logits_i = torch.mm(iq, all_ik.T)

            logits_m = logits_m / cfg.temperature
            logits_i = logits_i / cfg.temperature
            loss_m = criterion(logits_m, train_mlabels)
            loss_i = criterion(logits_i, train_ilabels)
            loss = loss_m + loss_i

            # compute gradients and run optimization
            [mq_grads, iq_grads, all_mk_grads, all_ik_grads] = autograd.grad(
                loss, [mq, iq, all_mk, all_ik])
            all_mk_grads = torch.chunk(all_mk_grads, cfg.world_size, dim=0)
            all_ik_grads = torch.chunk(all_ik_grads, cfg.world_size, dim=0)
            for rank in range(cfg.world_size):
                dist.reduce(all_mk_grads[rank], dst=rank)
                dist.reduce(all_ik_grads[rank], dst=rank)
            
            variables = torch.cat([mq, mk, iq, ik], dim=0)
            grads = torch.cat([
                mq_grads,
                all_mk_grads[cfg.rank].div_(cfg.world_size),
                iq_grads,
                all_ik_grads[cfg.rank].div_(cfg.world_size)])
            
            optimizer.zero_grad()
            variables.backward(grads)
            clip_grad_norm_(net.parameters(), cfg.clip_gradient)
            optimizer.step()

            with torch.no_grad():
                # compute top-k metrics
                _, pred_m = logits_m.topk(5, dim=1)
                top1_m = (pred_m[:, :1] == train_mlabels.view(-1, 1)).any(dim=1).float().mean()
                top5_m = (pred_m[:, :5] == train_mlabels.view(-1, 1)).any(dim=1).float().mean()
                _, pred_i = logits_i.topk(5, dim=1)
                top1_i = (pred_i[:, :1] == train_ilabels.view(-1, 1)).any(dim=1).float().mean()
                top5_i = (pred_i[:, :5] == train_ilabels.view(-1, 1)).any(dim=1).float().mean()
            
            metrics = torch.stack([loss_m.data, top1_m, top5_m, loss_i.data, top1_i, top5_i])
            dist.reduce(metrics, dst=0)
            
            if cfg.rank == 0:
                metrics /= cfg.world_size
                train_metrics[:6] += metrics.data
                train_metrics[-1] += 1

                if step == 1 or step == train_steps or step % cfg.log_frequency == 0:
                    metrics[[1, 2, 4, 5]] *= 100
                    elapsed_steps = (epoch - start_epoch) * (train_steps + val_steps) + step
                    elapsed_time = time.time() - start_time
                    eta = (total_steps - elapsed_steps) * (elapsed_time / elapsed_steps)
                    logging.info(
                        'ETA: {}s Epoch: {}/{} Step: {}/{} [M] Loss: {:.5f} Top1: {:.3f} Top5: {:.3f} ' \
                        '[I] Loss: {:.5f} Top1: {:.3f} Top5: {:.3f} lr: {:.6f}'.format(
                        int(eta), epoch, cfg.num_epochs, step, train_steps,
                        *metrics.tolist(), optimizer.param_groups[0]['lr']))
        
        # logging
        if cfg.rank == 0:
            train_metrics[:6] /= train_metrics[-1]
            train_metrics[[1, 2, 4, 5]] *= 100
            logging.info('Training on Epoch {}/{} completed!'.format(epoch, cfg.num_epochs))
            logging.info(
                '**** [M] Loss: {:.5f} Top1: {:.3f} Top5: {:.3f} [I] Loss: {:.5f} ' \
                'Top1: {:.3f} Top5: {:.3f}'.format(*train_metrics[:-1].tolist()))
        
        # save checkpoint asynchronously
        if cfg.rank == 0:
            save_key = osp.join(cfg.log_dir, 'latest.pth')
            Thread(target=torch.save, args=(
                    net.module.state_dict(), save_key)).start()
            if epoch % cfg.save_frequency == 0 or epoch == cfg.num_epochs:
                save_key = osp.join(cfg.log_dir, 'epoch_{}.pth'.format(epoch))
                Thread(target=torch.save, args=(
                        net.module.state_dict(), save_key)).start()
        
        if epoch % cfg.val_frequency != 0:
            # skip validation
            continue

        # validate on one epoch
        net.eval()
        if cfg.rank == 0:
            val_metrics = torch.zeros(7, device=gpu)
        
        for step in range(1, val_steps + 1):
            batch = next(val_iter)
            batch = [u.to(gpu, non_blocking=True) for u in batch]
            clips, m_clips, i_frames = batch

            with torch.no_grad():
                # run forward pass
                mq, mk, iq, ik = net(clips, m_clips, i_frames)

                # gather features from all other ranks
                if val_mfeats is None:
                    val_mfeats = [torch.zeros_like(mk) for _ in range(cfg.world_size)]
                    val_mlabels = torch.arange(
                        cfg.rank * len(mk),
                        cfg.rank * len(mk) + len(mq)).to(gpu)
                    val_ifeats = [torch.zeros_like(ik) for _ in range(cfg.world_size)]
                    val_ilabels = torch.arange(
                        cfg.rank * len(ik),
                        cfg.rank * len(ik) + len(iq)).to(gpu)
                
                dist.all_gather(val_mfeats, mk.data)
                dist.all_gather(val_ifeats, ik.data)
                all_mk = torch.cat(val_mfeats, dim=0)
                all_ik = torch.cat(val_ifeats, dim=0)
                all_mk.requires_grad_(True)
                all_ik.requires_grad_(True)

                # compute similarity matrices and loss
                logits_m = torch.mm(mq, all_mk.T)
                logits_i = torch.mm(iq, all_ik.T)

                logits_m = logits_m / cfg.temperature
                logits_i = logits_i / cfg.temperature
                loss_m = criterion(logits_m, val_mlabels)
                loss_i = criterion(logits_i, val_ilabels)
                loss = loss_m + loss_i

                _, pred_m = logits_m.topk(5, dim=1)
                top1_m = (pred_m[:, :1] == val_mlabels.view(-1, 1)).any(dim=1).float().mean()
                top5_m = (pred_m[:, :5] == val_mlabels.view(-1, 1)).any(dim=1).float().mean()
                _, pred_i = logits_i.topk(5, dim=1)
                top1_i = (pred_i[:, :1] == val_ilabels.view(-1, 1)).any(dim=1).float().mean()
                top5_i = (pred_i[:, :5] == val_ilabels.view(-1, 1)).any(dim=1).float().mean()
            
            metrics = torch.stack([loss_m.data, top1_m, top5_m, loss_i.data, top1_i, top5_i])
            dist.reduce(metrics, dst=0)

            if cfg.rank == 0:
                metrics /= cfg.world_size
                val_metrics[:6] += metrics.data
                val_metrics[-1] += 1
                
                if step == 1 or step == val_steps or step % cfg.log_frequency == 0:
                    metrics[[1, 2, 4, 5]] *= 100
                    logging.info(
                        'Val Epoch: {}/{} Step: {}/{} [M] Loss: {:.5f} Top1: {:.3f} Top5: {:.3f} ' \
                        '[I] Loss: {:.5f} Top1: {:.3f} Top5: {:.3f}'.format(
                        epoch, cfg.num_epochs, step, val_steps, *metrics.tolist()))
        
        # logging
        if cfg.rank == 0:
            val_metrics[:6] /= val_metrics[-1]
            val_metrics[[1, 2, 4, 5]] *= 100
            logging.info('Validation on Epoch {}/{} completed!'.format(epoch, cfg.num_epochs))
            logging.info(
                '**** [M] Loss: {:.5f} Top1: {:.3f} Top5: {:.3f} [I] Loss: {:.5f} ' \
                'Top1: {:.3f} Top5: {:.3f}'.format(*val_metrics[:6].tolist()))

    if cfg.rank == 0:
        logging.info('Congratulations! The training is completed!')
    
    # synchronize to finish some processes
    torch.cuda.synchronize()

if __name__ == '__main__':
    train()
