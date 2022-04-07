# Copyright (C) Alibaba Group Holding Limited. 

import decord
decord.bridge.set_bridge('torch')
import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from decord import VideoReader, cpu
from PIL import Image
import coviar

__all__ = ['MotionPredictionDataset', 'VideoDataset']

def get_frame_types(filename):
    cmd = 'ffprobe -v error -show_frames {} | grep pict_type'.format(filename)
    types = os.popen(cmd).read().strip().split('\n')
    types = [u[10:] for u in types]
    types = [{'I': 0, 'P': 1, 'B': 2}[u] for u in types]
    return types

def get_imr_frames(filename, indices, frame_types, accumulate=False,
                   representation_type=0):
    # 0: I (I-frame), 1: M (motion vector), 2: R (residual)
    assert representation_type in [0, 1, 2]
    frame_types = np.array(frame_types)
    output = []
    for i in indices:
        gop_ind = (frame_types[:i + 1] == 0).sum() - 1
        frame_ind = i - np.where(frame_types[:i + 1] == 0)[0][-1]
        frame = coviar.load(
            filename, gop_ind, frame_ind, representation_type, accumulate)
        assert frame is not None
        output.append(frame)
    return output

class MotionPredictionDataset(Dataset):

    def __init__(self,
                 root_dir,
                 list_file,
                 clip_frames=16,
                 clip_stride=4,
                 clip_size=112,
                 future_frames=8,
                 future_interval=2,
                 neg_num=3,
                 neg_interval=0,
                 transforms=None,
                 neg_transforms=None):
        self.root_dir = root_dir
        self.list_file = list_file
        self.clip_frames = clip_frames
        self.clip_stride = clip_stride
        self.clip_size = clip_size
        self.future_frames = future_frames
        self.future_interval = future_interval
        self.neg_num = neg_num
        self.neg_interval = neg_interval
        self.transforms = transforms
        self.neg_transforms = neg_transforms
        self.clip_len = (clip_frames - 1) * clip_stride + 1
        self.min_frames = self.clip_len + future_interval + future_frames

        # load and parse list_file
        with open(self.list_file, 'r') as f:
            items = f.read().strip().split('\n')
        items = [t.split()[0] for t in items]

        # Remove too short videos
        # Loading frame number of videos, this may take a while...
        # You can avoid by pre-computing these numbers

        # bad exceptions
        exceptions = ['v_WallPushups_g18_c03']
        paths = []
        for t in items:
            if os.path.basename(t)[:-4] in exceptions:
                continue
            path = os.path.join(root_dir, t[:-4] + '.mp4')
            num_frame = coviar.get_num_frames(path)
            if num_frame >= self.min_frames:
                paths.append((path, num_frame))
        self.paths = paths

    def __getitem__(self, index):
        path, num_frame = self.paths[index]
        try:
            path_orig = path.replace('_rawvideo', '')[:-4] + '.avi'
            video = VideoReader(path_orig, ctx=cpu(0))

            # [RGB clip] sample a random clip with RGB data
            total = min(len(video), num_frame)
            init_frame = np.random.randint(0, total - self.min_frames + 1)
            indices = np.arange(init_frame, init_frame + self.clip_len, self.clip_stride)
            clip = video.get_batch(indices)
            clip = [Image.fromarray(u) for u in clip.numpy()]

            # [Motion clip] extract motion vectors of near future
            frame_types = np.array(get_frame_types(path), dtype=int)
            m_indices = np.arange(
                indices[-1] + self.future_interval + 1,
                indices[-1] + self.future_interval + self.future_frames + 1)
            m_clip = get_imr_frames(path, m_indices, frame_types, False, 1)

            # [Neg-Motion clip] extract negative motion clips
            n_candidates = []
            if (m_indices[0] - self.neg_interval - self.future_frames) >= 0:
                n_candidates.append(np.arange(
                    0,
                    m_indices[0] - self.neg_interval - self.future_frames + 1))
            if (m_indices[-1] + self.neg_interval + self.future_frames) <= total - 1:
                n_candidates.append(np.arange(
                    m_indices[-1] + self.neg_interval + 1,
                    total - self.neg_interval - self.future_frames + 1))
            assert len(n_candidates) > 0, 'error at "len(n_candidates) > 0"'
            n_candidates = np.concatenate(n_candidates)
            assert len(n_candidates) > 0, 'error at "len(n_candidates) > 0"'

            init_frames = np.random.choice(
                n_candidates, self.neg_num, replace=len(n_candidates) < self.neg_num)
            n_indices = np.arange(0, self.future_frames)
            n_indices = init_frames[:, None] + n_indices[None, :]
            n_clips = [get_imr_frames(
                path, u, frame_types, False, 1) for u in n_indices]
            
            # [I frame] randomly sample a surrounding I-frame
            i_candidates = np.where(frame_types == 0)[0]
            i_candidates = i_candidates[(
                i_candidates >= indices[0] - 5) & (i_candidates <= indices[-1] + 5)]
            assert len(i_candidates) > 0, 'error at "len(i_candidates) > 0"'
            i_index = np.random.choice(i_candidates, 1)
            i_frame = get_imr_frames(path, i_index, frame_types, False, 0)[0]
            i_frame = Image.fromarray(i_frame)

            # data augmentation
            clip, m_clip, i_frame = self.transforms(clip, m_clip, i_frame)
            n_clips = [self.neg_transforms(u) for u in n_clips]
            m_clips = torch.stack([m_clip] + n_clips, dim=0)

            return clip, m_clips, i_frame

        except Exception as e:
            print(e)
            print('Loading video {} failed, return zeros'.format(path))
            sys.stdout.flush()
            return self._zero_output()
    
    def __len__(self):
        return len(self.paths)
    
    def _zero_output(self):
        f1, f2, n, s = self.clip_frames, self.future_frames, 1 + self.neg_num, self.clip_size
        clip = torch.zeros(3, f1, s, s)
        m_clips = torch.zeros(n, 2, f2, s, s)
        i_frame = torch.zeros(3, s, s)
        return clip, m_clips, i_frame

class VideoDataset(Dataset):
    def __init__(self,
                 root_dir,
                 list_file,
                 clsid_file,
                 video_frames=16,
                 video_stride=4,
                 video_size=112,
                 multicrop=10,
                 training=True,
                 transforms=None):
        
        self.root_dir = root_dir
        self.video_frames = video_frames
        self.video_stride = video_stride
        self.video_len = (video_frames - 1) * video_stride + 1
        self.video_size = video_size
        self.multicrop = multicrop
        self.training = training
        self.transforms = transforms

        # load and parse list_file
        with open(list_file, 'r') as f:
            items = f.read().strip().split('\n')

        entries = []
        if self.training:
            for t in items:
                video_path, clsid = t.split()
                entries.append((os.path.join(root_dir, video_path), int(clsid) - 1))
        else:
            with open(clsid_file, 'r') as f:
                clsitems = f.read().strip().split('\n')
            clsmaps = {} 
            for t in clsitems:
                clsid, name = t.split()
                clsmaps[name] = int(clsid) - 1

            for t in items:
                name = t.split('/')[0]
                entries.append((os.path.join(root_dir, t), clsmaps[name]))

        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        path, clsid = self.entries[index]
        try:
            video = VideoReader(path, ctx=cpu(0))
            total = len(video)
            if self.training:
                init_frame = np.random.choice(max(1, total - self.video_len + 1))
                indices = np.arange(init_frame, init_frame + self.video_len, self.video_stride)
                indices[indices >= total] = 0

                # extract video frames
                vclips = video.get_batch(indices)
                vclips = [Image.fromarray(u) for u in vclips.numpy()]
                if self.transforms is not None:
                    vclips = self.transforms(vclips)
            else:
                if total <= self.video_len:
                    init_frames = np.zeros(self.multicrop, dtype=int)
                else:
                    init_frames = np.linspace(0, total - self.video_len, self.multicrop + 1)
                    init_frames = ((init_frames[1:] + init_frames[:-1]) / 2.).astype(int)
                indices = np.arange(0, self.video_len, self.video_stride)

                indices = (init_frames[:, None] + indices[None, :]).reshape(-1)
                indices[indices >= total] = 0

                # extract video frames
                vclips = video.get_batch(indices).chunk(self.multicrop, dim=0)
                vclips = [self.transforms([Image.fromarray(f) for f in u.numpy()])
                            for u in vclips]
                vclips = torch.stack(vclips, dim=0)

            return vclips, clsid, index
      
        except Exception as e:
            print(e)
            print('Loading video {} failed, return zeros'.format(path))
            sys.stdout.flush()
            return self._zero_output()

    def _zero_output(self):
        if self.training:
            vclips = torch.zeros(3, self.video_frames, self.video_size, self.video_size)
        else:
            vclips = torch.zeros(self.multicrop, 3, self.video_frames,
                             self.video_size, self.video_size)
        return vclips, -1, -1
