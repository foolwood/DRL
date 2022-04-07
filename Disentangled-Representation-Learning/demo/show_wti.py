from os.path import join,dirname,abspath
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import torch

from tvr.models.modeling import DRL
from tvr.models.tokenization_clip import SimpleTokenizer

import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode


def show_single_pair(t_feat, v_feat, t_tokens, v_frames, save_fig=None, dense_text_fc_weight=None,
                     dense_vision_fc_weight=None, i_min=None, i_max=None):
    with torch.no_grad():
        if dense_text_fc_weight:
            t_weight = torch.softmax(dense_text_fc_weight(t_feat).squeeze(1), dim=0)
        else:
            t_weight = torch.ones(t_feat.shape[0]) / t_feat.shape[0]

        if dense_vision_fc_weight:
            v_weight = torch.softmax(dense_vision_fc_weight(v_feat).squeeze(1), dim=0)
        else:
            v_weight = torch.ones(v_feat.shape[0]) / v_feat.shape[0]

    t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)
    v_feat = v_feat / v_feat.norm(dim=-1, keepdim=True)
    corr = torch.einsum('tc,vc->tv', [t_feat, v_feat])  # txv
    t_max = corr.max(dim=1)[0]
    v_max = corr.max(dim=0)[0]

    t2v = (t_weight.T * t_max).sum()
    v2t = (v_weight * v_max).sum()

    n_t = len(t_feat)
    n_v = len(v_feat)

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(1, 1, 1)

    cmap = plt.cm.rainbow
    if (i_min is not None) and (i_max is not None):
        norm = matplotlib.colors.Normalize(vmin=i_min, vmax=i_max)
    else:
        norm = matplotlib.colors.Normalize(vmin=corr.min(), vmax=corr.max())
    t_w_norm = matplotlib.colors.Normalize(vmin=t_weight[1:-1].min(), vmax=t_weight[1:-1].max())
    v_w_norm = matplotlib.colors.Normalize(vmin=v_weight.min(), vmax=v_weight.max())

    t_loc = [i / n_t for i in range(n_t)]
    # text_len = len(' '.join(t_tokens))
    # cum_len = np.cumsum([len(_t) for _t in t_tokens]).tolist()
    # t_loc = [(i+cum_len[i])/text_len for i in range(n_t)]

    for i in range(n_t):
        plt.text(t_loc[i], 1, t_tokens[i], fontsize=30, horizontalalignment='center', verticalalignment='bottom')
        plt.text(t_loc[i], 1 + 0.05, '%.3f' % (t_max[i].item()), fontsize=25, horizontalalignment='center',
                 verticalalignment='bottom')
        plt.text(t_loc[i], 1 + 0.08, 'x', fontsize=35, horizontalalignment='center',
                 verticalalignment='bottom')
        ax.scatter(t_loc[i], 1.04, s=1000, c=[cmap(norm(t_max[i].item()))])
        plt.text(t_loc[i], 1 + 0.11, '%.3f' % (t_weight[i].item()), fontsize=25, horizontalalignment='center',
                 verticalalignment='bottom')
        ax.scatter(t_loc[i], 1.15, s=1000, c=[cmap(t_w_norm(t_weight[i].item()))])

    plt.text(0.5, 1 + 0.17, '(%.3f + %.3f)/2 = %.3f' % (t2v.item(), v2t.item(), (t2v.item() + v2t.item()) / 2.0),
             fontsize=35, horizontalalignment='center',
             verticalalignment='bottom')

    for j in range(n_v):
        ax.add_artist(  # ax can be added image as artist.
            AnnotationBbox(
                OffsetImage(v_frames[j])
                , (j / n_v, 0.5)
                , frameon=False
                , box_alignment=(0.5, 1)
            )
        )
        plt.text(j / n_v, 0.5 - 0.12, '%.3f' % (v_max[j].item()), fontsize=25, horizontalalignment='center',
                 verticalalignment='top')
        plt.text(j / n_v, 0.5 - 0.15, "x", fontsize=35, horizontalalignment='center',
                 verticalalignment='top')
        ax.scatter(j / n_v, 0.4, s=1000, c=[cmap(norm(v_max[j].item()))])
        plt.text(j / n_v, 0.5 - 0.18, '%.3f' % (v_weight[j].item()), fontsize=25, horizontalalignment='center',
                 verticalalignment='top')
        ax.scatter(j / n_v, 0.5 - 0.22, s=1000, c=[cmap(v_w_norm(v_weight[j].item()))])

    for i in range(n_t):
        for j in range(n_v):
            _norm = norm(corr[i, j])
            plt.plot([t_loc[i], j / n_v], [1, 0.5], color=cmap(_norm), linewidth=_norm * 4, alpha=_norm)

    plt.axis('off')
    plt.axis('equal')
    plt.tight_layout()

    if save_fig:
        fig.savefig(save_fig, dpi=100)

    plt.show()
    plt.close('all')


def preprocess_video(video_path, image_resolution=224, max_frames=12):
    transform = Compose([
        Resize(image_resolution, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(image_resolution),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    video_mask = torch.zeros(max_frames, dtype=torch.long)
    max_video_length = 0

    # T x 3 x H x W
    video = torch.zeros((max_frames, 3, image_resolution, image_resolution), dtype=torch.float32)

    vreader = VideoReader(video_path, ctx=cpu(0))
    fps = int(vreader.get_avg_fps())
    f_start = 0
    f_end = int(len(vreader) - 1)
    num_frames = f_end - f_start + 1
    if num_frames > 0:
        # T x 3 x H x W
        all_pos = list(range(f_start, f_end + 1, fps))
        if len(all_pos) > max_frames:
            sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)]
        else:
            sample_pos = all_pos

        video_raw = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]
        patch_images = torch.stack([transform(img) for img in video_raw])
        slice_len = patch_images.shape[0]
        max_video_length = slice_len
        if slice_len < 1:
            pass
        else:
            video[:slice_len, ...] = patch_images
    video_mask[:max_video_length] = torch.tensor([1] * max_video_length)
    return video_raw, video, video_mask


def preprocess_text(text_input, max_words=32):
    tokenizer = SimpleTokenizer()
    words = tokenizer.tokenize(text_input)
    SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                     "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
    words = [SPECIAL_TOKEN["CLS_TOKEN"]] + words
    total_length_with_CLS = max_words - 1
    if len(words) > total_length_with_CLS:
        words = words[:total_length_with_CLS]
    words = words + [SPECIAL_TOKEN["SEP_TOKEN"]]
    input_ids = tokenizer.convert_tokens_to_ids(words)
    t_tokens = [tokenizer.decode([_t_id]) for _t_id in input_ids]
    t_tokens = [t.replace('<|startoftext|>', '[CLS]').replace('<|endoftext|>', '[SEP]').strip() for t in t_tokens]

    return t_tokens, torch.tensor(input_ids)


def main():
    # init model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from collections import namedtuple
    Config = namedtuple('Config', ["base_encoder", "agg_module", "interaction", "wti_arch"])
    model = DRL(Config("ViT-B/32", "seqTransf", "wti", 2)).to(device).eval()

    # please finetune a wti model first
    finetune_path = join(dirname(abspath(__file__)), '../ckpts/ckpt_msrvtt_wti/pytorch_model.bin.4')
    model.load_state_dict(torch.load(finetune_path, map_location="cpu"))

    video_path = "video7500.mp4"
    text_input = "a soccer team walking out on the field"

    # preprocess text and video
    t_tokens, text = preprocess_text(text_input, 32)
    text = text.to(device)
    text_mask = (text > -1).type(torch.long).to(device)

    raw_video_data, video, video_mask = preprocess_video(video_path)
    video = video.unsqueeze(0).to(device)
    video_mask = video_mask.unsqueeze(0).to(device)

    with torch.no_grad():
        text_feat = model.get_text_feat(text, text_mask)
        video_feat = model.get_video_feat(video, video_mask, False)
        t2v, v2t, _ = model.get_similarity_logits(text_feat, video_feat, text_mask, video_mask)
        print(f"t2v similarity: {t2v.item()} and v2t similarity: {v2t.item()}")
        v_frames = [img.resize([112, 112]) for img in raw_video_data]

        show_single_pair(text_feat.squeeze(0), video_feat.squeeze(0), t_tokens, v_frames, 'wti.jpg',
                         model.text_weight_fc, model.video_weight_fc)
    return True


if __name__ == "__main__":
    main()
