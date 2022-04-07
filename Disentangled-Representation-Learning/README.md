# Disentangled Representation Learning for Text-Video Retrieval
[![MSR-VTT](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/disentangled-representation-learning-for-text/video-retrieval-on-msr-vtt-1ka)](https://paperswithcode.com/sota/video-retrieval-on-msr-vtt-1ka?p=disentangled-representation-learning-for-text)
[![DiDeMo](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/disentangled-representation-learning-for-text/video-retrieval-on-didemo)](https://paperswithcode.com/sota/video-retrieval-on-didemo?p=disentangled-representation-learning-for-text)

This is a PyTorch implementation of the paper [Disentangled Representation Learning for Text-Video Retrieval](https://arxiv.org/abs/2203.07111):
<p align="center">
  <img src="demo/pipeline.png" width="800">
</p>

```
@Article{DRLTVR2022,
  author  = {Qiang Wang and Yanhao Zhang and Yun Zheng and Pan Pan and Xian-Sheng Hua},
  journal = {arXiv:2203.07111},
  title   = {Disentangled Representation Learning for Text-Video Retrieval},
  year    = {2022},
}
```

### Catalog

- [x] Setup
- [x] Fine-tuning code
- [x] Visualization demo

### Setup

#### Setup code environment
```shell
cd Disentangled-Representation-Learning
conda create -n drl python=3.9
conda activate drl
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
```

#### Download CLIP Model (as pretraining)

```shell
cd tvr/models
wget https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
# wget https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
# wget https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt
```

#### Download Datasets

```shell
cd data/MSR-VTT
wget https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip ; unzip MSRVTT.zip
mv MSRVTT/videos/all ./videos ; mv MSRVTT/annotation/MSR_VTT.json ./anns/MSRVTT_data.json
```

### Fine-tuning code

- Train on MSR-VTT 1k.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
main.py --do_train 1 --workers 8 --n_display 50 \
--epochs 5 --lr 1e-4 --coef_lr 1e-3 --batch_size 128 --batch_size_val 128 \
--anno_path data/MSR-VTT/anns --video_path data/MSR-VTT/videos --datatype msrvtt \
--max_words 32 --max_frames 12 --video_framerate 1 \
--base_encoder ViT-B/32 --agg_module seqTransf \
--interaction wti --wti_arch 2 --cdcr 3 --cdcr_alpha1 0.11 --cdcr_alpha2 0.0 --cdcr_lambda 0.001 \
--output_dir ckpts/ckpt_msrvtt_wti_cdcr
```

<p align="center">
  <img src="demo/interaction.png" width="800">
</p>

Reproduce the ablation experiments [scripts](scripts/msrvtt.sh)

<table><tbody>
<thead>
  <tr>
    <th rowspan="2">configs<br></th>
    <th rowspan="2">feature</th>
    <th rowspan="2">gpus</th>
    <th colspan="5">Text-Video</th>
    <th colspan="5">Video-Text</th>
    <th rowspan="2">train time (h)<br></th>
  </tr>
  <tr>
    <th>R@1</th>
    <th>R@5</th>
    <th>R@10</th>
    <th>MdR</th>
    <th>MnR</th>
    <th>R@1</th>
    <th>R@5</th>
    <th>R@10</th>
    <th>MdR</th>
    <th>MnR</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>CLIP4Clip</td>
    <td>ViT/B-32</td>
    <td>4</td>
    <td>42.8 </td>
    <td>72.1</td>
    <td>81.4</td>
    <td>2.0</td>
    <td>16.3</td>
    <td>44.1</td>
    <td>70.5</td>
    <td>80.5</td>
    <td>2.0</td>
    <td>11.8</td>
    <td>10.5</td>
  </tr>
  <tr>
    <td>zero-shot</td>
    <td>ViT/B-32</td>
    <td>4</td>
    <td>31.1</td>
    <td>53.7</td>
    <td>63.4</td>
    <td>4.0</td>
    <td>41.6</td>
    <td>26.5</td>
    <td>50.1</td>
    <td>61.7</td>
    <td>5.0</td>
    <td>39.9</td>
    <td>-</td>
  </tr>
  <tr>
    <td colspan="14">Interaction</td>
  </tr>
  <tr>
    <td>DP+None</td>
    <td>ViT/B-32</td>
    <td>4</td>
    <td>42.9</td>
    <td>70.6</td>
    <td>81.4</td>
    <td>2.0</td>
    <td>15.4</td>
    <td>43.0</td>
    <td>71.1</td>
    <td>81.1</td>
    <td>2.0</td>
    <td>11.8</td>
    <td>2.5</td>
  </tr>
  <tr>
    <td>DP+seqTransf</td>
    <td>ViT/B-32</td>
    <td>4</td>
    <td>42.8</td>
    <td>71.1</td>
    <td>81.1</td>
    <td>2.0</td>
    <td>15.6</td>
    <td>44.1</td>
    <td>70.9</td>
    <td>80.9</td>
    <td>2.0</td>
    <td>11.7</td>
    <td>2.6</td>
  </tr>
  <tr>
    <td>XTI+None</td>
    <td>ViT/B-32</td>
    <td>4</td>
    <td>40.5</td>
    <td>71.1</td>
    <td>82.6</td>
    <td>2.0</td>
    <td>13.6</td>
    <td>42.7</td>
    <td>70.8</td>
    <td>80.2</td>
    <td>2.0</td>
    <td>12.5</td>
    <td>14.3</td>
  </tr>
  <tr>
    <td>XTI+seqTransf</td>
    <td>ViT/B-32</td>
    <td>4</td>
    <td>42.4</td>
    <td>71.3</td>
    <td>80.9</td>
    <td>2.0</td>
    <td>15.2</td>
    <td>40.1</td>
    <td>69.2</td>
    <td>79.6</td>
    <td>2.0</td>
    <td>15.8</td>
    <td>16.8</td>
  </tr>
  <tr>
    <td>TI+seqTransf</td>
    <td>ViT/B-32</td>
    <td>4</td>
    <td>44.8</td>
    <td>73.0</td>
    <td>82.2</td>
    <td>2.0</td>
    <td>13.4</td>
    <td>42.6</td>
    <td>72.7</td>
    <td>82.8</td>
    <td>2.0</td>
    <td>9.1</td>
    <td>2.6</td>
  </tr>
  <tr>
    <td>WTI+seqTransf</td>
    <td>ViT/B-32</td>
    <td>4</td>
    <td>46.6</td>
    <td>73.4</td>
    <td>83.5</td>
    <td>2.0</td>
    <td>13.0</td>
    <td>45.4</td>
    <td>73.4</td>
    <td>81.9</td>
    <td>2.0</td>
    <td>9.2</td>
    <td>2.6</td>
  </tr>
  <tr>
    <td colspan="14">Channel DeCorrelation Regularization</td>
  </tr>
  <tr>
    <td>DP+seqTransf+CDCR</td>
    <td>ViT/B-32</td>
    <td>4</td>
    <td>43.9</td>
    <td>71.1</td>
    <td>81.2</td>
    <td>2.0</td>
    <td>15.3</td>
    <td>42.3</td>
    <td>70.3</td>
    <td>81.1</td>
    <td>2.0</td>
    <td>11.4</td>
    <td>2.6</td>
  </tr>
  <tr>
    <td>TI+seqTransf+CDCR</td>
    <td>ViT/B-32</td>
    <td>4</td>
    <td>45.8</td>
    <td>73.0</td>
    <td>81.9</td>
    <td>2.0</td>
    <td>12.8</td>
    <td>43.3</td>
    <td>71.8</td>
    <td>82.7</td>
    <td>2.0</td>
    <td>8.9</td>
    <td>2.6</td>
  </tr>
  <tr>
    <td>WTI+seqTransf+CDCR</td>
    <td>ViT/B-32</td>
    <td>4</td>
    <td>47.6</td>
    <td>73.4</td>
    <td>83.3</td>
    <td>2.0</td>
    <td>12.8</td>
    <td>45.1</td>
    <td>72.9</td>
    <td>83.5</td>
    <td>2.0</td>
    <td>9.2</td>
    <td>2.6</td>
  </tr>
</tbody>
</table>

Note: the performances are slight boosts due to new hyperparameters.


### Visualization demo

Run our visualization demo using [matplotlib](demo/show_wti.py) (no GPU needed):
<p align="center">
  <img src="demo/wti.png" width="400">
</p>

### License

See [LICENSE](LICENSE) for details.


### Acknowledgments
Our code is partly based on [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip).
