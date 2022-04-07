# Self-supervised Video Representation Learning by Context and Motion Decoupling (CVPR 2021)
Lianghua Huang, Yu Liu, Bin Wang, Pan Pan, Yinghui Xu, Rong Jin <br/>
In CVPR, 2021. [[Paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Huang_Self-Supervised_Video_Representation_Learning_by_Context_and_Motion_Decoupling_CVPR_2021_paper.pdf).

## Introduction
This work presents a self-supervised video representation learning framework
that explicitly decouples the context and motion supervision in the pretext task.
We exploit I-frames and motion vectors in compressed videos as the efficient supervision sources. Significant improvements are achieved on downstream tasks of action recognition and
video retrieval.

![CMD](data/framework.png)

## Datasets
- Download [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php) dataset

```
cd data
sh get_ucf101_data.sh
```

- Install [FFmpeg] (https://github.com/FFmpeg/FFmpeg.git)
- Use FFmpeg to reencode the videos into a mpeg4 format. 

```
sh reencode.sh ucf101/UCF-101/ ucf101/UCF-101_rawvideo/
```
We use the scripts from CoViAR. Please go to [CoViAR GETTING_STARTED](https://github.com/chaoyuaw/pytorch-coviar/blob/master/GETTING_STARTED.md) for details.


By default, the dataset will be saved with the following structure:
```
ucf101
  ├── UCF-101/
  ├── UCF-101_rawvideo/
  └── ucfTrainTestlist/
```

## Data loader
Modify `ops/coviar_loader/setup.py` to use your FFmpeg path, build and install [CoViAR](https://github.com/chaoyuaw/pytorch-coviar).
```
cd ops/coviar_loader && sh install.sh
```

## Requirements
The required packages include: Pytorch 1.8.1, torchvision 0.9.1, decord 0.4.0, Pillow, opencv-python.
 
You can also setup the environment using the following scripts:
```
cd Context-Motion-Decoupling
conda create -n cmd python=3.7
conda activate cmd
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
```

## Usage
Pretrain R(2+1)D-26 on UCF-101 dataset:
```
sh pretrain.sh
```
Finetune R(2+1)D-26 on UCF-101 dataset:
```
sh finetune.sh
```

You can change the configuration in `cfg/ucf_config.py` and `cfg/finetune_ucf_config.py`, respectively.

## Citation
If you use our work, please cite:
```BibTeX
@inproceedings{huang2021self,
  title={Self-supervised video representation learning by context and motion decoupling},
  author={Huang, Lianghua and Liu, Yu and Wang, Bin and Pan, Pan and Xu, Yinghui and Jin, Rong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13886--13895},
  year={2021}
}
```

## Acknowledgment
We use [pytorch-coviar](https://github.com/chaoyuaw/pytorch-coviar) for the compressed video data loader.
