# PANet
## PANet: LiDAR Panoptic Segmentation with Sparse Instance Proposal and Aggregation

This repository provides the official implementation for PANet in the following paper. [[arxiv paper]](https://arxiv.org/abs/2306.15348)

## Requirements
- easydict
- numba
- numpy
- pyyaml
- python=3.7
- scikit-learn
- scipy
- [spconv](https://github.com/traveller59/spconv)=1.1
- tensorboard=2.7.0
- torch=1.6
- torchvision=0.7.0
- [torch-cluster](https://github.com/rusty1s/pytorch_cluster)=1.6.0
- [torch-scatter](https://github.com/rusty1s/pytorch_scatter)=2.0.8
- tqdm

## Data Preparation
Please download the [SemanticKITTI](http://www.semantic-kitti.org/dataset.html#overview) dataset to the folder `data` and the structure of the folder should look like:

```
./
├── 
├── ...
└── data/
    ├──sequences
        ├── 00/           
        │   ├── velodyne/	
        |   |	├── 000000.bin
        |   |	├── 000001.bin
        |   |	└── ...
        │   └── labels/ 
        |       ├── 000000.label
        |       ├── 000001.label
        |       └── ...
        ├── 08/ # for validation
        ├── 11/ # 11-21 for testing
        └── 21/
	        └── ...
```

## Getting Started
The training pipeline of our PANet consists of two steps: 1) semantic segmentation training following [GASN](https://github.com/ItIsFriday/PcdSeg); 2) instance aggregation training. The first step give us the semantic backbone. Then our learning-free sparse proposal module (SIP) can be directly used to group instances. The last step gives our instance aggregation module (IA), which impoves the segmentation performance of large objects. We provide the corresponding pretrained model of each step. The training and inferencing details are further explained in this section. 

All the slurm and pytorch version of scripts are tested and should work well. 

### Pretrained Models

#### PANet	
If you wish to use our pretrained models, remember to create a new folder `weights` and put all the downloaded models there.
| Step | Module | PQ | PQ (Truck) | mIoU| Download Link                                      |
| --- | ---- |----|----|---------------|------------------------------------- |
| 1 | +SIP |61.5 | 61.2 | 68.1 |[kitti_backbone_v2.pth](https://drive.google.com/file/d/10Aqexb-9LC5oFChFYEvwlFOqHew-oDWY/view?usp=sharing) |
| 2 | +SIP, +IA | 61.7 | 64.3 | 68.1 | [kitti_panet.pth](https://drive.google.com/file/d/1gaKbyx6PHVFJYUUxvF04sZBPqSwDrY3d/view?usp=sharing) |

### Inferencing with the Pretrained Models
We provide inferencing scripts for our PANet.

#### Backbone
Our semantic backbone combines the sparse 3D CNN and 2D CNN. You can train the semantic backbone using the code released by [GASN](https://github.com/ItIsFriday/PcdSeg).


#### SIP
Before using our SIP module for inference, please make sure you have downloaded the pretrained semantic backbone model (of step 1) or put the models trained by yourself (in step 1) to `./weights`. For validaton, you can use the following command. Feel free to play around with different parameter settings in `./cfgs/sip.yaml`.
```
$ python cfg_train.py --cfg cfgs/sip.yaml --onlyval
```

#### PANet
Before inferencing with our full model PANet, remember to download the pretrained model (of step 2) or put the model trained by yourself (in step 2) to `./weights` and make sure you pass the right path to `--pretrained_ckpt` option. For validation, you can use the following command.
```
$ python cfg_train.py --cfg cfgs/panet.yaml --onlyval --pretrained_ckpt </path/to/kitti_panet.pth>
```
For testing on SemanticKITTI test set, we used the ones that performed best on the validation set during training in step 2. you can use the following command to obtain the predictions.
```
$ python cfg_train.py --cfg cfgs/panet.yaml --onlytest --pretrained_ckpt </path/to/kitti_panet.pth>
```
You can use `--log_dir` and `--tag` to pass the path for desired output folder.

### Training from scratch
#### 1. Semantic segmentation training
The training code for the semantic backbone has not been incorporated yet. For now, please train the semantic backbone using the code released by [GASN](https://github.com/ItIsFriday/PcdSeg) or download the step 1 pretrained model using the above link.
#### 2. Instance aggregation training
The training scripts of step 2 could be found in `./scripts/*_train.sh`. Before using the training scripts of this part, please download the pretrained model (of step 1) to folder `./weights` or put the model trained (in step 1) to `./weights`. You could experiment with different parameter settings in `./cfgs/panet.yaml`. For training with a single GPU, you can also use the following command.
```
$ python cfg_train.py --cfg cfgs/panet.yaml --log_dir </path/to/logs> --tag <panet>
```

## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- ## Citation

If you find our work useful in your research, please consider citing the following paper:

```

``` -->

## Acknowledgments
In our implementation, we refer to the following open-source databases:
- [spconv](https://github.com/traveller59/spconv)
- [DSNet](https://github.com/hongfz16/DS-Net)
- [GASN](https://github.com/ItIsFriday/PcdSeg)
