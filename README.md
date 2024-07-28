# ISSGA-Stereo 


Pytorch implementation of the paper  "Inter-Scale Similarity Guided Cost Aggregation for Stereo Matching". This repo is keeping update. 

** I've been involved in a car accident recently. As a result, I have sustained a fracture in my right hand. This unfortunate incident is expected to impact the timeline for the upcoming code updates. 😭**

> For the core code, please refer to  ./models./issga.py

## Training

    python train.py --maxdisp 384 --batchsize 6 --database data --savemodel ./checkpoints  --epochs 30 

## Evaluation

### Sceneflow
    CUDA_VISIBLE_DEVICES=0 python test_sceneflow_raw.py --maxdisp 192 --database ./data --loadmodel  "./checkpoints/sceneflow.tar"

## Environment

- NVIDIA RTX 3090
- Python 3.8
- Pytorch 1.19

## Dependencies
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c nvidia
    pip install opencv-python
    pip install scikit-image
    pip install tensorboardX
    pip install matplotlib 
 
## Required Data
To evaluate/train ISSGA-Stereo, you will need to download the required datasets. 
* [Scene Flow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
* [Middlebury](https://vision.middlebury.edu/stereo/submit3/)
* [ETH3D](https://www.eth3d.net/datasets#low-res-two-view-test-data)


```
├── /data
    ├── sceneflow
        ├── frames_finalpass
        ├── disparity
    ├── KITTI
        ├── KITTI_2012
            ├── training
            ├── testing
            ├── vkitti
        ├── KITTI_2015
            ├── training
            ├── testing
            ├── vkitti
    ├── Middlebury
        ├── trainingH
        ├── trainingH_GT
        ├── trainingQ
        ├── trainingQ_GT
    ├── ETH3D
        ├── two_view_training
        ├── two_view_training_gt

```


## Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex
@inproceedings{issga-stereo,
  title={Inter-Scale Similarity Guided Cost Aggregation for
Stereo Matching},
  author={Pengxiang Li, Chengtang Yao, Yunde Jia, and Yuwei Wu},
  year={2023}
}
```

# Acknowledgements

This project is heavily based on [HSM-Net](https://github.com/gengshan-y/high-res-stereo) and [CF-Net](https://github.com/gallenszl/CFNet), we thank the original authors for their excellent work.
